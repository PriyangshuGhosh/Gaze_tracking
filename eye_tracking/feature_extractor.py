import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class PupilFeature:
    center: np.ndarray          # (2,) subpixel xy
    axes: np.ndarray            # (2,) semi-axes of fitted ellipse
    angle: float                # ellipse rotation angle
    confidence: float           # 0-1 detection confidence
    contour_area: float
    ellipse_fit_error: float    # mean dist of contour pts to ellipse


@dataclass
class GlintFeature:
    centers: np.ndarray         # (N, 2) up to 4 glints
    valid_mask: np.ndarray      # (N,) bool — which glints are valid
    led_assignment: np.ndarray  # (N,) int — which LED each glint maps to


@dataclass
class EyeFeatures:
    pupil: Optional[PupilFeature]
    glints: Optional[GlintFeature]
    iris_radius: float          # pixels — used for normalization
    eyelid_closure: float       # 0 (closed) to 1 (open) — PERCLOS proxy
    timestamp: float
    frame_id: int
    valid: bool


class PupilDetector:
    """
    Pipeline:
      1. Gaussian blur → contrast stretch
      2. Adaptive threshold (dark region prior — pupil is darkest)
      3. Morphological cleanup
      4. Contour filter by: area, aspect ratio, convexity
      5. Ellipse fit → subpixel center via intensity-weighted centroid
    """

    def __init__(self, cfg: dict):
        self.min_pupil_area = cfg.get("min_pupil_area", 300)
        self.max_pupil_area = cfg.get("max_pupil_area", 8000)
        self.min_aspect_ratio = cfg.get("min_aspect_ratio", 0.4)
        self.max_ellipse_error = cfg.get("max_ellipse_error", 1.5)  # px
        self.dark_percentile = cfg.get("dark_percentile", 20)       # use darkest 20%
        self.blur_ksize = cfg.get("blur_ksize", 5)

        # Morphological kernels
        self.morph_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.morph_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: np.ndarray) -> Optional[PupilFeature]:
        """
        frame: uint8 grayscale eye ROI
        Returns PupilFeature or None
        """
        blurred = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), 0)

        # Dynamic threshold: use dark percentile as upper bound
        thresh_val = int(np.percentile(blurred, self.dark_percentile))
        thresh_val = np.clip(thresh_val + 15, 20, 80)

        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

        # Morphological ops: fill holes, remove noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.morph_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  self.morph_open)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return None

        best = None
        best_score = -1.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (self.min_pupil_area <= area <= self.max_pupil_area):
                continue
            if len(cnt) < 5:
                continue

            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (ma, mi), angle = ellipse

            if mi < 1e-3:
                continue
            aspect = mi / ma
            if aspect < self.min_aspect_ratio:
                continue

            # Convexity check
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            if convexity < 0.85:
                continue

            # Ellipse fit error
            fit_error = self._ellipse_fit_error(cnt, ellipse)
            if fit_error > self.max_ellipse_error:
                continue

            # Score: area + aspect + convexity - fit_error
            score = aspect * convexity * (1.0 / (1.0 + fit_error)) * np.sqrt(area)
            if score > best_score:
                best_score = score
                best = (cnt, ellipse, area, fit_error, aspect)

        if best is None:
            return None

        cnt, ellipse, area, fit_error, aspect = best
        (cx, cy), (ma, mi), angle = ellipse

        # Subpixel refinement: intensity-weighted centroid in ellipse mask
        cx_sub, cy_sub = self._subpixel_refine(frame, (cx, cy), max(ma, mi) / 2)

        confidence = np.clip(aspect * (1.0 / (1.0 + fit_error)), 0.0, 1.0)

        return PupilFeature(
            center=np.array([cx_sub, cy_sub]),
            axes=np.array([ma / 2, mi / 2]),
            angle=angle,
            confidence=float(confidence),
            contour_area=float(area),
            ellipse_fit_error=float(fit_error),
        )

    def _ellipse_fit_error(self, contour: np.ndarray, ellipse: tuple) -> float:
        """Mean algebraic distance of contour pts to fitted ellipse"""
        pts = contour.reshape(-1, 2).astype(np.float32)
        (cx, cy), (ma, mi), angle = ellipse
        a, b = ma / 2, mi / 2
        if a < 1e-3 or b < 1e-3:
            return 999.0
        theta = np.deg2rad(angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        x_rot = cos_t * dx + sin_t * dy
        y_rot = -sin_t * dx + cos_t * dy
        err = np.abs((x_rot / a) ** 2 + (y_rot / b) ** 2 - 1)
        return float(np.mean(err))

    def _subpixel_refine(self, frame: np.ndarray, center: tuple, radius: float) -> Tuple[float, float]:
        cx, cy = center
        r = int(radius * 1.2)
        h, w = frame.shape[:2]
        x0, x1 = max(0, int(cx - r)), min(w, int(cx + r))
        y0, y1 = max(0, int(cy - r)), min(h, int(cy + r))
        roi = frame[y0:y1, x0:x1].astype(np.float32)
        if roi.size == 0:
            return cx, cy
        # Dark region → invert for weighting
        weights = (255.0 - roi)
        weights = np.maximum(weights, 0)
        total = weights.sum()
        if total < 1e-6:
            return cx, cy
        ys, xs = np.mgrid[y0:y1, x0:x1]
        cx_sub = float((xs * weights).sum() / total)
        cy_sub = float((ys * weights).sum() / total)
        return cx_sub, cy_sub


class GlintDetector:
    """
    Detects up to N_LEDS corneal reflections (bright spots from IR LEDs).
    Uses blob detection + LED geometry prior for assignment.

    LED layout (example, 4-LED ring around camera):
        LED 0: top     (0, -d)
        LED 1: right   (+d, 0)
        LED 2: bottom  (0, +d)
        LED 3: left    (-d, 0)
    """

    def __init__(self, cfg: dict):
        self.n_leds = cfg.get("n_leds", 4)
        self.led_positions_norm = np.array(cfg.get(
            "led_positions_norm",
            [[0, -1], [1, 0], [0, 1], [-1, 0]]
        ), dtype=np.float32)  # normalized directions from camera center

        self.min_glint_area = cfg.get("min_glint_area", 5)
        self.max_glint_area = cfg.get("max_glint_area", 200)
        self.glint_thresh = cfg.get("glint_thresh", 200)   # bright threshold

        # Blob detector params
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255
        params.filterByArea = True
        params.minArea = self.min_glint_area
        params.maxArea = self.max_glint_area
        params.filterByCircularity = True
        params.minCircularity = 0.5
        params.filterByConvexity = True
        params.minConvexity = 0.7
        self.blob_detector = cv2.SimpleBlobDetector_create(params)

    def detect(self, frame: np.ndarray, pupil: Optional[PupilFeature]) -> Optional[GlintFeature]:
        """
        frame: uint8 grayscale
        pupil: used to exclude pupil region from glint search
        """
        # Threshold bright spots
        _, bright = cv2.threshold(frame, self.glint_thresh, 255, cv2.THRESH_BINARY)

        # Mask out pupil region to avoid false glints inside pupil
        if pupil is not None:
            mask = np.zeros_like(bright)
            cx, cy = int(pupil.center[0]), int(pupil.center[1])
            r = int(max(pupil.axes) * 1.5)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            bright = cv2.bitwise_and(bright, cv2.bitwise_not(mask))

        keypoints = self.blob_detector.detect(bright)

        if not keypoints:
            return GlintFeature(
                centers=np.zeros((self.n_leds, 2), dtype=np.float32),
                valid_mask=np.zeros(self.n_leds, dtype=bool),
                led_assignment=np.arange(self.n_leds, dtype=np.int32),
            )

        candidates = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)

        # Assign glints to LEDs using nearest-neighbor with LED geometry prior
        centers = np.zeros((self.n_leds, 2), dtype=np.float32)
        valid_mask = np.zeros(self.n_leds, dtype=bool)
        led_assignment = np.arange(self.n_leds, dtype=np.int32)

        if pupil is not None and len(candidates) >= 1:
            # Compute direction vectors from pupil to each glint
            dirs = candidates - pupil.center  # (K, 2)
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-6, 1.0, norms)
            dirs_norm = dirs / norms

            # Match to LED layout
            used = set()
            for led_idx, led_dir in enumerate(self.led_positions_norm):
                if len(candidates) == 0:
                    break
                # Cosine similarity to expected LED direction
                sims = dirs_norm @ led_dir  # (K,)
                # Rank by similarity, skip already used
                order = np.argsort(-sims)
                for j in order:
                    if j not in used:
                        centers[led_idx] = candidates[j]
                        valid_mask[led_idx] = True
                        led_assignment[led_idx] = led_idx
                        used.add(j)
                        break
        else:
            # No pupil reference: just take first N_LEDS glints by brightness
            for i, kp in enumerate(keypoints[:self.n_leds]):
                centers[i] = [kp.pt[0], kp.pt[1]]
                valid_mask[i] = True

        return GlintFeature(centers=centers, valid_mask=valid_mask, led_assignment=led_assignment)


class EyelidAnalyzer:
    """
    Estimates eyelid closure (PERCLOS proxy) using vertical extent
    of the visible eye region relative to iris radius.
    """

    def __init__(self, cfg: dict):
        self.edge_thresh1 = cfg.get("edge_thresh1", 30)
        self.edge_thresh2 = cfg.get("edge_thresh2", 80)

    def compute_closure(self, frame: np.ndarray, pupil: Optional[PupilFeature]) -> float:
        """Returns 0.0 (fully closed) to 1.0 (fully open)"""
        if pupil is None:
            return 0.0
        cx, cy = int(pupil.center[0]), int(pupil.center[1])
        iris_r = max(pupil.axes) * 2.5  # estimate iris from pupil
        iris_r = max(iris_r, 10)

        h, w = frame.shape[:2]
        x0 = max(0, int(cx - iris_r))
        x1 = min(w, int(cx + iris_r))
        y0 = max(0, int(cy - iris_r * 1.5))
        y1 = min(h, int(cy + iris_r * 1.5))

        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return 1.0

        edges = cv2.Canny(roi, self.edge_thresh1, self.edge_thresh2)

        # Find top and bottom eyelid edges using column projections
        proj = edges.sum(axis=1).astype(np.float32)
        if proj.max() < 1e-3:
            return 1.0

        roi_h = y1 - y0
        cy_local = cy - y0

        # Find first strong edge above and below pupil center
        above = proj[:cy_local]
        below = proj[cy_local:]

        top_edge_idx = 0
        if above.max() > 0:
            top_edge_idx = len(above) - 1 - np.argmax(above[::-1] > above.max() * 0.3)

        bot_edge_idx = roi_h - 1
        if below.max() > 0:
            bot_edge_idx = cy_local + np.argmax(below > below.max() * 0.3)

        visible_height = (bot_edge_idx - top_edge_idx)
        closure = np.clip(visible_height / (2.0 * iris_r), 0.0, 1.0)
        return float(closure)


class EyeFeatureExtractor:
    """
    Top-level extractor: runs pupil + glint + eyelid per frame.
    Designed to process at 200 FPS — no heavy ops inside.
    """

    def __init__(self, cfg: dict):
        self.pupil_detector  = PupilDetector(cfg.get("pupil", {}))
        self.glint_detector  = GlintDetector(cfg.get("glint", {}))
        self.eyelid_analyzer = EyelidAnalyzer(cfg.get("eyelid", {}))
        self.roi = cfg.get("roi", None)  # (x, y, w, h) or None for full frame

    def process(self, frame: np.ndarray, timestamp: float, frame_id: int) -> EyeFeatures:
        """
        frame: uint8 grayscale (or BGR — will be converted)
        """
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply ROI crop if specified
        if self.roi is not None:
            x, y, w, h = self.roi
            frame = frame[y:y+h, x:x+w]

        pupil   = self.pupil_detector.detect(frame)
        glints  = self.glint_detector.detect(frame, pupil)
        closure = self.eyelid_analyzer.compute_closure(frame, pupil)

        iris_radius = 0.0
        if pupil is not None:
            iris_radius = float(max(pupil.axes)) * 2.5

        valid = (pupil is not None) and (pupil.confidence > 0.4)

        return EyeFeatures(
            pupil=pupil,
            glints=glints,
            iris_radius=iris_radius,
            eyelid_closure=closure,
            timestamp=timestamp,
            frame_id=frame_id,
            valid=valid,
        )