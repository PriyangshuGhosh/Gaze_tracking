"""
Scene Mapping Module
  - Projects world-frame gaze onto scene camera(s)
  - Integrates with YOLOv8 object detections
  - Computes gazed-at object labels
  - Supports 360° multi-camera setup (front stereo + rear stereo + L/R)
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from fusion.imu_fusion import WorldGaze


@dataclass
class CameraConfig:
    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    R_cam2world: np.ndarray   # (3,3) rotation: camera → world
    t_cam2world: np.ndarray   # (3,) translation: camera position in world (mm)
    dist_coeffs: np.ndarray   # (5,) radial/tangential distortion


@dataclass
class GazedObject:
    label: str
    confidence: float
    bbox: np.ndarray          # (4,) x1, y1, x2, y2 in scene image pixels
    distance_m: float         # estimated from stereo (if available)
    gaze_overlap_ratio: float # fraction of gaze heatmap inside bbox


@dataclass
class SceneGaze:
    timestamp: float
    camera_name: str
    gaze_px: np.ndarray       # (2,) pixel coordinates on scene camera
    gaze_px_valid: bool
    gazed_objects: List[GazedObject]
    world_gaze: WorldGaze
    scene_image: Optional[np.ndarray] = None  # debug visualization


class GazeProjector:
    """
    Projects a 3D gaze ray (origin in world coords, direction in world coords)
    onto a scene camera image.

    Steps:
      1. Ray origin = rider's head position in world (approx)
      2. Gaze direction in world frame
      3. Transform ray to scene camera frame
      4. Project onto image using camera intrinsics + distortion
    """

    def __init__(self, cameras: List[CameraConfig]):
        self.cameras = {cam.name: cam for cam in cameras}

    def project_gaze(
        self,
        world_gaze: WorldGaze,
        head_pos_world: np.ndarray,  # (3,) mm
        projection_depth_m: float = 5000.0,  # project onto plane at 5m
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Returns dict: {camera_name → (2,) pixel or None if behind camera}
        """
        results = {}
        # Point on gaze ray at projection_depth
        gaze_pt_world = head_pos_world + world_gaze.gaze_dir_world * projection_depth_m

        for name, cam in self.cameras.items():
            px = self._project_point(gaze_pt_world, head_pos_world, cam)
            results[name] = px
        return results

    def _project_point(
        self,
        world_pt: np.ndarray,
        origin_world: np.ndarray,
        cam: CameraConfig,
    ) -> Optional[np.ndarray]:
        """Project a 3D world point onto camera image"""
        # World → camera: p_cam = R^T * (p_world - t_cam)
        R_w2c = cam.R_cam2world.T
        p_cam = R_w2c @ (world_pt - cam.t_cam2world)

        # Must be in front of camera
        if p_cam[2] < 10.0:
            return None

        # Perspective divide
        xn = p_cam[0] / p_cam[2]
        yn = p_cam[1] / p_cam[2]

        # Apply distortion: radial + tangential
        k1, k2, p1, p2, k3 = cam.dist_coeffs[:5]
        r2 = xn*xn + yn*yn
        r4 = r2 * r2
        r6 = r2 * r4
        radial = 1 + k1*r2 + k2*r4 + k3*r6
        xd = xn*radial + 2*p1*xn*yn + p2*(r2 + 2*xn*xn)
        yd = yn*radial + p1*(r2 + 2*yn*yn) + 2*p2*xn*yn

        # Pixel coordinates
        u = cam.fx * xd + cam.cx
        v = cam.fy * yd + cam.cy

        if not (0 <= u < cam.width and 0 <= v < cam.height):
            return None

        return np.array([u, v])

    def select_best_camera(
        self,
        world_gaze: WorldGaze,
        head_pos_world: np.ndarray,
    ) -> Optional[str]:
        """Returns name of camera best aligned with gaze direction"""
        best_name = None
        best_dot = -1.0
        gaze_dir = world_gaze.gaze_dir_world

        for name, cam in self.cameras.items():
            cam_forward = cam.R_cam2world[:, 2]  # Z-axis of camera in world
            dot = float(np.dot(gaze_dir, cam_forward))
            if dot > best_dot:
                best_dot = dot
                best_name = name

        return best_name if best_dot > 0.1 else None


class YOLOSceneIntegrator:
    """
    Integrates gaze with YOLOv8 detections on scene camera feed.
    Uses a Gaussian gaze heatmap to compute overlap with each detected object.

    Gaze heatmap sigma: ~2° FOV → compute in pixels from camera FoV.
    """

    def __init__(self, model_path: str, device: str = "cpu", gaze_sigma_deg: float = 2.0):
        self.device = device
        self.gaze_sigma_deg = gaze_sigma_deg
        self._model = None
        self._model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
        except ImportError:
            import logging
            logging.warning("ultralytics not installed — YOLO integration disabled")
            self._model = None

    def detect_and_integrate(
        self,
        frame: np.ndarray,
        gaze_px: np.ndarray,
        cam_cfg: CameraConfig,
        conf_threshold: float = 0.4,
    ) -> List[GazedObject]:
        """
        frame: (H, W, 3) BGR
        gaze_px: (2,) pixel location of gaze
        Returns list of GazedObject sorted by gaze overlap (desc)
        """
        if self._model is None:
            return []

        results = self._model(frame, conf=conf_threshold, verbose=False)
        if not results or len(results[0].boxes) == 0:
            return []

        # Build gaze heatmap
        sigma_px = self._deg_to_pixels(self.gaze_sigma_deg, cam_cfg)
        heatmap = self._gaussian_heatmap(
            frame.shape[:2], gaze_px, sigma_px
        )

        gazed_objects = []
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            label = results[0].names[int(box.cls[0])]
            det_conf = float(box.conf[0])

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Gaze overlap: sum of heatmap inside bbox / total heatmap
            roi_sum = heatmap[y1:y2, x1:x2].sum()
            total_sum = heatmap.sum()
            overlap = float(roi_sum / (total_sum + 1e-9))

            gazed_objects.append(GazedObject(
                label=label,
                confidence=det_conf,
                bbox=np.array([x1, y1, x2, y2]),
                distance_m=0.0,  # filled by StereoDepth if available
                gaze_overlap_ratio=overlap,
            ))

        gazed_objects.sort(key=lambda o: o.gaze_overlap_ratio, reverse=True)
        return gazed_objects

    def _gaussian_heatmap(
        self, shape: tuple, center: np.ndarray, sigma: float
    ) -> np.ndarray:
        H, W = shape
        ys, xs = np.mgrid[0:H, 0:W]
        hmap = np.exp(-((xs - center[0])**2 + (ys - center[1])**2) / (2 * sigma**2))
        return hmap.astype(np.float32)

    def _deg_to_pixels(self, deg: float, cam: CameraConfig) -> float:
        """Convert angular radius to pixel radius at image center"""
        focal = (cam.fx + cam.fy) / 2.0
        return focal * np.tan(np.deg2rad(deg))


class SceneMapper:
    """Top-level scene mapping: orchestrates projection + YOLO integration"""

    def __init__(
        self,
        cameras: List[CameraConfig],
        yolo_model_path: str,
        yolo_device: str = "cpu",
    ):
        self.projector = GazeProjector(cameras)
        self.yolo      = YOLOSceneIntegrator(yolo_model_path, device=yolo_device)
        self.cameras   = {cam.name: cam for cam in cameras}

        # Head position in world (updated from IMU/telemetry)
        self.head_pos_world = np.array([0.0, 1600.0, 0.0])  # ~1.6m above ground

    def process(
        self,
        world_gaze: WorldGaze,
        scene_frames: Dict[str, np.ndarray],  # {cam_name: frame}
    ) -> Optional[SceneGaze]:
        """
        world_gaze: fused gaze in world coordinates
        scene_frames: dict of scene camera frames (BGR)
        """
        # Select best camera
        best_cam = self.projector.select_best_camera(world_gaze, self.head_pos_world)
        if best_cam is None or best_cam not in scene_frames:
            return None

        # Project gaze
        projections = self.projector.project_gaze(world_gaze, self.head_pos_world)
        gaze_px = projections.get(best_cam)

        if gaze_px is None:
            return SceneGaze(
                timestamp=world_gaze.timestamp,
                camera_name=best_cam,
                gaze_px=np.array([0.0, 0.0]),
                gaze_px_valid=False,
                gazed_objects=[],
                world_gaze=world_gaze,
            )

        # YOLO integration
        cam_cfg = self.cameras[best_cam]
        frame = scene_frames[best_cam]
        gazed_objects = self.yolo.detect_and_integrate(frame, gaze_px, cam_cfg)

        return SceneGaze(
            timestamp=world_gaze.timestamp,
            camera_name=best_cam,
            gaze_px=gaze_px,
            gaze_px_valid=True,
            gazed_objects=gazed_objects,
            world_gaze=world_gaze,
        )

    def update_head_position(self, pos: np.ndarray):
        self.head_pos_world = pos