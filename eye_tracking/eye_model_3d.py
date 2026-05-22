import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from eye_tracking.feature_extractor import EyeFeatures


@dataclass
class EyeModel3D:
    # Eyeball geometry (in mm, real-world)
    R_cornea: float = 7.8           # cornea sphere radius
    R_eyeball: float = 12.0         # eyeball radius
    cornea_center_offset: float = 4.2  # distance from eyeball center to cornea center (along optical axis)
    n1: float = 1.0                 # refractive index air
    n2: float = 1.336               # refractive index aqueous humor

    # Camera intrinsics
    fx: float = 600.0
    fy: float = 600.0
    cx: float = 320.0
    cy: float = 240.0

    # Kappa/alpha angles (visual axis correction) — personalized during calibration
    kappa_horizontal: float = 5.0   # degrees (nasal offset)
    kappa_vertical: float = 1.5     # degrees (superior offset)


@dataclass
class GazeRay3D:
    optical_axis: np.ndarray    # (3,) unit vector — optical axis in camera coords
    visual_axis: np.ndarray     # (3,) unit vector — visual axis (kappa-corrected)
    eyeball_center: np.ndarray  # (3,) in camera coords (mm)
    cornea_center: np.ndarray   # (3,) in camera coords (mm)
    confidence: float
    method: str                 # "glint_model" or "pupil_only"


class EyeModelEstimator:
    """
    Estimates 3D gaze from eye features using corneal glint model.

    Camera model:
        p_img = K @ p_cam / p_cam_z  (pinhole)

    Optical axis estimation:
        Given pupil center p and N glint centers g_i (from N LEDs at known 3D positions L_i):
        1. Back-project p to a ray
        2. For each glint: compute reflection off cornea sphere
        3. Minimize reprojection error to estimate eyeball center E and cornea center C
        4. Optical axis = (C - E) / ||C - E||

    Fallback (no glints): use pupil position directly (less accurate).
    """

    def __init__(self, model: EyeModel3D, led_positions_3d: np.ndarray):
        """
        led_positions_3d: (N, 3) LED positions in camera coordinate frame (mm)
        """
        self.model = model
        self.led_positions_3d = led_positions_3d  # (N, 3)
        self.K = np.array([
            [model.fx, 0, model.cx],
            [0, model.fy, model.cy],
            [0, 0, 1],
        ], dtype=np.float64)
        self.K_inv = np.linalg.inv(self.K)

        # Kappa rotation matrix (visual axis correction)
        self._build_kappa_rotation()

    def _build_kappa_rotation(self):
        kh = np.deg2rad(self.model.kappa_horizontal)
        kv = np.deg2rad(self.model.kappa_vertical)
        Ry = np.array([[np.cos(kh), 0, np.sin(kh)],
                       [0, 1, 0],
                       [-np.sin(kh), 0, np.cos(kh)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(kv), -np.sin(kv)],
                       [0, np.sin(kv), np.cos(kv)]])
        self.R_kappa = Ry @ Rx

    def update_kappa(self, kappa_h: float, kappa_v: float):
        self.model.kappa_horizontal = kappa_h
        self.model.kappa_vertical   = kappa_v
        self._build_kappa_rotation()

    def estimate(self, features: EyeFeatures) -> Optional[GazeRay3D]:
        if not features.valid or features.pupil is None:
            return None

        pupil_px = features.pupil.center  # (2,)

        # Try glint-based 3D model first
        if (features.glints is not None and
                features.glints.valid_mask.sum() >= 2):
            result = self._glint_based_estimation(pupil_px, features.glints)
            if result is not None:
                return result

        # Fallback: pupil-only ray casting
        return self._pupil_only_estimation(pupil_px)

    def _glint_based_estimation(self, pupil_px: np.ndarray, glints) -> Optional[GazeRay3D]:
        """
        Guestrin-Eizenman model implementation.
        Uses: pupil image point + at least 2 glint image points + LED 3D positions.
        """
        valid_idx = np.where(glints.valid_mask)[0]
        if len(valid_idx) < 2:
            return None

        # Back-project pupil to unit ray
        p_ray = self._backproject(pupil_px)  # (3,)

        # For each valid glint, set up reflection constraint
        # Corneal glint constraint: for LED at L_i, glint at g_i
        # The reflection condition: g_i lies on the cornea sphere,
        # L_i, g_i, and camera are related by Snell's law of reflection.
        # Simplified: the cornea center C lies such that:
        #   ||g_i_3d - C|| = R_cornea  (on sphere)
        #   and the reflected ray passes through camera origin

        glint_rays = []
        led_pts = []
        for idx in valid_idx:
            g_px = glints.centers[idx]
            g_ray = self._backproject(g_px)
            glint_rays.append(g_ray)
            led_pts.append(self.led_positions_3d[idx])

        glint_rays = np.array(glint_rays)  # (N, 3)
        led_pts    = np.array(led_pts)     # (N, 3)

        # Estimate cornea center via least-squares
        # For each glint ray d_i from camera origin O (=0):
        #   Point on ray: t_i * d_i
        #   Sphere constraint: ||t_i * d_i - C||^2 = R^2
        #
        # Also: glint point is midpoint direction between camera→glint and LED→glint
        # C = intersection of bisectors (approximate)

        C = self._estimate_cornea_center(glint_rays, led_pts)
        if C is None:
            return None

        # Eyeball center along optical axis behind cornea center
        # Initial estimate: pupil ray direction defines optical axis
        p_ray_norm = p_ray / np.linalg.norm(p_ray)

        # Project C onto pupil ray to get depth estimate
        t_C = np.dot(C, p_ray_norm)
        E = C - self.model.cornea_center_offset * p_ray_norm

        optical_axis = (C - E)
        optical_axis = optical_axis / (np.linalg.norm(optical_axis) + 1e-9)

        # Visual axis = kappa-rotated optical axis
        visual_axis = self.R_kappa @ optical_axis
        visual_axis = visual_axis / (np.linalg.norm(visual_axis) + 1e-9)

        conf = float(glints.valid_mask.sum()) / len(glints.valid_mask)

        return GazeRay3D(
            optical_axis=optical_axis,
            visual_axis=visual_axis,
            eyeball_center=E,
            cornea_center=C,
            confidence=conf,
            method="glint_model",
        )

    def _estimate_cornea_center(
            self, glint_rays: np.ndarray, led_pts: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Estimate cornea center C given glint directions and LED positions.

        Each glint g_i = t_i * d_i (on ray from camera origin).
        Cornea sphere: ||g_i - C||^2 = R^2
        Also: incident ray (L_i → g_i) and reflected ray (g_i → camera)
              bisector points toward C (law of reflection, sphere normal).

        We solve the linear system:
          For pairs (i, j):
            ||g_i - C||^2 - ||g_j - C||^2 = 0
          → 2*(g_j - g_i)^T * C = ||g_j||^2 - ||g_i||^2
        First we get t_i from each ray depth (initial estimate: use LED-camera geometry).
        """
        N = len(glint_rays)
        R = self.model.R_cornea

        # Initial depth estimates for glints using the sphere constraint
        # ||t_i * d_i - C||^2 = R^2 with C unknown → iterative
        # Bootstrap: assume C is along mean glint ray at nominal depth
        nominal_depth = 40.0  # mm from camera — typical eye-camera distance
        t_init = np.array([
            nominal_depth / (d[2] + 1e-9) for d in glint_rays
        ])
        glint_pts = glint_rays * t_init[:, np.newaxis]  # (N, 3)

        # Linear least squares: 2*(g_j - g_i)^T * C = ||g_j||^2 - ||g_i||^2
        A_rows = []
        b_rows = []
        for i in range(N):
            for j in range(i + 1, N):
                row = 2.0 * (glint_pts[j] - glint_pts[i])
                rhs = np.dot(glint_pts[j], glint_pts[j]) - np.dot(glint_pts[i], glint_pts[i])
                A_rows.append(row)
                b_rows.append(rhs)

        if len(A_rows) < 1:
            return None

        A = np.array(A_rows)
        b = np.array(b_rows)

        # Regularized least squares
        try:
            C, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None

        # Refine t_i given C
        for _ in range(3):  # 3 Newton iterations
            for i in range(N):
                d = glint_rays[i]
                # Solve: ||t*d - C||^2 = R^2
                a_ = np.dot(d, d)
                b_ = -2 * np.dot(d, C)
                c_ = np.dot(C, C) - R ** 2
                disc = b_ ** 2 - 4 * a_ * c_
                if disc < 0:
                    continue
                t1 = (-b_ + np.sqrt(disc)) / (2 * a_)
                t2 = (-b_ - np.sqrt(disc)) / (2 * a_)
                # Take the one closer to initial estimate
                t_init[i] = t1 if abs(t1 - t_init[i]) < abs(t2 - t_init[i]) else t2
            glint_pts = glint_rays * t_init[:, np.newaxis]

            # Re-solve for C
            A_rows.clear()
            b_rows.clear()
            for i in range(N):
                for j in range(i + 1, N):
                    row = 2.0 * (glint_pts[j] - glint_pts[i])
                    rhs = np.dot(glint_pts[j], glint_pts[j]) - np.dot(glint_pts[i], glint_pts[i])
                    A_rows.append(row)
                    b_rows.append(rhs)
            if A_rows:
                A = np.array(A_rows)
                b_ = np.array(b_rows)
                try:
                    C, _, _, _ = np.linalg.lstsq(A, b_, rcond=None)
                except np.linalg.LinAlgError:
                    break

        # Sanity check: C should be in front of camera at reasonable depth
        if C[2] < 10.0 or C[2] > 120.0:
            return None

        return C

    def _pupil_only_estimation(self, pupil_px: np.ndarray) -> GazeRay3D:
        """Fallback: back-project pupil center to get approximate gaze ray"""
        ray = self._backproject(pupil_px)
        ray = ray / (np.linalg.norm(ray) + 1e-9)
        visual_axis = self.R_kappa @ ray
        visual_axis = visual_axis / (np.linalg.norm(visual_axis) + 1e-9)
        return GazeRay3D(
            optical_axis=ray,
            visual_axis=visual_axis,
            eyeball_center=np.array([0.0, 0.0, 40.0]),  # placeholder
            cornea_center=np.array([0.0, 0.0, 36.0]),
            confidence=0.3,
            method="pupil_only",
        )

    def _backproject(self, px: np.ndarray) -> np.ndarray:
        """Convert image point to unit ray in camera frame"""
        p_hom = np.array([px[0], px[1], 1.0])
        ray = self.K_inv @ p_hom
        return ray / (np.linalg.norm(ray) + 1e-9)