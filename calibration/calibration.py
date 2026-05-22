"""
Calibration System
  - Initial: 9-point or 16-point gaze target calibration
  - Online: Kalman-based adaptive calibration during ride
  - Kappa estimation: minimize reprojection error over calibration points
  - Polynomial mapping: raw gaze → calibrated gaze (backup to neural model)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.optimize import minimize
import cv2


@dataclass
class CalibrationPoint:
    target_az_deg: float        # where the rider was looking (known)
    target_el_deg: float
    raw_az_deg: float           # what the geometric model estimated
    raw_el_deg: float
    timestamp: float
    confidence: float


@dataclass
class CalibrationResult:
    kappa_h: float              # personalized kappa horizontal (deg)
    kappa_v: float              # personalized kappa vertical (deg)
    poly_coeffs: np.ndarray     # (N_coeffs,) polynomial mapping coefficients
    rmse_deg: float             # calibration RMSE
    n_points: int
    valid: bool


class InitialCalibration:
    """
    Screen-based or scene-based initial calibration.

    Protocol:
      - Display N targets at known screen positions (converted to gaze angles)
      - Collect 200ms of stable gaze per target (discard first 300ms after target appears)
      - Fit kappa angles + polynomial correction

    Polynomial mapping (5th-order):
      calibrated_az = Σ c_ij * raw_az^i * raw_el^j  (i+j ≤ 5)
      calibrated_el = Σ d_ij * raw_az^i * raw_el^j
    """

    TARGET_PATTERNS = {
        "9_point":  [(x, y) for x in [-1, 0, 1] for y in [-1, 0, 1]],
        "16_point": [(x, y) for x in [-1.5, -0.5, 0.5, 1.5] for y in [-1.5, -0.5, 0.5, 1.5]],
        "5_point":  [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)],
    }

    def __init__(
        self,
        pattern: str = "9_point",
        fov_h_deg: float = 60.0,   # horizontal FoV of scene
        fov_v_deg: float = 40.0,   # vertical FoV of scene
        collect_duration_ms: float = 500.0,
        stability_thresh_deg: float = 1.5,  # max std for stable gaze
    ):
        self.pattern = self.TARGET_PATTERNS[pattern]
        self.fov_h   = fov_h_deg
        self.fov_v   = fov_v_deg
        self.collect_duration_ms = collect_duration_ms
        self.stability_thresh    = stability_thresh_deg

        self.points: List[CalibrationPoint] = []
        self._current_target_idx = 0
        self._collecting = False
        self._buffer: List[Tuple[float, float, float]] = []  # (az, el, t)

    @property
    def n_targets(self):
        return len(self.pattern)

    @property
    def current_target_angles(self) -> Optional[Tuple[float, float]]:
        if self._current_target_idx >= len(self.pattern):
            return None
        nx, ny = self.pattern[self._current_target_idx]
        az = nx * (self.fov_h / 2.0)
        el = -ny * (self.fov_v / 2.0)  # y inverted (up = positive elevation)
        return az, el

    def start_collection(self):
        self._buffer.clear()
        self._collecting = True

    def add_gaze_sample(
        self, az: float, el: float, timestamp: float, confidence: float
    ) -> bool:
        """
        Returns True when enough stable samples collected for current target.
        """
        if not self._collecting or confidence < 0.5:
            return False

        self._buffer.append((az, el, timestamp))

        # Check duration
        if len(self._buffer) < 2:
            return False
        duration_ms = (self._buffer[-1][2] - self._buffer[0][2]) * 1000.0
        if duration_ms < self.collect_duration_ms:
            return False

        # Check stability
        azs = np.array([s[0] for s in self._buffer])
        els = np.array([s[1] for s in self._buffer])
        if azs.std() > self.stability_thresh or els.std() > self.stability_thresh:
            # Not stable — slide window
            self._buffer.pop(0)
            return False

        # Accept this calibration point
        target = self.current_target_angles
        if target is None:
            return False

        self.points.append(CalibrationPoint(
            target_az_deg=target[0],
            target_el_deg=target[1],
            raw_az_deg=float(azs.mean()),
            raw_el_deg=float(els.mean()),
            timestamp=timestamp,
            confidence=confidence,
        ))
        self._collecting = False
        self._current_target_idx += 1
        self._buffer.clear()
        return True

    def is_complete(self) -> bool:
        return self._current_target_idx >= len(self.pattern)

    def compute_calibration(self) -> CalibrationResult:
        """
        Fit kappa angles + polynomial correction from collected points.
        """
        if len(self.points) < 4:
            return CalibrationResult(
                kappa_h=5.0, kappa_v=1.5,
                poly_coeffs=np.zeros(12),
                rmse_deg=999.0, n_points=len(self.points), valid=False
            )

        raw = np.array([[p.raw_az_deg, p.raw_el_deg] for p in self.points])
        target = np.array([[p.target_az_deg, p.target_el_deg] for p in self.points])

        # Step 1: Optimize kappa angles
        def kappa_residuals(kappa_vec):
            kh, kv = kappa_vec
            corrected = raw.copy()
            corrected[:, 0] += kh
            corrected[:, 1] += kv
            return np.mean((corrected - target) ** 2)

        result = minimize(kappa_residuals, x0=[5.0, 1.5], method="Nelder-Mead",
                          options={"xatol": 0.01, "fatol": 0.001})
        kappa_h, kappa_v = result.x

        # Step 2: Polynomial correction (after kappa)
        kappa_corrected = raw.copy()
        kappa_corrected[:, 0] += kappa_h
        kappa_corrected[:, 1] += kappa_v

        poly_coeffs_az = self._fit_polynomial(
            kappa_corrected[:, 0], kappa_corrected[:, 1], target[:, 0]
        )
        poly_coeffs_el = self._fit_polynomial(
            kappa_corrected[:, 0], kappa_corrected[:, 1], target[:, 1]
        )
        poly_coeffs = np.concatenate([poly_coeffs_az, poly_coeffs_el])

        # Compute RMSE
        pred_az = self._apply_polynomial(
            kappa_corrected[:, 0], kappa_corrected[:, 1], poly_coeffs_az
        )
        pred_el = self._apply_polynomial(
            kappa_corrected[:, 0], kappa_corrected[:, 1], poly_coeffs_el
        )
        residuals = np.sqrt((pred_az - target[:, 0])**2 + (pred_el - target[:, 1])**2)
        rmse = float(residuals.mean())

        return CalibrationResult(
            kappa_h=float(kappa_h),
            kappa_v=float(kappa_v),
            poly_coeffs=poly_coeffs,
            rmse_deg=rmse,
            n_points=len(self.points),
            valid=rmse < 3.0,  # accept if RMSE < 3 degrees
        )

    def _build_poly_features(self, az: np.ndarray, el: np.ndarray, max_order: int = 3):
        """Build polynomial feature matrix up to given order"""
        features = []
        for i in range(max_order + 1):
            for j in range(max_order + 1 - i):
                features.append((az ** i) * (el ** j))
        return np.column_stack(features)

    def _fit_polynomial(self, az: np.ndarray, el: np.ndarray, target: np.ndarray) -> np.ndarray:
        X = self._build_poly_features(az, el)
        coeffs, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
        return coeffs

    def _apply_polynomial(self, az: np.ndarray, el: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
        X = self._build_poly_features(az, el)
        return X @ coeffs


class OnlineCalibrationAdapter:
    """
    Adaptive online calibration using natural fixations as pseudo-calibration points.

    Strategy:
      During riding, when rider looks at a high-confidence scene object
      (e.g., a road sign, traffic light) that can be localized in scene image,
      we get a pseudo ground-truth gaze position.

    Kalman filter update:
      State: [kappa_h, kappa_v]
      Measurement: difference between scene-projected gaze and known object center
      Q (process noise): small (kappa drifts slowly with fatigue/sweat)
      R (measurement noise): depends on scene detection confidence
    """

    def __init__(self, initial_kappa: Tuple[float, float] = (5.0, 1.5)):
        self.kappa = np.array(initial_kappa, dtype=np.float64)

        # Kalman state: [kappa_h, kappa_v]
        self.P = np.eye(2) * 1.0            # state covariance
        self.Q = np.eye(2) * 0.001          # process noise (drifts slowly)
        self.R = np.eye(2) * 0.5            # measurement noise (degrees)
        self.F = np.eye(2)                   # state transition (identity: kappa is constant)
        self.H = np.eye(2)                   # observation model (direct)

        self._n_updates = 0
        self._update_history: List[Tuple[float, float]] = []

    def update(
        self,
        raw_gaze: np.ndarray,          # (2,) [az, el] raw geometric estimate
        true_gaze: np.ndarray,         # (2,) [az, el] from scene detection
        detection_confidence: float,
    ):
        """Kalman update step"""
        if detection_confidence < 0.6:
            return

        # Predicted measurement: current kappa correction applied to raw
        z_pred = raw_gaze + self.kappa        # what we predict scene gaze should be
        z_meas = true_gaze                    # actual gaze from scene

        # Measurement noise scaled by detection confidence
        R_scaled = self.R / detection_confidence

        # Kalman gain
        S = self.P + R_scaled                # innovation covariance
        K = self.P @ np.linalg.inv(S)        # Kalman gain

        # Innovation
        innovation = z_meas - z_pred         # residual

        # State update: kappa correction
        kappa_correction = K @ innovation
        self.kappa = self.kappa + kappa_correction

        # Covariance update
        self.P = (np.eye(2) - K) @ self.P + self.Q

        # Clip kappa to reasonable range
        self.kappa = np.clip(self.kappa, [-15.0, -8.0], [15.0, 8.0])

        self._n_updates += 1
        self._update_history.append(tuple(self.kappa.tolist()))

    def get_kappa(self) -> Tuple[float, float]:
        return float(self.kappa[0]), float(self.kappa[1])

    def apply(self, raw_gaze: np.ndarray) -> np.ndarray:
        """Apply current kappa estimate to raw gaze"""
        return raw_gaze + self.kappa