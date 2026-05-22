"""
IMU Fusion + Head Pose Compensation

Strategy:
  - Madgwick AHRS filter for absolute orientation from IMU
  - Transforms gaze ray from camera frame → head frame → world frame
  - VOR: head rotation causes compensatory eye movement
    VOR mismatch = |expected_eye_rotation - actual_eye_rotation|
  - IMU @ 200 Hz, gaze @ 200 Hz → 1:1 sync

Coordinate conventions (right-handed):
  World: X=East, Y=Up, Z=South
  Head:  X=right, Y=up, Z=backward (from rider's POV)
  Camera: Z=forward into eye
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Deque
from collections import deque
from eye_tracking.eye_model_3d import GazeRay3D


@dataclass
class IMUSample:
    timestamp: float
    accel: np.ndarray      # (3,) m/s^2 body frame
    gyro:  np.ndarray      # (3,) rad/s body frame
    mag:   Optional[np.ndarray] = None  # (3,) μT


@dataclass
class HeadPose:
    timestamp: float
    R_head2world: np.ndarray   # (3,3) rotation: head → world
    angular_velocity: np.ndarray  # (3,) rad/s in head frame
    linear_accel: np.ndarray      # (3,) gravity-free accel in world frame
    euler: np.ndarray             # (3,) roll, pitch, yaw in degrees


@dataclass
class WorldGaze:
    timestamp: float
    gaze_dir_world: np.ndarray    # (3,) unit vector in world frame
    gaze_dir_head: np.ndarray     # (3,) unit vector in head frame
    head_pose: HeadPose
    vor_mismatch: float           # magnitude of VOR violation
    confidence: float


class MadgwickAHRS:
    """
    Madgwick AHRS filter — quaternion-based complementary filter.
    Fuses accelerometer + gyroscope (+ optional magnetometer).
    O(1) per update — suitable for 200 Hz.

    q: [w, x, y, z] convention
    """

    def __init__(self, sample_rate: float = 200.0, beta: float = 0.033):
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.beta = beta   # algorithm gain (0.033 is Madgwick's recommendation)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # initial quaternion (identity)

    def update_imu(self, gyro: np.ndarray, accel: np.ndarray):
        """Update with gyro (rad/s) + accel (m/s^2, not normalized)"""
        q = self.q
        gx, gy, gz = gyro
        ax, ay, az = accel

        # Normalize accel
        a_norm = np.linalg.norm([ax, ay, az])
        if a_norm < 1e-6:
            self._integrate_gyro(gyro)
            return
        ax, ay, az = ax / a_norm, ay / a_norm, az / a_norm

        # Gradient of objective function F_g (accel)
        qw, qx, qy, qz = q
        _2qw = 2.0 * qw
        _2qx = 2.0 * qx
        _2qy = 2.0 * qy
        _2qz = 2.0 * qz
        _4qw = 4.0 * qw
        _4qx = 4.0 * qx
        _4qy = 4.0 * qy
        _8qx = 8.0 * qx
        _8qy = 8.0 * qy
        q2w = qw * qw
        q2x = qx * qx
        q2y = qy * qy
        q2z = qz * qz

        s0 = _4qw * q2y - _2qy * ax + _4qw * q2x - _2qx * ay
        s1 = (_4qx * q2z - _2qz * ax + _4qx * q2w - _2qw * ay
              - _4qx + _8qx * q2x + _8qx * q2y + _4qx * az)
        s2 = (-_4qy + _8qy * q2x + _8qy * q2y + _4qy * az
              + _4qy * q2z - _2qz * ax - 2.0 * qy * ay)
        s3 = 4.0 * qy * q2z - 2.0 * qy * ax + 4.0 * qx * q2z - 2.0 * qx * ay

        s_norm = np.linalg.norm([s0, s1, s2, s3])
        if s_norm > 1e-6:
            s0, s1, s2, s3 = s0/s_norm, s1/s_norm, s2/s_norm, s3/s_norm

        # Rate of change of quaternion from gyroscope
        qdot_w = 0.5 * (-qx*gx - qy*gy - qz*gz) - self.beta * s0
        qdot_x = 0.5 * ( qw*gx + qy*gz - qz*gy) - self.beta * s1
        qdot_y = 0.5 * ( qw*gy - qx*gz + qz*gx) - self.beta * s2
        qdot_z = 0.5 * ( qw*gz + qx*gy - qy*gx) - self.beta * s3

        q = q + np.array([qdot_w, qdot_x, qdot_y, qdot_z]) * self.dt
        self.q = q / np.linalg.norm(q)

    def update_marg(self, gyro: np.ndarray, accel: np.ndarray, mag: np.ndarray):
        """Update with gyro + accel + magnetometer"""
        q = self.q
        gx, gy, gz = gyro

        a_norm = np.linalg.norm(accel)
        m_norm = np.linalg.norm(mag)
        if a_norm < 1e-6 or m_norm < 1e-6:
            self._integrate_gyro(gyro)
            return

        ax, ay, az = accel / a_norm
        mx, my, mz = mag / m_norm

        qw, qx, qy, qz = q

        # Auxiliary variables
        _2qw = 2.0 * qw; _2qx = 2.0 * qx; _2qy = 2.0 * qy; _2qz = 2.0 * qz
        _2qwmx = _2qw * mx; _2qwmy = _2qw * my; _2qwmz = _2qw * mz
        _2qxmx = _2qx * mx
        q2w=qw*qw; q2x=qx*qx; q2y=qy*qy; q2z=qz*qz

        # Reference direction of Earth's magnetic field
        hx = mx*q2w - _2qwmy*qz + _2qwmz*qy + mx*q2x + _2qx*my*qy + _2qx*mz*qz - mx*q2y - mx*q2z
        hy = _2qwmx*qz + my*q2w - _2qwmz*qx + _2qxmx*qy - my*q2x + my*q2y + _2qy*mz*qz - my*q2z
        _2bx = np.sqrt(hx*hx + hy*hy)
        _2bz = -_2qwmx*qy + _2qwmy*qx + mz*q2w + _2qxmx*qz - mz*q2x + _2qy*my*qz - mz*q2y + mz*q2z

        # Gradient of objective function
        s0 = -_2qy*(2.0*(qx*qz - qw*qy) - ax) + _2qx*(2.0*(qw*qx + qy*qz) - ay)
        s1 = _2qz*(2.0*(qx*qz - qw*qy) - ax) + _2qw*(2.0*(qw*qx + qy*qz) - ay) - 4.0*qx*(1 - 2.0*(q2x + q2y) - az) + (-_2bz*qy)*(_2bz*(0.5 - q2y - q2z) + _2bx*(qx*qz - qw*qy) - mx) + (_2bx*qy + _2bz*qw)*(_2bx*(qx*qy - qw*qz) + _2bz*(qw*qy + qx*qz) - my) + (_2bx*qz - 4.0*_2bz*qx)*(_2bx*(qw*qy + qx*qz) - _2bz*(qx*qy - qw*qz) - mz)
        s2 = -_2qw*(2.0*(qx*qz - qw*qy) - ax) + _2qz*(2.0*(qw*qx + qy*qz) - ay) - 4.0*qy*(1 - 2.0*(q2x + q2y) - az) + (-4.0*_2bx*qy - _2bz*qw)*(_2bx*(0.5 - q2y - q2z) + _2bx*(qx*qz - qw*qy) - mx) + (_2bx*qx + _2bz*qz)*(_2bx*(qx*qy - qw*qz) + _2bz*(qw*qy + qx*qz) - my) + (_2bx*qw - 4.0*_2bz*qy)*(_2bx*(qw*qy + qx*qz) - _2bz*(qx*qy - qw*qz) - mz)
        s3 = _2qx*(2.0*(qx*qz - qw*qy) - ax) + _2qy*(2.0*(qw*qx + qy*qz) - ay) + (-_2bx*qz + _2bz*qx)*(_2bx*(0.5 - q2y - q2z) + _2bx*(qx*qz - qw*qy) - mx) + (-_2bx*qw + _2bz*qy)*(_2bx*(qx*qy - qw*qz) + _2bz*(qw*qy + qx*qz) - my) + _2bx*qx*(_2bx*(qw*qy + qx*qz) - _2bz*(qx*qy - qw*qz) - mz)

        s_norm = np.linalg.norm([s0, s1, s2, s3])
        if s_norm > 1e-6:
            s0, s1, s2, s3 = s0/s_norm, s1/s_norm, s2/s_norm, s3/s_norm

        qdot_w = 0.5*(-qx*gx - qy*gy - qz*gz) - self.beta*s0
        qdot_x = 0.5*( qw*gx + qy*gz - qz*gy) - self.beta*s1
        qdot_y = 0.5*( qw*gy - qx*gz + qz*gx) - self.beta*s2
        qdot_z = 0.5*( qw*gz + qx*gy - qy*gx) - self.beta*s3

        q = q + np.array([qdot_w, qdot_x, qdot_y, qdot_z]) * self.dt
        self.q = q / np.linalg.norm(q)

    def _integrate_gyro(self, gyro: np.ndarray):
        """Pure gyro integration when accel is unavailable"""
        qw, qx, qy, qz = self.q
        gx, gy, gz = gyro
        qdot = 0.5 * np.array([
            -qx*gx - qy*gy - qz*gz,
             qw*gx + qy*gz - qz*gy,
             qw*gy - qx*gz + qz*gx,
             qw*gz + qx*gy - qy*gx,
        ])
        q = self.q + qdot * self.dt
        self.q = q / np.linalg.norm(q)

    def get_rotation_matrix(self) -> np.ndarray:
        """Returns 3x3 rotation matrix R_body2world"""
        qw, qx, qy, qz = self.q
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
            [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qw*qx)],
            [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)],
        ])
        return R

    def get_euler_angles(self) -> np.ndarray:
        """Returns (roll, pitch, yaw) in degrees"""
        qw, qx, qy, qz = self.q
        roll  = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
        pitch = np.degrees(np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1)))
        yaw   = np.degrees(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
        return np.array([roll, pitch, yaw])


class VORAnalyzer:
    """
    Vestibulo-Ocular Reflex (VOR) analysis.

    Expected VOR: when head rotates at angular_velocity ω,
    the eye should counter-rotate at approximately -ω (VOR gain ≈ 1.0).

    VOR mismatch = ||ω_eye_actual + R_cam2head @ ω_head|| / ||ω_head||
    where ω_eye is estimated from gaze direction change rate.

    Window: last 100ms of samples at 200Hz = 20 samples.
    """

    def __init__(self, window_ms: float = 100.0, fps: float = 200.0):
        self.window_size = int(window_ms / 1000.0 * fps)
        self.gaze_history: Deque = deque(maxlen=self.window_size)
        self.head_omega_history: Deque = deque(maxlen=self.window_size)
        self.dt = 1.0 / fps

    def update(self, gaze_dir_head: np.ndarray, head_omega: np.ndarray):
        self.gaze_history.append(gaze_dir_head.copy())
        self.head_omega_history.append(head_omega.copy())

    def compute_vor_mismatch(self) -> float:
        """Returns VOR mismatch [0, ∞) — higher = worse VOR"""
        if len(self.gaze_history) < 3:
            return 0.0

        # Estimate eye angular velocity from gaze direction changes
        gazes = np.array(list(self.gaze_history))      # (N, 3)
        omegas = np.array(list(self.head_omega_history))  # (N, 3)

        # Numerical derivative of gaze direction
        gaze_dot = np.diff(gazes, axis=0) / self.dt    # (N-1, 3)
        head_omega_mean = omegas[:-1].mean(axis=0)
        gaze_omega_mean = gaze_dot.mean(axis=0)

        omega_head_mag = np.linalg.norm(head_omega_mean)
        if omega_head_mag < 0.01:  # < ~0.5 deg/s: no meaningful head movement
            return 0.0

        # Expected: gaze should counter-rotate = -omega_head (VOR)
        vor_expected = -head_omega_mean
        vor_error = gaze_omega_mean - vor_expected
        mismatch = np.linalg.norm(vor_error) / omega_head_mag
        return float(np.clip(mismatch, 0.0, 10.0))


class IMUFusionModule:
    """
    Full pipeline:
    1. Accept IMU samples at 200 Hz
    2. Run Madgwick AHRS for head orientation
    3. Accept gaze ray in camera frame
    4. Transform to head frame and world frame
    5. Compute VOR mismatch
    """

    def __init__(self, cfg: dict):
        fps = cfg.get("fps", 200.0)
        beta = cfg.get("madgwick_beta", 0.033)
        self.ahrs = MadgwickAHRS(sample_rate=fps, beta=beta)
        self.vor_analyzer = VORAnalyzer(
            window_ms=cfg.get("vor_window_ms", 100.0), fps=fps
        )

        # Static transform: camera → head (determined by calibration rig)
        # Default: camera looks straight into eye, head frame aligned with helmet
        cam2head_euler = cfg.get("cam2head_euler_deg", [0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.R_cam2head = self._euler_to_R(*[np.deg2rad(a) for a in cam2head_euler])

        self.last_head_pose: Optional[HeadPose] = None
        self.gravity_vec = np.array([0.0, -9.81, 0.0])  # world frame

    def update_imu(self, sample: IMUSample):
        """Feed IMU sample into AHRS"""
        if sample.mag is not None:
            self.ahrs.update_marg(sample.gyro, sample.accel, sample.mag)
        else:
            self.ahrs.update_imu(sample.gyro, sample.accel)

        R = self.ahrs.get_rotation_matrix()
        euler = self.ahrs.get_euler_angles()

        # Gravity-free linear accel (world frame)
        accel_world = R @ sample.accel - self.gravity_vec
        gyro_world = R @ sample.gyro

        self.last_head_pose = HeadPose(
            timestamp=sample.timestamp,
            R_head2world=R,
            angular_velocity=sample.gyro.copy(),
            linear_accel=accel_world,
            euler=euler,
        )

    def transform_gaze(self, gaze_ray: GazeRay3D, timestamp: float) -> Optional[WorldGaze]:
        """
        Transform gaze ray from camera frame → head frame → world frame.
        """
        if self.last_head_pose is None:
            return None

        # Camera → Head
        gaze_head = self.R_cam2head @ gaze_ray.visual_axis
        gaze_head = gaze_head / (np.linalg.norm(gaze_head) + 1e-9)

        # Head → World
        R_h2w = self.last_head_pose.R_head2world
        gaze_world = R_h2w @ gaze_head
        gaze_world = gaze_world / (np.linalg.norm(gaze_world) + 1e-9)

        # VOR analysis
        self.vor_analyzer.update(gaze_head, self.last_head_pose.angular_velocity)
        vor_mismatch = self.vor_analyzer.compute_vor_mismatch()

        return WorldGaze(
            timestamp=timestamp,
            gaze_dir_world=gaze_world,
            gaze_dir_head=gaze_head,
            head_pose=self.last_head_pose,
            vor_mismatch=vor_mismatch,
            confidence=gaze_ray.confidence,
        )

    def get_head_pose(self) -> Optional[HeadPose]:
        return self.last_head_pose

    @staticmethod
    def _euler_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll),  np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw),  np.cos(yaw), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx