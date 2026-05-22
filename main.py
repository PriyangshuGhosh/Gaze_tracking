import cv2
import numpy as np
import time
import threading
import queue
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import torch

from eye_tracking.feature_extractor import EyeFeatureExtractor, EyeFeatures
from eye_tracking.eye_model_3d import EyeModelEstimator, EyeModel3D
from fusion.imu_fusion import IMUFusionModule, IMUSample, WorldGaze
from scene.scene_mapper import SceneMapper, CameraConfig
from models.neural_models import (
    GazeResidualCorrector, GazeTemporalModel,
    FixationSaccadeClassifier, NeuralCorrectionPipeline
)
from research.cognitive_features import (
    FixationDetector, GazeEntropyAnalyzer,
    EEDAnalyzer, RDIComputer, VORDisruptionMetric
)
from calibration.calibration import OnlineCalibrationAdapter

logger = logging.getLogger(__name__)


@dataclass
class GazeOutput:
    """Complete per-frame output from the pipeline"""
    timestamp: float
    frame_id: int

    # Core gaze
    gaze_az_deg: float
    gaze_el_deg: float
    gaze_confidence: float

    # Research metrics
    eye_class: str              # fixation / saccade / pursuit / blink
    vor_mismatch: float
    rdi: float
    rdi_level: str

    # Scene
    gazed_object_label: Optional[str]
    gazed_object_conf: float

    # Latency
    pipeline_latency_ms: float


class GazeTrackingPipeline:
    """
    Full real-time pipeline.
    Threading model:
      Thread 1: Eye camera → EyeFeatureExtractor → queue_eye
      Thread 2: IMU stream → IMUFusionModule (in-line, lightweight)
      Thread 3: Eye features → 3D model → neural correction → temporal → queue_gaze
      Thread 4: Scene camera → YOLO → scene mapping (lower priority, async)
      Main:     Consume queue_gaze → research features → output
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fps = cfg.get("fps", 200)
        self.device = cfg.get("device", "cpu")

        # ── Eye feature extractor
        self.eye_extractor = EyeFeatureExtractor(cfg.get("eye", {}))

        # ── 3D eye model
        led_positions = np.array(cfg.get("led_positions_3d", [
            [0, -15, 40], [15, 0, 40], [0, 15, 40], [-15, 0, 40]
        ]), dtype=np.float64)
        eye_model = EyeModel3D(**cfg.get("eye_model", {}))
        self.eye_model_estimator = EyeModelEstimator(eye_model, led_positions)

        # ── IMU fusion
        self.imu_module = IMUFusionModule(cfg.get("imu", {"fps": self.fps}))

        # ── Neural correction pipeline
        corrector  = GazeResidualCorrector()
        temporal   = GazeTemporalModel()
        classifier = FixationSaccadeClassifier()

        # Load pretrained weights if available
        ckpt = cfg.get("corrector_checkpoint")
        if ckpt and Path(ckpt).exists():
            state = torch.load(ckpt, map_location=self.device)
            corrector.load_state_dict(state["model_state"])
            logger.info(f"Loaded corrector checkpoint: {ckpt}")

        self.neural_pipeline = NeuralCorrectionPipeline(
            corrector, temporal, classifier, device=self.device
        )

        # ── Online calibration
        kappa_init = cfg.get("kappa_init", [5.0, 1.5])
        self.online_calib = OnlineCalibrationAdapter(tuple(kappa_init))
        self.eye_model_estimator.update_kappa(kappa_init[0], kappa_init[1])

        # ── Scene mapper (optional — lower priority thread)
        self.scene_mapper: Optional[SceneMapper] = None
        if "scene_cameras" in cfg:
            cameras = [CameraConfig(**c) for c in cfg["scene_cameras"]]
            self.scene_mapper = SceneMapper(
                cameras,
                yolo_model_path=cfg.get("yolo_model", "yolov8n.pt"),
                yolo_device=self.device,
            )

        # ── Research features
        self.fixation_detector = FixationDetector({"fps": self.fps})
        self.entropy_analyzer  = GazeEntropyAnalyzer({})
        self.eed_analyzer      = EEDAnalyzer({})
        self.rdi_computer      = RDIComputer({})
        self.vor_metric        = VORDisruptionMetric({})

        # ── Queues
        self.queue_eye   = queue.Queue(maxsize=20)
        self.queue_imu   = queue.Queue(maxsize=50)
        self.queue_gaze  = queue.Queue(maxsize=20)
        self.queue_scene = queue.Queue(maxsize=5)
        self.output_queue = queue.Queue(maxsize=50)

        # ── State
        self._running = False
        self._frame_id = 0
        self._last_world_gaze: Optional[WorldGaze] = None
        self._last_rdi = 0.0
        self._last_rdi_level = "normal"

        # ── Profiling
        self._latency_buffer = []

    # ──────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────

    def start(self):
        self._running = True
        self._threads = [
            threading.Thread(target=self._imu_processing_thread, daemon=True, name="IMUThread"),
            threading.Thread(target=self._gaze_estimation_thread, daemon=True, name="GazeThread"),
        ]
        for t in self._threads:
            t.start()
        logger.info("GazeTrackingPipeline started")

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        logger.info("GazeTrackingPipeline stopped")

    def push_eye_frame(self, frame: np.ndarray, timestamp: float):
        """Call from eye camera callback at 200 Hz"""
        try:
            self.queue_eye.put_nowait((frame, timestamp, self._frame_id))
            self._frame_id += 1
        except queue.Full:
            pass  # Drop frame — non-blocking

    def push_imu_sample(self, sample: IMUSample):
        """Call from IMU callback at 200 Hz"""
        try:
            self.queue_imu.put_nowait(sample)
        except queue.Full:
            pass

    def push_scene_frames(self, frames: Dict[str, np.ndarray], timestamp: float):
        """Call from scene camera callback (lower frequency OK — 30-60 FPS)"""
        try:
            self.queue_scene.put_nowait((frames, timestamp))
        except queue.Full:
            pass

    def get_output(self, timeout: float = 0.01) -> Optional[GazeOutput]:
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ──────────────────────────────────────────────
    # Processing Threads
    # ──────────────────────────────────────────────

    def _imu_processing_thread(self):
        """Processes IMU queue — very fast, O(1) per sample"""
        while self._running:
            try:
                sample = self.queue_imu.get(timeout=0.005)
                self.imu_module.update_imu(sample)
            except queue.Empty:
                continue

    def _gaze_estimation_thread(self):
        """Main gaze estimation thread"""
        while self._running:
            try:
                frame, timestamp, frame_id = self.queue_eye.get(timeout=0.005)
            except queue.Empty:
                continue

            t0 = time.perf_counter()
            output = self._process_eye_frame(frame, timestamp, frame_id)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if output is not None:
                output.pipeline_latency_ms = latency_ms
                try:
                    self.output_queue.put_nowait(output)
                except queue.Full:
                    pass

            self._latency_buffer.append(latency_ms)
            if len(self._latency_buffer) > 200:
                self._latency_buffer.pop(0)

    def _process_eye_frame(
        self, frame: np.ndarray, timestamp: float, frame_id: int
    ) -> Optional[GazeOutput]:
        # 1. Eye feature extraction (~1-2ms)
        eye_features = self.eye_extractor.process(frame, timestamp, frame_id)

        if not eye_features.valid:
            return None

        # 2. 3D gaze estimation (~0.5ms)
        gaze_ray = self.eye_model_estimator.estimate(eye_features)
        if gaze_ray is None:
            return None

        # 3. IMU fusion → world gaze (~0.1ms)
        world_gaze = self.imu_module.transform_gaze(gaze_ray, timestamp)

        # If IMU not available yet, use camera-frame gaze
        if world_gaze is None:
            az = float(np.degrees(np.arctan2(gaze_ray.visual_axis[0],
                                              gaze_ray.visual_axis[2])))
            el = float(np.degrees(np.arctan2(gaze_ray.visual_axis[1],
                                              gaze_ray.visual_axis[2])))
            vor_mismatch = 0.0
        else:
            self._last_world_gaze = world_gaze
            az = float(np.degrees(np.arctan2(world_gaze.gaze_dir_world[0],
                                              world_gaze.gaze_dir_world[2])))
            el = float(np.degrees(np.arctan2(world_gaze.gaze_dir_world[1],
                                              world_gaze.gaze_dir_world[2])))
            vor_mismatch = world_gaze.vor_mismatch

        # 4. Build neural correction features
        eye_patch = self._extract_patch(frame, eye_features)
        geo_feat  = self._build_geo_features(eye_features)
        imu_feat  = self._build_imu_features(world_gaze)
        raw_gaze_arr = np.array([az, el,
                                  eye_features.pupil.confidence if eye_features.pupil else 0.0,
                                  vor_mismatch,
                                  eye_features.eyelid_closure], dtype=np.float32)

        # 5. Neural correction + temporal (~2-4ms)
        nn_result = self.neural_pipeline.process(
            eye_patch, geo_feat, imu_feat, raw_gaze_arr
        )

        smoothed_az = float(nn_result["smoothed_gaze"][0])
        smoothed_el = float(nn_result["smoothed_gaze"][1])
        velocity    = nn_result["velocity"]
        eye_class   = nn_result["class_name"]

        # 6. Research features (~0.5ms)
        gaze_2d = np.array([smoothed_az, smoothed_el])
        gaze_event = self.fixation_detector.update(
            gaze_2d, velocity, timestamp
        )

        if gaze_event is not None and gaze_event.fixation is not None:
            self.eed_analyzer.add_fixation(gaze_event.fixation)
            self.entropy_analyzer.add_fixation(gaze_event.fixation)
        if gaze_event is not None and gaze_event.saccade is not None:
            self.rdi_computer.add_saccade(timestamp)

        self.rdi_computer.add_eyelid_sample(timestamp, eye_features.eyelid_closure)
        self.rdi_computer.add_vor_mismatch(vor_mismatch)

        # Compute RDI at lower rate (every 10 frames)
        if frame_id % 10 == 0:
            hs = self.entropy_analyzer.compute_spatial_entropy()
            ht = self.entropy_analyzer.compute_temporal_entropy()
            eed_result = self.eed_analyzer.compute()
            rdi_result = self.rdi_computer.compute(hs, ht, eed_result.get("chi", 0.0))
            self._last_rdi = rdi_result["rdi"]
            self._last_rdi_level = rdi_result["level"]

        # 7. Online calibration update (if scene mapper provides feedback)
        calib_corrected = self.online_calib.apply(np.array([smoothed_az, smoothed_el]))
        final_az = float(calib_corrected[0])
        final_el = float(calib_corrected[1])

        return GazeOutput(
            timestamp=timestamp,
            frame_id=frame_id,
            gaze_az_deg=final_az,
            gaze_el_deg=final_el,
            gaze_confidence=float(gaze_ray.confidence),
            eye_class=eye_class,
            vor_mismatch=vor_mismatch,
            rdi=self._last_rdi,
            rdi_level=self._last_rdi_level,
            gazed_object_label=None,
            gazed_object_conf=0.0,
            pipeline_latency_ms=0.0,
        )

    # ──────────────────────────────────────────────
    # Helper: Feature Builders
    # ──────────────────────────────────────────────

    def _extract_patch(self, frame: np.ndarray, features: EyeFeatures) -> np.ndarray:
        """Extract 32x32 normalized eye patch centered on pupil"""
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        if features.pupil is not None:
            cx, cy = int(features.pupil.center[0]), int(features.pupil.center[1])
        else:
            cx, cy = w // 2, h // 2

        r = 24
        x0, x1 = max(0, cx - r), min(w, cx + r)
        y0, y1 = max(0, cy - r), min(h, cy + r)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return np.zeros((32, 32), dtype=np.float32)
        patch = cv2.resize(roi, (32, 32)).astype(np.float32) / 255.0
        return patch

    def _build_geo_features(self, features: EyeFeatures) -> np.ndarray:
        """Build 12-dim geometric feature vector"""
        feat = np.zeros(12, dtype=np.float32)
        if features.pupil is not None:
            feat[0] = features.pupil.center[0] / 640.0  # normalize to [0,1]
            feat[1] = features.pupil.center[1] / 480.0
            feat[2] = features.pupil.contour_area / 8000.0
            feat[3] = features.pupil.confidence
        if features.glints is not None:
            for i in range(min(4, len(features.glints.centers))):
                if features.glints.valid_mask[i]:
                    feat[4 + i*2]   = features.glints.centers[i, 0] / 640.0
                    feat[4 + i*2+1] = features.glints.centers[i, 1] / 480.0
        return feat

    def _build_imu_features(self, world_gaze: Optional[WorldGaze]) -> np.ndarray:
        """Build 6-dim IMU feature vector"""
        feat = np.zeros(6, dtype=np.float32)
        if world_gaze is not None and world_gaze.head_pose is not None:
            euler = world_gaze.head_pose.euler
            omega = world_gaze.head_pose.angular_velocity
            feat[0] = float(np.sin(np.deg2rad(euler[0])))  # roll sin
            feat[1] = float(np.sin(np.deg2rad(euler[1])))  # pitch sin
            feat[2] = float(np.sin(np.deg2rad(euler[2])))  # yaw sin
            feat[3:6] = np.clip(omega / 10.0, -1.0, 1.0)  # angular vel normalized
        return feat

    def get_latency_stats(self) -> dict:
        if not self._latency_buffer:
            return {}
        arr = np.array(self._latency_buffer)
        return {
            "mean_ms": float(arr.mean()),
            "p95_ms":  float(np.percentile(arr, 95)),
            "p99_ms":  float(np.percentile(arr, 99)),
            "max_ms":  float(arr.max()),
        }


# ──────────────────────────────────────────────
# Real-time Optimization Utilities
# ──────────────────────────────────────────────

class ModelOptimizer:
    """
    Strategies to hit <20ms total pipeline latency on Jetson Orin Nano.

    Profiling targets (200 FPS = 5ms budget per frame):
      Eye extraction:     1.5ms
      3D model:           0.5ms
      Neural corrector:   1.5ms  (TensorRT INT8)
      Temporal GRU:       0.5ms
      Research features:  0.3ms
      Scene mapping:      async (not on critical path)
      Total:             ~4.3ms  → headroom for overhead
    """

    @staticmethod
    def export_torchscript(model: torch.nn.Module, path: str, example_inputs: tuple):
        """Export model to TorchScript for 2x speedup"""
        scripted = torch.jit.trace(model, example_inputs)
        scripted = torch.jit.optimize_for_inference(scripted)
        torch.jit.save(scripted, path)
        return scripted

    @staticmethod
    def export_tensorrt(model: torch.nn.Module, path: str, example_inputs: tuple, int8: bool = False):
        """
        Export to TensorRT (requires torch2trt or tensorrt python bindings).
        INT8 quantization: ~4x speedup vs FP32, ~1.5% accuracy drop.
        """
        try:
            from torch2trt import torch2trt
            trt_model = torch2trt(
                model, [*example_inputs],
                fp16_mode=not int8,
                int8_mode=int8,
                max_workspace_size=1 << 25,
            )
            torch.save(trt_model.state_dict(), path)
            return trt_model
        except ImportError:
            logger.warning("torch2trt not available — TensorRT export skipped")
            return None

    @staticmethod
    def quantize_dynamic(model: torch.nn.Module) -> torch.nn.Module:
        """Dynamic quantization: INT8 weights, FP32 activations. No calibration needed."""
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.GRU}, dtype=torch.qint8
        )

    @staticmethod
    def prune_model(model: torch.nn.Module, amount: float = 0.3):
        """Unstructured L1 pruning: remove 30% lowest magnitude weights"""
        import torch.nn.utils.prune as prune
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=amount)
                prune.remove(module, "weight")
        return model
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Minimal config
    cfg = {
        "fps": 30,
        "device": "cpu",
    }

    pipeline = GazeTrackingPipeline(cfg)
    pipeline.start()

    print("Pipeline initialized...")

    # Dummy loop (simulate camera input)
    for i in range(20):
        dummy_eye = np.zeros((240, 320, 3), dtype=np.uint8)
        timestamp = time.time()

        pipeline.push_eye_frame(dummy_eye, timestamp)

        # Fake IMU data
        imu_sample = IMUSample(
            timestamp=timestamp,
            accel=np.zeros(3),
            gyro=np.zeros(3)
        )
        pipeline.push_imu_sample(imu_sample)

        # Read output
        output = pipeline.get_output()

        if output:
            print(f"[Frame {output.frame_id}] Gaze: ({output.gaze_az_deg:.2f}, {output.gaze_el_deg:.2f}) | RDI: {output.rdi:.3f}")

        time.sleep(0.03)

    pipeline.stop()
    print("Pipeline stopped.")