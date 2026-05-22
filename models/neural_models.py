"""
Neural Residual Correction Model
  - Input: eye patch (32x32) + geometric features (12-dim) + IMU (6-dim)
  - Output: gaze correction vector (2-dim: Δazimuth, Δelevation in degrees)
  - Architecture: MobileNetV2-mini encoder + MLP head
  - Target: <2ms inference on embedded GPU (Jetson Orin Nano)

Temporal GRU Model
  - Input: sequence of gaze vectors + features
  - Output: smoothed gaze + velocity features
  - Used for: saccade/fixation detection, EED computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────
# Neural Residual Correction Model
# ─────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution — 8-9x fewer params than standard conv"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu6(self.bn(self.pw(self.dw(x))))


class EyePatchEncoder(nn.Module):
    """
    Encodes 32x32 grayscale eye patch to 64-dim feature vector.
    Input: (B, 1, 32, 32)
    Output: (B, 64)
    ~50K parameters, <0.5ms on Jetson.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            DepthwiseSeparableConv(16, 32, stride=2),               # 8x8
            DepthwiseSeparableConv(32, 64, stride=2),               # 4x4
            DepthwiseSeparableConv(64, 64, stride=2),               # 2x2
            nn.AdaptiveAvgPool2d(1),                                 # 1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)  # (B, 64)


class GazeResidualCorrector(nn.Module):
    """
    Full residual correction model.

    Input streams:
      - eye_patch:   (B, 1, 32, 32) normalized IR image
      - geo_feat:    (B, 12) geometric features:
                       [pupil_x, pupil_y, pupil_area, pupil_conf,
                        glint0_x, glint0_y, glint1_x, glint1_y,
                        glint2_x, glint2_y, glint3_x, glint3_y]  — all normalized [0,1]

    Output:
      - delta_gaze:  (B, 2) [Δazimuth_deg, Δelevation_deg]
    """

    def __init__(self, geo_dim: int = 12):
        super().__init__()
        self.patch_encoder = EyePatchEncoder()  # → 64

        # Geometric feature branch
        self.geo_net = nn.Sequential(
            nn.Linear(geo_dim, 32), nn.ReLU6(inplace=True),
            nn.Linear(32, 32),
        )

        # Fusion head: 64 + 32 = 96 → 2
        self.fusion = nn.Sequential(
            nn.Linear(96, 64), nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU6(inplace=True),
            nn.Linear(32, 2),  # Δazimuth, Δelevation
        )

        # Initialize last layer near zero — residual should start small
        nn.init.normal_(self.fusion[-1].weight, std=0.01)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(
        self,
        eye_patch: torch.Tensor,
        geo_feat: torch.Tensor,
    ) -> torch.Tensor:
        patch_emb = self.patch_encoder(eye_patch)  # (B, 64)
        geo_emb   = self.geo_net(geo_feat)         # (B, 32)
        fused = torch.cat([patch_emb, geo_emb], dim=1)  # (B, 96)
        return self.fusion(fused)  # (B, 2)

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.jit.export
    def infer(
        self,
        eye_patch: torch.Tensor,
        geo_feat: torch.Tensor,
    ) -> torch.Tensor:
        """TorchScript-compatible inference"""
        with torch.no_grad():
            return self.forward(eye_patch, geo_feat)


# ─────────────────────────────────────────────
# Temporal GRU Model
# ─────────────────────────────────────────────

class GazeTemporalModel(nn.Module):
    """
    GRU-based temporal smoother and feature extractor.

    Input per timestep:
      - gaze_vec: (B, T, 5)
          [azimuth, elevation, confidence, vor_mismatch, eyelid_closure]

    Outputs:
      - smoothed_gaze:  (B, T, 2) — smoothed azimuth, elevation
      - gaze_velocity:  (B, T, 2) — angular velocity estimates
      - hidden_state:   (B, hidden_dim) — for downstream classifiers

    Designed for:
      - Online streaming: process one step at a time
      - Offline batch: process full sequence
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )

        # Smoothed gaze output
        self.gaze_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.Tanh(),
            nn.Linear(32, 2),
        )

        # Velocity estimation head
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(
        self,
        gaze_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        gaze_seq: (B, T, input_dim)
        hidden:   (num_layers, B, hidden_dim) or None

        Returns:
            smoothed_gaze: (B, T, 2)
            gaze_velocity: (B, T, 2)
            hidden_out:    (num_layers, B, hidden_dim)
        """
        out, hidden_out = self.gru(gaze_seq, hidden)  # (B, T, hidden_dim)

        smoothed = self.gaze_head(out)       # (B, T, 2)
        velocity = self.velocity_head(out)   # (B, T, 2)

        return smoothed, velocity, hidden_out

    def step(
        self,
        gaze_t: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step online inference.
        gaze_t: (B, input_dim) → unsqueeze to (B, 1, input_dim)
        """
        gaze_t = gaze_t.unsqueeze(1)
        s, v, h = self.forward(gaze_t, hidden)
        return s.squeeze(1), v.squeeze(1), h  # (B, 2), (B, 2), hidden


class FixationSaccadeClassifier(nn.Module):
    """
    Classifies each timestep into: fixation / saccade / smooth_pursuit / blink
    Input: hidden state from GazeTemporalModel + velocity
    """
    CLASS_NAMES = ["fixation", "saccade", "smooth_pursuit", "blink"]

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 2, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 4),  # 4 classes
        )

    def forward(self, hidden: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """
        hidden:   (B, hidden_dim)
        velocity: (B, 2)
        Returns:  (B, 4) logits
        """
        x = torch.cat([hidden, velocity], dim=1)
        return self.net(x)


# ─────────────────────────────────────────────
# Runtime correction pipeline (numpy interface)
# ─────────────────────────────────────────────

class NeuralCorrectionPipeline:
    """
    Wraps GazeResidualCorrector + GazeTemporalModel for runtime use.
    Maintains GRU hidden state across frames.
    """

    def __init__(
        self,
        corrector: GazeResidualCorrector,
        temporal: GazeTemporalModel,
        classifier: FixationSaccadeClassifier,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.corrector = corrector.to(self.device).eval()
        self.temporal  = temporal.to(self.device).eval()
        self.classifier = classifier.to(self.device).eval()
        self.hidden: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def process(
        self,
        eye_patch_np: np.ndarray,       # (32, 32) float32 [0, 1]
        geo_feat_np: np.ndarray,         # (12,) float32
        raw_gaze_np: np.ndarray,         # (5,) [az, el, conf, vor, eyelid]
    ) -> dict:
        """
        Returns:
          corrected_gaze: (2,) [azimuth, elevation] in degrees
          smoothed_gaze:  (2,)
          velocity:       (2,) deg/s
          eye_class:      int [0=fixation, 1=saccade, 2=pursuit, 3=blink]
          class_name:     str
        """
        # Build tensors
        patch = torch.from_numpy(eye_patch_np).float().unsqueeze(0).unsqueeze(0).to(self.device)
        geo   = torch.from_numpy(geo_feat_np).float().unsqueeze(0).to(self.device)
        gaze  = torch.from_numpy(raw_gaze_np).float().unsqueeze(0).to(self.device)

        # Residual correction
        delta = self.corrector(patch, geo)  # (1, 2)

        # Apply correction to raw gaze
        raw_az, raw_el = raw_gaze_np[0], raw_gaze_np[1]
        # delta is a torch tensor on device; bring to CPU numpy for arithmetic
        delta_np = delta[0].cpu().numpy()
        corrected_az = float(raw_az + float(delta_np[0]))
        corrected_el = float(raw_el + float(delta_np[1]))

        # Update gaze with corrected values for temporal model (assign Python floats)
        gaze[0, 0] = corrected_az
        gaze[0, 1] = corrected_el

        # Temporal step
        smoothed, velocity, self.hidden = self.temporal.step(gaze, self.hidden)

        # Classification
        hidden_last = self.hidden[-1]  # last GRU layer: (1, hidden_dim)
        logits = self.classifier(hidden_last, velocity)
        eye_class = int(logits.argmax(dim=1).item())

        return {
            "corrected_gaze": np.array([corrected_az, corrected_el]),
            "smoothed_gaze": smoothed[0].cpu().numpy(),
            "velocity": velocity[0].cpu().numpy(),
            "eye_class": eye_class,
            "class_name": FixationSaccadeClassifier.CLASS_NAMES[eye_class],
            "delta": delta[0].cpu().numpy(),
        }

    def reset_state(self):
        self.hidden = None