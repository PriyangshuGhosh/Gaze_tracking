"""
Training Pipeline for GazeResidualCorrector + GazeTemporalModel

Dataset format, loss functions (explicit formulas), training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, Tuple, List
from models.neural_models import GazeResidualCorrector, GazeTemporalModel, FixationSaccadeClassifier


# ─────────────────────────────────────────────
# Dataset Format
# ─────────────────────────────────────────────
#
# Root: /data/gaze_dataset/
# ├── train/
# │   ├── session_001/
# │   │   ├── frames/           # eye patch images: frame_000000.png (32x32 grayscale)
# │   │   ├── features.npy      # (N, 12) geometric features per frame
# │   │   ├── imu.npy           # (N, 6)  IMU features per frame
# │   │   ├── raw_gaze.npy      # (N, 2)  geometric gaze estimate (az, el) in degrees
# │   │   ├── gt_gaze.npy       # (N, 2)  ground-truth gaze from calibration target
# │   │   ├── timestamps.npy    # (N,)    timestamps in seconds
# │   │   └── meta.json         # rider ID, session info, calibration params
# └── val/
#     └── ...
#
# meta.json format:
# {
#   "rider_id": "R001",
#   "session_id": "S001",
#   "fps": 200,
#   "kappa_h": 5.2,
#   "kappa_v": 1.3,
#   "calibration_rmse_deg": 0.8
# }


class GazeDataset(Dataset):
    """
    Loads eye patches + features + IMU + ground truth gaze.
    Returns single-frame samples for residual corrector training.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        sequence_len: int = 1,     # 1 for corrector, >1 for temporal model
        augment: bool = True,
    ):
        self.root = Path(root_dir) / split
        self.seq_len = sequence_len
        self.augment = augment

        self.samples: List[dict] = []
        self._load_sessions()

    def _load_sessions(self):
        for session_dir in sorted(self.root.iterdir()):
            if not session_dir.is_dir():
                continue

            features = np.load(session_dir / "features.npy").astype(np.float32)  # (N, 12)
            imu      = np.load(session_dir / "imu.npy").astype(np.float32)        # (N, 6)
            raw_gaze = np.load(session_dir / "raw_gaze.npy").astype(np.float32)   # (N, 2)
            gt_gaze  = np.load(session_dir / "gt_gaze.npy").astype(np.float32)    # (N, 2)
            frames_dir = session_dir / "frames"

            N = len(gt_gaze)
            frame_paths = sorted(frames_dir.glob("*.png"))

            if len(frame_paths) != N:
                print(f"Warning: frame count mismatch in {session_dir}")
                N = min(N, len(frame_paths))

            for i in range(N - self.seq_len + 1):
                self.samples.append({
                    "frame_paths": [str(frame_paths[i + k]) for k in range(self.seq_len)],
                    "features":    features[i:i+self.seq_len],
                    "imu":         imu[i:i+self.seq_len],
                    "raw_gaze":    raw_gaze[i:i+self.seq_len],
                    "gt_gaze":     gt_gaze[i:i+self.seq_len],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        import cv2
        s = self.samples[idx]

        patches = []
        for fp in s["frame_paths"]:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((32, 32), dtype=np.uint8)
            img = cv2.resize(img, (32, 32))
            patch = img.astype(np.float32) / 255.0
            if self.augment and self.seq_len == 1:
                patch = self._augment_patch(patch)
            patches.append(patch)

        patches = np.stack(patches, axis=0)  # (T, 32, 32)

        features = s["features"].copy()  # (T, 12)
        imu      = s["imu"].copy()       # (T, 6)
        raw_gaze = s["raw_gaze"].copy()  # (T, 2)
        gt_gaze  = s["gt_gaze"].copy()   # (T, 2)

        # Compute ground-truth delta (correction needed)
        delta_gt = gt_gaze - raw_gaze    # (T, 2)

        if self.seq_len == 1:
            return {
                "patch":    torch.from_numpy(patches[0]).unsqueeze(0),  # (1, 32, 32)
                "geo":      torch.from_numpy(features[0]),               # (12,)
                "imu":      torch.from_numpy(imu[0]),                    # (6,)
                "raw_gaze": torch.from_numpy(raw_gaze[0]),               # (2,)
                "delta_gt": torch.from_numpy(delta_gt[0]),               # (2,)
                "gt_gaze":  torch.from_numpy(gt_gaze[0]),                # (2,)
            }
        else:
            return {
                "patch":    torch.from_numpy(patches).unsqueeze(1),  # (T, 1, 32, 32)
                "geo":      torch.from_numpy(features),               # (T, 12)
                "imu":      torch.from_numpy(imu),                    # (T, 6)
                "raw_gaze": torch.from_numpy(raw_gaze),               # (T, 2)
                "delta_gt": torch.from_numpy(delta_gt),               # (T, 2)
                "gt_gaze":  torch.from_numpy(gt_gaze),                # (T, 2)
            }

    def _augment_patch(self, patch: np.ndarray) -> np.ndarray:
        """Noise + blur augmentation for IR image robustness"""
        # Gaussian noise
        noise = np.random.normal(0, 0.02, patch.shape).astype(np.float32)
        patch = np.clip(patch + noise, 0.0, 1.0)
        # Random brightness
        patch = np.clip(patch * np.random.uniform(0.85, 1.15), 0.0, 1.0)
        # Horizontal flip (eye is symmetric to some extent)
        if np.random.random() < 0.3:
            patch = patch[:, ::-1].copy()
        return patch


# ─────────────────────────────────────────────
# Loss Functions (explicit formulas)
# ─────────────────────────────────────────────

class AngularLoss(nn.Module):
    """
    Angular error between predicted and GT gaze vectors.

    L_ang = mean( arccos( clip( v_pred · v_gt / (||v_pred|| * ||v_gt||), -1, 1 ) ) )

    Converts azimuth/elevation to unit 3D vectors before computing angle.
    More meaningful than L2 in degrees when errors are large.
    """

    def __init__(self):
        super().__init__()

    def az_el_to_vec(self, az_el: torch.Tensor) -> torch.Tensor:
        """
        az_el: (B, 2) in degrees [azimuth, elevation]
        Returns: (B, 3) unit vectors
        """
        az  = torch.deg2rad(az_el[:, 0])
        el  = torch.deg2rad(az_el[:, 1])
        x = torch.cos(el) * torch.sin(az)
        y = torch.sin(el)
        z = torch.cos(el) * torch.cos(az)
        return torch.stack([x, y, z], dim=1)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred, gt: (B, 2) in degrees
        """
        v_pred = self.az_el_to_vec(pred)
        v_gt   = self.az_el_to_vec(gt)
        dot = (v_pred * v_gt).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle_rad = torch.acos(dot)
        return angle_rad.mean()


class HuberGazeLoss(nn.Module):
    """
    Combined loss for residual corrector:

    L = α * L_ang + β * L_huber + γ * L_reg

    L_huber(δ) = {
        0.5 * δ^2             if |δ| ≤ 1
        |δ| - 0.5             otherwise
    }  (element-wise, then mean)

    L_reg = ||delta||^2 * λ   (regularize correction to be small)

    α=0.6, β=0.3, γ=0.1  (tuned on driving dataset)
    """

    def __init__(self, alpha: float = 0.6, beta: float = 0.3, gamma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.ang_loss   = AngularLoss()
        self.huber_loss = nn.HuberLoss(delta=1.0, reduction="mean")

    def forward(
        self,
        pred_delta: torch.Tensor,   # (B, 2)
        raw_gaze: torch.Tensor,     # (B, 2)
        gt_gaze: torch.Tensor,      # (B, 2)
    ) -> Tuple[torch.Tensor, dict]:
        pred_gaze = raw_gaze + pred_delta
        delta_gt  = gt_gaze - raw_gaze

        L_ang   = self.ang_loss(pred_gaze, gt_gaze)
        L_huber = self.huber_loss(pred_delta, delta_gt)
        L_reg   = (pred_delta ** 2).mean()

        L_total = self.alpha * L_ang + self.beta * L_huber + self.gamma * L_reg

        return L_total, {
            "L_total": L_total.item(),
            "L_ang":   L_ang.item(),
            "L_huber": L_huber.item(),
            "L_reg":   L_reg.item(),
        }


class TemporalGazeLoss(nn.Module):
    """
    Loss for GazeTemporalModel:

    L = L_smooth + λ_v * L_velocity + λ_c * L_consistency

    L_smooth:      HuberLoss between smoothed gaze and GT gaze
    L_velocity:    MSE between predicted velocity and finite-diff GT velocity
                   v_gt[t] = (gaze_gt[t+1] - gaze_gt[t-1]) / (2*dt) * fps
    L_consistency: MSE between consecutive smoothed gaze predictions
                   (encourages smooth output even when GT is noisy)
    """

    def __init__(self, fps: float = 200.0, lambda_v: float = 0.1, lambda_c: float = 0.05):
        super().__init__()
        self.fps       = fps
        self.lambda_v  = lambda_v
        self.lambda_c  = lambda_c
        self.huber     = nn.HuberLoss(delta=1.0)

    def forward(
        self,
        smoothed: torch.Tensor,   # (B, T, 2)
        velocity: torch.Tensor,   # (B, T, 2)
        gt_gaze: torch.Tensor,    # (B, T, 2)
    ) -> Tuple[torch.Tensor, dict]:
        # L_smooth
        L_smooth = self.huber(smoothed, gt_gaze)

        # L_velocity: compute GT velocity via central differences
        if gt_gaze.shape[1] >= 3:
            v_gt = (gt_gaze[:, 2:, :] - gt_gaze[:, :-2, :]) * self.fps / 2.0
            L_vel = F.mse_loss(velocity[:, 1:-1, :], v_gt)
        else:
            L_vel = torch.tensor(0.0, device=smoothed.device)

        # L_consistency: smoothness of output
        if smoothed.shape[1] >= 2:
            diff = smoothed[:, 1:, :] - smoothed[:, :-1, :]
            L_cons = (diff ** 2).mean()
        else:
            L_cons = torch.tensor(0.0, device=smoothed.device)

        L_total = L_smooth + self.lambda_v * L_vel + self.lambda_c * L_cons

        return L_total, {
            "L_total":    L_total.item(),
            "L_smooth":   L_smooth.item(),
            "L_velocity": L_vel.item(),
            "L_consist":  L_cons.item(),
        }


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

class CorrectorTrainer:
    """Full training loop for GazeResidualCorrector"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

        self.model = GazeResidualCorrector(
            geo_dim=cfg.get("geo_dim", 12),
            imu_dim=cfg.get("imu_dim", 6),
        ).to(self.device)

        self.criterion = HuberGazeLoss(
            alpha=cfg.get("alpha", 0.6),
            beta=cfg.get("beta", 0.3),
            gamma=cfg.get("gamma", 0.1),
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.get("lr", 1e-3),
            total_steps=cfg.get("total_steps", 50000),
            pct_start=0.1,
        )

        self.best_val_loss = float("inf")
        self.checkpoint_dir = Path(cfg.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        metrics_sum = {"L_total": 0.0, "L_ang": 0.0, "L_huber": 0.0, "L_reg": 0.0}
        n_batches = 0

        for batch in loader:
            patch    = batch["patch"].to(self.device)     # (B, 1, 32, 32)
            geo      = batch["geo"].to(self.device)        # (B, 12)
            imu      = batch["imu"].to(self.device)        # (B, 6)
            raw_gaze = batch["raw_gaze"].to(self.device)   # (B, 2)
            gt_gaze  = batch["gt_gaze"].to(self.device)    # (B, 2)

            self.optimizer.zero_grad()

            pred_delta = self.model(patch, geo, imu)      # (B, 2)
            loss, metrics = self.criterion(pred_delta, raw_gaze, gt_gaze)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            for k, v in metrics.items():
                metrics_sum[k] += v
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in metrics_sum.items()}

    @torch.inference_mode()
    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        angular_errors = []

        ang_loss_fn = AngularLoss().to(self.device)

        for batch in loader:
            patch    = batch["patch"].to(self.device)
            geo      = batch["geo"].to(self.device)
            imu      = batch["imu"].to(self.device)
            raw_gaze = batch["raw_gaze"].to(self.device)
            gt_gaze  = batch["gt_gaze"].to(self.device)

            pred_delta = self.model(patch, geo, imu)
            pred_gaze  = raw_gaze + pred_delta

            ang_err = ang_loss_fn(pred_gaze, gt_gaze)
            angular_errors.append(torch.rad2deg(ang_err).item())

        mean_err = float(np.mean(angular_errors))
        return {
            "mean_angular_error_deg": mean_err,
            "p90_angular_error_deg": float(np.percentile(angular_errors, 90)),
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        n_epochs = self.cfg.get("n_epochs", 50)
        log_interval = self.cfg.get("log_interval", 10)

        for epoch in range(n_epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics   = self.validate(val_loader)

            if epoch % log_interval == 0:
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train L={train_metrics['L_total']:.4f} "
                    f"(ang={train_metrics['L_ang']:.4f}) | "
                    f"Val AE={val_metrics['mean_angular_error_deg']:.2f}° "
                    f"P90={val_metrics['p90_angular_error_deg']:.2f}°"
                )

            # Save best model
            val_loss = val_metrics["mean_angular_error_deg"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "cfg": self.cfg,
                }, self.checkpoint_dir / "best_corrector.pt")

        print(f"Training complete. Best val AE: {self.best_val_loss:.2f}°")


def build_dataloaders(cfg: dict):
    """Convenience function to build train/val loaders"""
    train_ds = GazeDataset(cfg["data_root"], split="train", augment=True)
    val_ds   = GazeDataset(cfg["data_root"], split="val",   augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.get("batch_size", 64),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader