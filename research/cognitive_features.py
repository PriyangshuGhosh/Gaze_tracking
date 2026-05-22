"""
Research Feature Extraction
  - Fixation detection (I-VT + I-DT hybrid)
  - Saccade detection + metrics
  - Spatial entropy (Hs) and Temporal entropy (Ht)
  - Extended Engagement Duration (EED) — Cognitive Hangover metric
  - Rider Distraction Index (RDI) — composite score
  - VOR disruption metric
  - OKR (Optokinetic Reflex) disruption detection
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Deque, Dict
from collections import deque
import scipy.stats as stats


# ─────────────────────────────────────────────
# Gaze Event Structures
# ─────────────────────────────────────────────

@dataclass
class Fixation:
    start_time: float
    end_time: float
    mean_gaze: np.ndarray     # (2,) mean azimuth, elevation
    std_gaze: float           # dispersion
    duration_ms: float
    object_label: Optional[str] = None


@dataclass
class Saccade:
    start_time: float
    end_time: float
    amplitude_deg: float      # angular amplitude
    peak_velocity_deg_s: float
    direction_deg: float      # 0° = right, CCW positive
    is_return_saccade: bool   # True if returning toward previous fixation


@dataclass
class GazeEvent:
    event_type: str           # "fixation" | "saccade" | "smooth_pursuit" | "blink"
    fixation: Optional[Fixation] = None
    saccade: Optional[Saccade] = None
    timestamp: float = 0.0


# ─────────────────────────────────────────────
# Velocity-Threshold Fixation Detector (I-VT)
# ─────────────────────────────────────────────

class FixationDetector:
    """
    I-VT (Velocity Threshold) method with minimum duration filter.

    Fixation:  velocity < fixation_thresh for >= min_fixation_duration_ms
    Saccade:   velocity > saccade_thresh
    Buffer: rolling window at 200 Hz
    """

    def __init__(self, cfg: dict):
        self.fps = cfg.get("fps", 200.0)
        self.dt  = 1.0 / self.fps
        self.fixation_vel_thresh  = cfg.get("fixation_vel_thresh_deg_s", 30.0)
        self.saccade_vel_thresh   = cfg.get("saccade_vel_thresh_deg_s", 100.0)
        self.min_fixation_dur_ms  = cfg.get("min_fixation_dur_ms", 80.0)
        self.min_saccade_dur_ms   = cfg.get("min_saccade_dur_ms", 10.0)

        self._timestamps: Deque[float] = deque(maxlen=500)
        self._gaze: Deque[np.ndarray]  = deque(maxlen=500)
        self._vel: Deque[float]        = deque(maxlen=500)
        self._labels: Deque[str]       = deque(maxlen=500)

        self.current_fixation: Optional[dict] = None
        self.current_saccade: Optional[dict]  = None
        self.events: List[GazeEvent] = []

    def update(
        self,
        gaze: np.ndarray,      # (2,) [azimuth, elevation] in degrees
        velocity: np.ndarray,  # (2,) deg/s
        timestamp: float,
        object_label: Optional[str] = None,
    ) -> Optional[GazeEvent]:
        speed = float(np.linalg.norm(velocity))

        self._timestamps.append(timestamp)
        self._gaze.append(gaze.copy())
        self._vel.append(speed)

        label = self._classify_velocity(speed)
        self._labels.append(label)

        # State machine
        return self._process_state(timestamp, gaze, speed, label, object_label)

    def _classify_velocity(self, speed: float) -> str:
        if speed < self.fixation_vel_thresh:
            return "fixation"
        elif speed > self.saccade_vel_thresh:
            return "saccade"
        else:
            return "transition"

    def _process_state(
        self, t: float, gaze: np.ndarray, speed: float,
        label: str, obj: Optional[str]
    ) -> Optional[GazeEvent]:
        event = None

        if label == "fixation":
            if self.current_saccade is not None:
                # Saccade just ended
                sac = self.current_saccade
                dur_ms = (t - sac["start"]) * 1000.0
                if dur_ms >= self.min_saccade_dur_ms:
                    event = GazeEvent(
                        event_type="saccade",
                        saccade=Saccade(
                            start_time=sac["start"],
                            end_time=t,
                            amplitude_deg=sac["amplitude"],
                            peak_velocity_deg_s=sac["peak_vel"],
                            direction_deg=sac["direction"],
                            is_return_saccade=sac.get("is_return", False),
                        ),
                        timestamp=t,
                    )
                    self.events.append(event)
                self.current_saccade = None

            if self.current_fixation is None:
                self.current_fixation = {
                    "start": t, "gazes": [gaze.copy()], "object": obj
                }
            else:
                self.current_fixation["gazes"].append(gaze.copy())
                self.current_fixation["object"] = obj  # update with latest

        elif label == "saccade":
            if self.current_fixation is not None:
                # Fixation just ended
                fix = self.current_fixation
                gazes_arr = np.array(fix["gazes"])
                dur_ms = (t - fix["start"]) * 1000.0
                if dur_ms >= self.min_fixation_dur_ms:
                    event = GazeEvent(
                        event_type="fixation",
                        fixation=Fixation(
                            start_time=fix["start"],
                            end_time=t,
                            mean_gaze=gazes_arr.mean(axis=0),
                            std_gaze=float(gazes_arr.std()),
                            duration_ms=dur_ms,
                            object_label=fix["object"],
                        ),
                        timestamp=t,
                    )
                    self.events.append(event)
                self.current_fixation = None

                # Start saccade: compute direction from last fixation
                if len(self.events) >= 2:
                    prev_fix_gaze = np.zeros(2)
                    for ev in reversed(self.events[:-1]):
                        if ev.fixation is not None:
                            prev_fix_gaze = ev.fixation.mean_gaze
                            break
                    diff = gaze - prev_fix_gaze
                    direction = float(np.degrees(np.arctan2(diff[1], diff[0])))
                    amplitude = float(np.linalg.norm(diff))
                else:
                    direction, amplitude = 0.0, 0.0

                self.current_saccade = {
                    "start": t, "peak_vel": speed,
                    "amplitude": amplitude, "direction": direction,
                }
            elif self.current_saccade is not None:
                self.current_saccade["peak_vel"] = max(
                    self.current_saccade["peak_vel"], speed
                )

        return event


# ─────────────────────────────────────────────
# Entropy Metrics
# ─────────────────────────────────────────────

class GazeEntropyAnalyzer:
    """
    Spatial Entropy Hs: how widely is gaze distributed across scene?
    Temporal Entropy Ht: how irregular are gaze transitions over time?

    Both use Shannon entropy H = -Σ p_i * log2(p_i)

    Grid-based spatial binning: scene divided into 8x6 zones.
    Temporal: histogram of inter-fixation intervals.
    """

    def __init__(self, cfg: dict):
        self.n_zones_x = cfg.get("n_zones_x", 8)
        self.n_zones_y = cfg.get("n_zones_y", 6)
        self.n_zones = self.n_zones_x * self.n_zones_y

        # Gaze range in degrees (scene FoV approximation)
        self.az_range  = cfg.get("az_range_deg",  [-60.0, 60.0])
        self.el_range  = cfg.get("el_range_deg",  [-40.0, 40.0])

        # Temporal histogram: inter-fixation intervals in ms, 0-2000ms
        self.ifi_bins = np.linspace(0, 2000, 21)  # 20 bins

        self._fixation_positions: List[np.ndarray] = []
        self._fixation_durations: List[float] = []
        self._ifi_list: List[float] = []
        self._last_fixation_end: Optional[float] = None

    def add_fixation(self, fixation: Fixation):
        self._fixation_positions.append(fixation.mean_gaze.copy())
        self._fixation_durations.append(fixation.duration_ms)

        # Inter-fixation interval
        if self._last_fixation_end is not None:
            ifi_ms = (fixation.start_time - self._last_fixation_end) * 1000.0
            self._ifi_list.append(max(0.0, ifi_ms))
        self._last_fixation_end = fixation.end_time

    def compute_spatial_entropy(self) -> float:
        """
        Hs = -Σ_i p_i * log2(p_i)
        where p_i = fraction of fixations in zone i

        Max entropy = log2(n_zones) (uniform distribution)
        Normalized Hs_norm = Hs / log2(n_zones) ∈ [0, 1]
        """
        if len(self._fixation_positions) < 2:
            return 0.0

        positions = np.array(self._fixation_positions)
        az = positions[:, 0]
        el = positions[:, 1]

        # Bin into zones
        az_bins = np.linspace(self.az_range[0], self.az_range[1], self.n_zones_x + 1)
        el_bins = np.linspace(self.el_range[0], self.el_range[1], self.n_zones_y + 1)

        hist, _, _ = np.histogram2d(az, el, bins=[az_bins, el_bins])
        hist_flat = hist.flatten()
        total = hist_flat.sum()
        if total < 1e-9:
            return 0.0

        probs = hist_flat / total
        probs = probs[probs > 0]
        Hs = -np.sum(probs * np.log2(probs))
        Hs_max = np.log2(self.n_zones)
        return float(Hs / Hs_max) if Hs_max > 0 else 0.0

    def compute_temporal_entropy(self) -> float:
        """
        Ht = -Σ_i p_i * log2(p_i) over IFI histogram bins
        Higher Ht = more irregular gaze timing (potential distraction)
        """
        if len(self._ifi_list) < 3:
            return 0.0

        hist, _ = np.histogram(self._ifi_list, bins=self.ifi_bins)
        total = hist.sum()
        if total < 1e-9:
            return 0.0

        probs = hist / total
        probs = probs[probs > 0]
        Ht = -np.sum(probs * np.log2(probs))
        Ht_max = np.log2(len(self.ifi_bins) - 1)
        return float(Ht / Ht_max) if Ht_max > 0 else 0.0

    def reset(self):
        self._fixation_positions.clear()
        self._fixation_durations.clear()
        self._ifi_list.clear()
        self._last_fixation_end = None


# ─────────────────────────────────────────────
# EED — Extended Engagement Duration (Cognitive Hangover)
# ─────────────────────────────────────────────

class EEDAnalyzer:
    """
    EED measures how long gaze stays on a single object/region BEYOND normal.

    Normal maximum fixation duration (empirical, driving context): ~800ms
    EED threshold: configurable (default 800ms)

    EED event: a fixation that exceeds EED_threshold_ms.
    Cognitive Hangover Index (CHI) = mean(EED events) / EED_threshold

    Also tracks:
      - EED frequency (EED events / minute)
      - Maximum EED duration in window
      - EED by object category
    """

    def __init__(self, cfg: dict):
        self.eed_threshold_ms = cfg.get("eed_threshold_ms", 800.0)
        self.window_size      = cfg.get("window_size", 100)  # fixations
        self._fixation_buffer: Deque[Fixation] = deque(maxlen=self.window_size)

    def add_fixation(self, fixation: Fixation):
        self._fixation_buffer.append(fixation)

    def compute(self) -> dict:
        """
        Returns:
          eed_events:          list of fixations exceeding threshold
          eed_count:           number of EED events
          chi:                 Cognitive Hangover Index [0, ∞)
          eed_frequency_per_min: EED events per minute
          max_eed_ms:          longest EED event duration
          eed_by_object:       {label: count}
        """
        if not self._fixation_buffer:
            return {"eed_events": [], "eed_count": 0, "chi": 0.0,
                    "eed_frequency_per_min": 0.0, "max_eed_ms": 0.0,
                    "eed_by_object": {}}

        fixations = list(self._fixation_buffer)
        eed_events = [f for f in fixations if f.duration_ms > self.eed_threshold_ms]

        eed_count = len(eed_events)
        if eed_count == 0:
            return {"eed_events": [], "eed_count": 0, "chi": 0.0,
                    "eed_frequency_per_min": 0.0, "max_eed_ms": 0.0,
                    "eed_by_object": {}}

        durations = np.array([f.duration_ms for f in eed_events])
        chi = float(durations.mean() / self.eed_threshold_ms)
        max_eed = float(durations.max())

        # Time span of window
        t_span_s = fixations[-1].end_time - fixations[0].start_time
        t_span_min = max(t_span_s / 60.0, 1e-6)
        freq = eed_count / t_span_min

        eed_by_object: Dict[str, int] = {}
        for f in eed_events:
            label = f.object_label or "unknown"
            eed_by_object[label] = eed_by_object.get(label, 0) + 1

        return {
            "eed_events": eed_events,
            "eed_count": eed_count,
            "chi": chi,
            "eed_frequency_per_min": freq,
            "max_eed_ms": max_eed,
            "eed_by_object": eed_by_object,
        }


# ─────────────────────────────────────────────
# RDI — Rider Distraction Index
# ─────────────────────────────────────────────

class RDIComputer:
    """
    Rider Distraction Index (RDI) — composite cognitive load / distraction score.

    RDI = w1*PERCLOS + w2*(1 - Hs_norm) + w3*CHI + w4*VOR_mismatch_norm
          + w5*saccade_inhibition + w6*Ht_norm

    Component definitions:
      PERCLOS:             fraction of time eyelid closure > 80% (over 1 min window)
      Hs_norm:             normalized spatial entropy [0,1] (low = narrow focus = distracted)
      CHI:                 Cognitive Hangover Index from EED
      VOR_mismatch_norm:   mean VOR mismatch / 2.0 (normalized)
      saccade_inhibition:  reduction in saccade rate vs. baseline
      Ht_norm:             temporal entropy (high = erratic timing)

    Weights: default per initial research; adjustable via calibration.
    RDI ∈ [0, 1] (clipped), thresholds: 0-0.3 normal, 0.3-0.6 moderate, >0.6 high
    """

    WEIGHTS = {
        "perclos":             0.25,
        "spatial_focus":       0.20,  # (1 - Hs_norm)
        "chi":                 0.20,
        "vor_mismatch":        0.15,
        "saccade_inhibition":  0.10,
        "temporal_entropy":    0.10,
    }

    # Thresholds
    PERCLOS_THRESH = 0.8          # eyelid closure fraction for PERCLOS
    BASELINE_SACCADE_RATE = 3.0   # saccades per second (normal driving)
    VOR_MISMATCH_MAX = 2.0        # normalization factor

    def __init__(self, cfg: dict = {}):
        self.weights = cfg.get("weights", self.WEIGHTS.copy())
        self.perclos_window_s = cfg.get("perclos_window_s", 60.0)

        self._eyelid_buffer:  Deque[tuple[float, float]] = deque()  # (t, closure)
        self._saccade_times:  Deque[float] = deque(maxlen=200)
        self._vor_buffer:     Deque[float] = deque(maxlen=200)
        self._rdi_history:    Deque[float] = deque(maxlen=500)

    def add_eyelid_sample(self, timestamp: float, closure: float):
        self._eyelid_buffer.append((timestamp, closure))
        # Prune old samples
        cutoff = timestamp - self.perclos_window_s
        while self._eyelid_buffer and self._eyelid_buffer[0][0] < cutoff:
            self._eyelid_buffer.popleft()

    def add_saccade(self, timestamp: float):
        self._saccade_times.append(timestamp)

    def add_vor_mismatch(self, mismatch: float):
        self._vor_buffer.append(mismatch)

    def compute(
        self,
        hs_norm: float,
        ht_norm: float,
        chi: float,
    ) -> dict:
        """
        Compute full RDI and all components.
        """
        # PERCLOS
        perclos = self._compute_perclos()

        # Spatial focus = 1 - Hs (low entropy = narrow/fixated gaze)
        spatial_focus = 1.0 - np.clip(hs_norm, 0.0, 1.0)

        # CHI normalization (cap at 3x threshold)
        chi_norm = np.clip(chi / 3.0, 0.0, 1.0)

        # VOR mismatch
        vor_mean = float(np.mean(self._vor_buffer)) if self._vor_buffer else 0.0
        vor_norm = np.clip(vor_mean / self.VOR_MISMATCH_MAX, 0.0, 1.0)

        # Saccade inhibition
        saccade_inhibition = self._compute_saccade_inhibition()

        # Temporal entropy
        ht_component = np.clip(ht_norm, 0.0, 1.0)

        components = {
            "perclos":            float(perclos),
            "spatial_focus":      float(spatial_focus),
            "chi":                float(chi_norm),
            "vor_mismatch":       float(vor_norm),
            "saccade_inhibition": float(saccade_inhibition),
            "temporal_entropy":   float(ht_component),
        }

        rdi = sum(self.weights[k] * v for k, v in components.items())
        rdi = float(np.clip(rdi, 0.0, 1.0))

        level = "normal" if rdi < 0.3 else ("moderate" if rdi < 0.6 else "high")

        self._rdi_history.append(rdi)

        return {
            "rdi": rdi,
            "level": level,
            "components": components,
            "trend": self._compute_trend(),
        }

    def _compute_perclos(self) -> float:
        if not self._eyelid_buffer:
            return 0.0
        closures = np.array([c for _, c in self._eyelid_buffer])
        return float((closures < (1.0 - self.PERCLOS_THRESH)).mean())

    def _compute_saccade_inhibition(self) -> float:
        """Returns 0 (normal rate) to 1 (complete inhibition)"""
        if len(self._saccade_times) < 2:
            return 0.5  # uncertain

        t_arr = np.array(list(self._saccade_times))
        t_span = t_arr[-1] - t_arr[0]
        if t_span < 1.0:
            return 0.5

        actual_rate = len(t_arr) / t_span
        inhibition = np.clip(
            1.0 - actual_rate / self.BASELINE_SACCADE_RATE, 0.0, 1.0
        )
        return float(inhibition)

    def _compute_trend(self) -> str:
        """Linear trend over last 50 RDI samples"""
        if len(self._rdi_history) < 10:
            return "stable"
        arr = np.array(list(self._rdi_history)[-50:])
        slope, _, _, pval, _ = stats.linregress(np.arange(len(arr)), arr)
        if pval > 0.1:
            return "stable"
        return "increasing" if slope > 0.001 else "decreasing"


# ─────────────────────────────────────────────
# VOR + OKR Disruption Metric
# ─────────────────────────────────────────────

class VORDisruptionMetric:
    """
    VOR gain = actual_eye_velocity / expected_compensatory_velocity
    Normal VOR gain: 0.9 - 1.1

    VOR disruption: |gain - 1.0| > 0.2 for > 100ms

    OKR (Optokinetic Reflex) detection:
      Smooth pursuit + low slip velocity when background moves uniformly
      OKR disruption: slip velocity > OKR_slip_threshold
    """

    def __init__(self, cfg: dict):
        self.vor_gain_normal_range = cfg.get("vor_gain_normal", [0.85, 1.15])
        self.vor_disruption_thresh_ms = cfg.get("vor_disruption_thresh_ms", 100.0)
        self.okr_slip_threshold_deg_s = cfg.get("okr_slip_thresh", 8.0)

        self._vor_gains: Deque[tuple[float, float]] = deque(maxlen=500)  # (t, gain)
        self._disruption_start: Optional[float] = None
        self._disruption_events: List[dict] = []

    def update(
        self,
        timestamp: float,
        head_omega: np.ndarray,     # (3,) rad/s
        eye_omega: np.ndarray,      # (3,) rad/s — from gaze velocity
    ):
        head_speed = float(np.linalg.norm(head_omega))
        eye_speed  = float(np.linalg.norm(eye_omega))

        if head_speed < 0.05:  # < ~3 deg/s: not meaningful
            return

        # VOR gain = eye_compensatory_speed / head_speed
        # Eye should move opposite to head → expected_eye_omega = -head_omega
        expected_eye_speed = head_speed
        gain = eye_speed / (expected_eye_speed + 1e-9)
        self._vor_gains.append((timestamp, gain))

        # Disruption detection
        lo, hi = self.vor_gain_normal_range
        disrupted = not (lo <= gain <= hi)

        if disrupted:
            if self._disruption_start is None:
                self._disruption_start = timestamp
        else:
            if self._disruption_start is not None:
                dur_ms = (timestamp - self._disruption_start) * 1000.0
                if dur_ms >= self.vor_disruption_thresh_ms:
                    self._disruption_events.append({
                        "start": self._disruption_start,
                        "end": timestamp,
                        "duration_ms": dur_ms,
                        "mean_gain": float(np.mean([g for _, g in self._vor_gains
                                                     if self._disruption_start <= _ <= timestamp])),
                    })
                self._disruption_start = None

    def get_vor_stats(self) -> dict:
        if not self._vor_gains:
            return {"mean_gain": 1.0, "std_gain": 0.0, "disruption_count": 0,
                    "total_disruption_ms": 0.0}

        gains = np.array([g for _, g in self._vor_gains])
        total_dis_ms = sum(e["duration_ms"] for e in self._disruption_events)

        return {
            "mean_gain": float(gains.mean()),
            "std_gain": float(gains.std()),
            "disruption_count": len(self._disruption_events),
            "total_disruption_ms": total_dis_ms,
            "disruption_events": self._disruption_events[-10:],  # last 10
        }

    def compute_okr_disruption(
        self,
        eye_velocity: np.ndarray,         # (2,) deg/s
        optic_flow_velocity: np.ndarray,  # (2,) deg/s — from scene optical flow
    ) -> float:
        """
        OKR slip = ||eye_velocity - optic_flow_velocity||
        Returns slip in deg/s. High slip = OKR disruption.
        """
        slip = np.linalg.norm(eye_velocity - optic_flow_velocity)
        return float(slip)