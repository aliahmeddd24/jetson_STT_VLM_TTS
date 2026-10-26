"""Gesture clips: format, resampling, validation and conversion from avp_teleoperate.

Clip file (JSON):
  {"name": "wave", "fps": 50, "joints": [15, ..., 28], "frames": [[q...], ...],
   "kp": optional float, "kd": optional float, "source": "...", "notes": "..."}
Frames are joint POSITION targets in radians (never torques or velocities).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import ARMS, G1_JOINTS, LEFT_ARM, RIGHT_ARM


@dataclass
class Clip:
    name: str
    fps: float
    joints: List[int]
    frames: np.ndarray                 # (T, J)
    kp: Optional[float] = None
    kd: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return (len(self.frames) - 1) / self.fps

    def to_json(self) -> dict:
        d = {"name": self.name, "fps": self.fps, "joints": self.joints,
             "frames": np.round(self.frames, 5).tolist()}
        if self.kp is not None:
            d["kp"] = self.kp
        if self.kd is not None:
            d["kd"] = self.kd
        d.update(self.meta)
        return d


def load_clip(path: str) -> Clip:
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    frames = np.asarray(d["frames"], dtype=np.float64)
    meta = {k: v for k, v in d.items() if k not in ("name", "fps", "joints", "frames", "kp", "kd")}
    return Clip(d["name"], float(d["fps"]), [int(j) for j in d["joints"]], frames,
                d.get("kp"), d.get("kd"), meta)


def save_clip(clip: Clip, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(clip.to_json(), fh, indent=1)


def resample(clip: Clip, out_hz: float, speed_scale: float = 1.0) -> np.ndarray:
    """Linear resample to the control rate. speed_scale < 1 plays slower."""
    if speed_scale <= 0:
        raise ValueError("speed_scale must be > 0")
    T = len(clip.frames)
    src_t = np.arange(T) / clip.fps / speed_scale
    n = int(np.floor(src_t[-1] * out_hz)) + 1
    dst_t = np.arange(n) / out_hz
    out = np.empty((n, clip.frames.shape[1]))
    for j in range(clip.frames.shape[1]):
        out[:, j] = np.interp(dst_t, src_t, clip.frames[:, j])
    out[-1] = clip.frames[-1]
    return out


def smoothstep(n: int) -> np.ndarray:
    s = np.linspace(0.0, 1.0, max(n, 2))
    return s * s * (3 - 2 * s)


# ------------------------------------------------------------------ validation
@dataclass
class Rules:
    max_clip_s: float = 4.0
    max_speed_scale: float = 1.0
    limits_margin_rad: float = 0.05
    max_joint_vel_rad_s: float = 2.5
    neutral_tol_rad: float = 0.02
    bubble_frame: str = "torso_link"
    bubble_links: Sequence[str] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    bubble_min: Sequence[float] = (-0.15, -0.55, -0.45)
    bubble_max: Sequence[float] = (0.65, 0.55, 0.55)


def validate(clip: Clip, rules: Rules, speed_scale: float = 1.0, urdf=None,
             fk_stride: int = 5) -> List[str]:
    """Returns a list of problems; empty means the clip passes."""
    errs: List[str] = []
    F = clip.frames
    if F.ndim != 2 or F.shape[0] < 2 or F.shape[1] != len(clip.joints):
        return [f"frames must be (T>=2, {len(clip.joints)}), got {F.shape}"]
    if not np.all(np.isfinite(F)):
        errs.append("non-finite values in frames")
    bad = [j for j in clip.joints if j not in ARMS]
    if bad:
        errs.append(f"only arm joints {ARMS[0]}..{ARMS[-1]} may be in a clip, got {bad}")
    if len(set(clip.joints)) != len(clip.joints):
        errs.append("duplicate joints")
    if errs:
        return errs          # structural problems: further checks would be meaningless
    if not (0 < speed_scale <= rules.max_speed_scale):
        errs.append(f"speed_scale {speed_scale} outside (0, {rules.max_speed_scale}]")
    dur = clip.duration_s / max(speed_scale, 1e-6)
    if dur > rules.max_clip_s:
        errs.append(f"duration {dur:.2f} s > {rules.max_clip_s} s")
    d0 = np.max(np.abs(F[0] - F[-1]))
    if d0 > rules.neutral_tol_rad:
        errs.append(f"clip must start and end at the same neutral pose (max diff {d0:.3f} rad)")
    vel = np.max(np.abs(np.diff(F, axis=0))) * clip.fps * speed_scale
    if vel > rules.max_joint_vel_rad_s:
        errs.append(f"peak joint speed {vel:.2f} rad/s > {rules.max_joint_vel_rad_s}")
    if urdf is None:
        errs.append("no URDF given: joint limits and the workspace bubble were NOT checked")
        return errs
    m = rules.limits_margin_rad
    for k, j in enumerate(clip.joints):
        lo, hi = urdf.limits(G1_JOINTS[j])
        if F[:, k].min() < lo + m or F[:, k].max() > hi - m:
            errs.append(f"{G1_JOINTS[j]} range [{F[:, k].min():.3f}, {F[:, k].max():.3f}] "
                        f"outside [{lo + m:.3f}, {hi - m:.3f}] (URDF limits with {m} rad margin)")
    lo_b, hi_b = np.asarray(rules.bubble_min), np.asarray(rules.bubble_max)
    for t in range(0, len(F), max(1, fk_stride)):
        q = {G1_JOINTS[j]: float(F[t, k]) for k, j in enumerate(clip.joints)}
        for link in rules.bubble_links:
            p = urdf.position_in(link, rules.bubble_frame, q)
            if np.any(p < lo_b) or np.any(p > hi_b):
                errs.append(f"frame {t}: {link} at {np.round(p, 3).tolist()} leaves the bubble "
                            f"{list(lo_b)}..{list(hi_b)} in {rules.bubble_frame}")
                return errs
    return errs


def clamp_to_limits(clip: Clip, urdf, margin: float) -> Clip:
    F = clip.frames.copy()
    for k, j in enumerate(clip.joints):
        lo, hi = urdf.limits(G1_JOINTS[j])
        F[:, k] = np.clip(F[:, k], lo + margin, hi - margin)
    return Clip(clip.name, clip.fps, clip.joints, F, clip.kp, clip.kd, dict(clip.meta))


# ------------------------------------------------------------------ conversion
def _frames_of(episode: dict) -> list:
    for key in ("data", "frames", "episode"):
        if isinstance(episode.get(key), list):
            return episode[key]
    raise ValueError("no frame list found (expected key 'data')")


def _arm_q(frame: dict, part: str, source: str) -> List[float]:
    block = frame.get(source) or {}
    arm = block.get(part) or {}
    q = arm.get("qpos", arm.get("q"))
    if q is None:
        raise KeyError(f"frame has no {source}.{part}.qpos")
    return [float(v) for v in q]


def convert_avp_episode(episode: dict, name: str, *, source: str = "actions", fps: Optional[float] = None,
                        start: int = 0, end: Optional[int] = None, smooth: int = 5,
                        out_fps: float = 50.0, blend_s: float = 0.5,
                        neutral: Optional[Sequence[float]] = None) -> Clip:
    """avp_teleoperate per-episode JSON -> Clip (both arms, joints 15..28).

    Format assumptions (VERIFY against a real episode): frames under 'data', each with
    {source: {'left_arm': {'qpos': [7]}, 'right_arm': {'qpos': [7]}}}; fps from
    info.image.fps unless given. The result is blended from/to `neutral` (default:
    the first recorded pose) so it starts and ends at the same pose.
    """
    frames = _frames_of(episode)[start:end]
    if len(frames) < 2:
        raise ValueError("need at least 2 frames")
    if fps is None:
        fps = float(((episode.get("info") or {}).get("image") or {}).get("fps") or 30.0)
    Q = np.array([_arm_q(f, "left_arm", source) + _arm_q(f, "right_arm", source) for f in frames])
    if Q.shape[1] != 14:
        raise ValueError(f"expected 7+7 arm joints, got {Q.shape[1]}")
    if smooth and smooth > 1:
        k = np.ones(smooth) / smooth
        pad = smooth // 2
        Qp = np.pad(Q, ((pad, smooth - 1 - pad), (0, 0)), mode="edge")
        Q = np.stack([np.convolve(Qp[:, j], k, mode="valid") for j in range(14)], axis=1)
    clip = Clip(name, fps, LEFT_ARM + RIGHT_ARM, Q)
    body = resample(clip, out_fps)
    n0 = np.asarray(neutral, dtype=float) if neutral is not None else body[0].copy()
    nb = max(2, int(blend_s * out_fps))
    s = smoothstep(nb)[:, None]
    head = n0 + (body[0] - n0) * s
    tail = body[-1] + (n0 - body[-1]) * s
    F = np.vstack([head[:-1], body, tail[1:]])
    return Clip(name, out_fps, LEFT_ARM + RIGHT_ARM, F,
                meta={"source": f"avp_teleoperate:{source}", "recorded_fps": fps})
