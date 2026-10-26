"""Arm gesture player (arm_sdk weight ramping) and lateral stepper."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from .backend import Backend
from .clips import Clip, resample, smoothstep

log = logging.getLogger(__name__)


class Refused(Exception):
    pass


class ArmPlayer:
    """Plays one clip at a time:
         ramp-in   weight 0 -> 1 holding the MEASURED pose (no motion)
         approach  measured pose -> clip start (smoothstep)
         play      clip frames at control rate
         ramp-out  weight 1 -> 0 holding the last target
       stop() jumps straight to ramp-out from the current target.
       A marker file exists whenever weight may be > 0, so a crash is detectable.
    """

    def __init__(self, backend: Backend, *, control_hz: float = 50.0, kp: float = 60.0, kd: float = 1.5,
                 ramp_in_s: float = 1.0, approach_s: float = 1.0, ramp_out_s: float = 1.5,
                 hold_joints: Sequence[int] = (), start_pose_tolerance_rad: float = 0.35,
                 marker_path: str = "state/arm_active",
                 on_done: Optional[Callable[[str], None]] = None):
        if ramp_out_s > 2.0:
            raise ValueError("ramp_out_s must be <= 2.0")
        self.b = backend
        self.hz, self.kp, self.kd = control_hz, kp, kd
        self.ramp_in_s, self.approach_s, self.ramp_out_s = ramp_in_s, approach_s, ramp_out_s
        self.hold_joints = list(hold_joints)
        self.tol = start_pose_tolerance_rad
        self.marker = marker_path
        self.on_done = on_done
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.phase = "idle"
        self.weight = 0.0
        self.tick_errors_ms: List[float] = []

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def play(self, clip: Clip, speed_scale: float = 1.0) -> None:
        if self.active:
            raise Refused("a gesture is already playing")
        joints = list(clip.joints) + [j for j in self.hold_joints if j not in clip.joints]
        q_meas, t_state = self.b.read_q(joints)
        if time.monotonic() - t_state > 0.2:
            raise Refused("robot state is stale (>200 ms)")
        n_clip = len(clip.joints)
        dev = float(np.max(np.abs(q_meas[:n_clip] - clip.frames[0])))
        if dev > self.tol:
            raise Refused(f"arms are {dev:.2f} rad from the clip start pose (> {self.tol})")
        frames = resample(clip, self.hz, speed_scale)
        hold = q_meas[n_clip:]
        kp = clip.kp if clip.kp is not None else self.kp
        kd = clip.kd if clip.kd is not None else self.kd
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(clip.name, joints, q_meas, frames, hold, kp, kd),
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # -------------------------------------------------------------- internals
    def _marker(self, on: bool) -> None:
        if not self.marker:
            return
        try:
            if on:
                os.makedirs(os.path.dirname(self.marker) or ".", exist_ok=True)
                with open(self.marker, "w") as fh:
                    fh.write(str(time.time()))
            elif os.path.exists(self.marker):
                os.remove(self.marker)
        except OSError as exc:
            log.error("marker file: %s", exc)

    def _tick(self, t_next: float) -> float:
        now = time.perf_counter()
        if t_next > now:
            time.sleep(t_next - now)
        err = (time.perf_counter() - t_next) * 1000.0
        self.tick_errors_ms.append(err)
        if len(self.tick_errors_ms) > 5000:
            del self.tick_errors_ms[:2500]
        return t_next + 1.0 / self.hz

    def _send(self, joints, q, kp, kd, w) -> None:
        self.weight = w
        self.b.send_arm(joints, q, kp, kd, w)

    def _run(self, name, joints, q_meas, frames, hold, kp, kd) -> None:
        dt = 1.0 / self.hz
        target = q_meas.copy()
        reason = "finished"
        self._marker(True)
        t_next = time.perf_counter()
        try:
            self.phase = "ramp_in"
            for w in np.linspace(0.0, 1.0, max(2, int(self.ramp_in_s / dt))):
                if self._stop.is_set():
                    reason = "stopped"
                    break
                self._send(joints, target, kp, kd, float(w))
                t_next = self._tick(t_next)
            if reason == "finished":
                self.phase = "approach"
                start = np.concatenate([frames[0], hold])
                for s in smoothstep(max(2, int(self.approach_s / dt))):
                    if self._stop.is_set():
                        reason = "stopped"
                        break
                    target = q_meas + (start - q_meas) * s
                    self._send(joints, target, kp, kd, 1.0)
                    t_next = self._tick(t_next)
            if reason == "finished":
                self.phase = "play"
                for f in frames:
                    if self._stop.is_set():
                        reason = "stopped"
                        break
                    target = np.concatenate([f, hold])
                    self._send(joints, target, kp, kd, 1.0)
                    t_next = self._tick(t_next)
        except Exception as exc:
            log.exception("arm player error")
            reason = f"error: {exc}"
        finally:
            self._ramp_out(joints, target, kp, kd, dt)
            self.phase = "idle"
            if self.on_done:
                self.on_done(reason)

    def _ramp_out(self, joints, target, kp, kd, dt) -> None:
        self.phase = "ramp_out"
        w0 = self.weight
        t_next = time.perf_counter()
        for w in np.linspace(w0, 0.0, max(2, int(self.ramp_out_s * max(w0, 0.05) / dt))):
            try:
                self._send(joints, target, kp, kd, float(w))
            except Exception as exc:
                log.error("send during ramp-out failed: %s", exc)
            t_next = self._tick(t_next)
        self.weight = 0.0
        self._marker(False)

    def release_from_unknown(self, joints: Sequence[int]) -> None:
        """Operator recovery after a crash: hold MEASURED pose at weight 1, ramp to 0."""
        if self.active:
            raise Refused("player is active")
        q, _ = self.b.read_q(joints)
        self.weight = 1.0
        self._ramp_out(list(joints), q, self.kp, self.kd, 1.0 / self.hz)

    def jitter_ms(self) -> dict:
        e = np.asarray(self.tick_errors_ms[-1000:])
        if e.size == 0:
            return {}
        return {"p50": round(float(np.percentile(e, 50)), 3), "p99": round(float(np.percentile(e, 99)), 3),
                "max": round(float(e.max()), 3)}


class Stepper:
    """Lateral-only steps: SetVelocity(0, vy, 0, <=1 s) re-sent until the step time ends."""

    def __init__(self, backend: Backend, *, max_vy: float = 0.2, max_duration_s: float = 2.0,
                 resend_period_s: float = 0.3, on_done: Optional[Callable[[str], None]] = None):
        self.b = backend
        self.max_vy, self.max_dur, self.period = max_vy, max_duration_s, resend_period_s
        self.on_done = on_done
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def step(self, vy: float, duration_s: float) -> None:
        if self.active:
            raise Refused("a step is already running")
        if abs(vy) > self.max_vy + 1e-9:
            raise Refused(f"|vy| {abs(vy)} > {self.max_vy}")
        if not (0 < duration_s <= self.max_dur):
            raise Refused(f"duration {duration_s} outside (0, {self.max_dur}]")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(vy, duration_s), daemon=True)
        self._thread.start()

    def _run(self, vy: float, duration_s: float) -> None:
        reason = "finished"
        t_end = time.monotonic() + duration_s
        try:
            while True:
                left = t_end - time.monotonic()
                if left <= 0 or self._stop.is_set():
                    reason = "stopped" if self._stop.is_set() else reason
                    break
                self.b.set_velocity(0.0, vy, 0.0, min(1.0, left + 0.05))
                self._stop.wait(min(self.period, left))
        except Exception as exc:
            log.exception("step error")
            reason = f"error: {exc}"
        finally:
            self.zero()
            if self.on_done:
                self.on_done(reason)

    def zero(self) -> None:
        try:
            self.b.set_velocity(0.0, 0.0, 0.0, 1.0)
        except Exception as exc:
            log.error("zero velocity failed: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        self.zero()

    def wait(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
