"""Local-only HTTP API for gestures and steps, with heartbeat watchdog.

  POST /heartbeat                 orchestrator liveness (motion requires a fresh one)
  POST /gesture   {"name": ..}    play an allowlisted, validated clip
  POST /step      {"dir": "left"|"right"}
  POST /stop                      zero velocity + arm weight ramp-out (always allowed)
  GET  /status
  POST /operator/walking {"enabled": bool}   needs X-Operator-Token
  POST /operator/release-arms                needs X-Operator-Token (crash recovery)
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple

import yaml

from . import ARMS
from .backend import Backend
from .clips import Clip, Rules, load_clip, validate
from .motion import ArmPlayer, Refused, Stepper

log = logging.getLogger(__name__)


def load_manifest(path: str, rules_override: Optional[dict] = None, strict: bool = True,
                  urdf_obj=None) -> Tuple[Dict[str, Tuple[Clip, float, bool]], Dict[str, str], Rules]:
    """-> ({name: (clip, speed_scale, llm_allowed)}, {name: problem}, rules).
    strict=True (live) requires a URDF-checked, problem-free clip."""
    base = os.path.dirname(os.path.abspath(path))
    with open(path, "r", encoding="utf-8") as fh:
        m = yaml.safe_load(fh) or {}
    r = dict(m.get("rules") or {})
    r.update(rules_override or {})
    rules = Rules(**r)
    urdf = urdf_obj
    urdf_path = m.get("urdf")
    if urdf is None and urdf_path:
        p = urdf_path if os.path.isabs(urdf_path) else os.path.join(base, urdf_path)
        if os.path.exists(p):
            from .urdf import URDF
            urdf = URDF(p)
        else:
            log.warning("URDF %s not found", p)
    ok: Dict[str, Tuple[Clip, float, bool]] = {}
    problems: Dict[str, str] = {}
    for name, g in (m.get("gestures") or {}).items():
        if not g or not g.get("enabled"):
            continue
        try:
            fp = g["file"] if os.path.isabs(g["file"]) else os.path.join(base, g["file"])
            clip = load_clip(fp)
            speed = float(g.get("speed_scale", 1.0))
            errs = validate(clip, rules, speed, urdf)
            soft = [e for e in errs if e.startswith("no URDF")]
            hard = [e for e in errs if not e.startswith("no URDF")]
            if hard or (strict and soft):
                problems[name] = "; ".join(hard + (soft if strict else []))
                continue
            if soft:
                log.warning("gesture %s: %s (allowed in dry-run only)", name, soft[0])
            ok[name] = (clip, speed, bool(g.get("llm_allowed", False)))
        except Exception as exc:
            problems[name] = f"load failed: {exc}"
    for n, p in problems.items():
        log.error("gesture %s disabled: %s", n, p)
    return ok, problems, rules


class ActionServer:
    def __init__(self, cfg: dict, backend: Backend, *, urdf_obj=None):
        self.cfg = cfg
        self.b = backend
        self.live = backend.live
        state_dir = cfg.get("state_dir", "state")
        os.makedirs(state_dir, exist_ok=True)
        self.marker = os.path.join(state_dir, "arm_active")
        self.arm_maybe_held = os.path.exists(self.marker)
        if self.arm_maybe_held:
            log.critical("marker %s exists: a previous run may have left arm_sdk weight > 0. "
                         "Gestures are blocked until an operator calls /operator/release-arms.", self.marker)
        a, lo = cfg["arm"], cfg["loco"]
        self.gestures, self.gesture_problems, self.rules = load_manifest(
            cfg["manifest"], {"max_clip_s": a["max_clip_s"], "max_speed_scale": a["max_speed_scale"]},
            strict=self.live, urdf_obj=urdf_obj)
        self.cooldown_s = a["cooldown_s"]
        self.player = ArmPlayer(backend, control_hz=a["control_hz"], kp=a["kp"], kd=a["kd"],
                                ramp_in_s=a["ramp_in_s"], approach_s=a["approach_s"], ramp_out_s=a["ramp_out_s"],
                                hold_joints=a.get("hold_joints", []),
                                start_pose_tolerance_rad=a["start_pose_tolerance_rad"],
                                marker_path=self.marker, on_done=self._done)
        self.stepper = Stepper(backend, max_vy=lo["max_vy"], max_duration_s=lo["max_duration_s"],
                               resend_period_s=lo["resend_period_s"], on_done=self._done)
        self.step_vy = min(float(lo["step_vy"]), float(lo["max_vy"]))
        self.step_s = min(float(lo["step_duration_s"]), float(lo["max_duration_s"]))
        mc = cfg.get("mode_check") or {}
        self.allowed_ids = set(mc.get("allowed_fsm_ids") or [])
        self.allowed_modes = set(mc.get("allowed_fsm_modes") or [])
        self.hb_timeout = float(cfg.get("heartbeat_timeout_s", 0.5))
        self.last_hb = 0.0
        self.walking_enabled = False
        self.last_end = 0.0
        self.last_event: dict = {}
        self.lock = threading.Lock()
        self.token = self._load_token(cfg.get("operator_token_file", os.path.join(state_dir, "operator_token")))
        self._stop = threading.Event()
        self._wd = threading.Thread(target=self._watchdog, daemon=True)
        self._wd.start()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _load_token(path: str) -> str:
        if os.path.exists(path):
            with open(path) as fh:
                return fh.read().strip()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tok = secrets.token_hex(16)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write(tok)
        log.info("operator token written to %s", path)
        return tok

    def _done(self, reason: str) -> None:
        with self.lock:
            self.last_end = time.monotonic()
            self.last_event = {"event": "motion_done", "reason": reason, "t": time.time()}

    def _refuse(self, why: str) -> dict:
        self.last_event = {"event": "refused", "reason": why, "t": time.time()}
        log.warning("refused: %s", why)
        return {"ok": False, "reason": why}

    def _precheck(self) -> Optional[str]:
        if time.monotonic() - self.last_hb > self.hb_timeout:
            return "no fresh heartbeat from the orchestrator"
        if self.player.active or self.stepper.active:
            return "another motion is running"
        if time.monotonic() - self.last_end < self.cooldown_s:
            return "cooldown"
        if not self.allowed_ids:
            return "mode_check.allowed_fsm_ids is empty (confirm FSM ids on the robot first)"
        fsm_id, fsm_mode = self.b.fsm()
        if fsm_id is None:
            return "cannot read FSM state"
        if fsm_id not in self.allowed_ids:
            return f"FSM id {fsm_id} not in allowed {sorted(self.allowed_ids)}"
        if self.allowed_modes and fsm_mode not in self.allowed_modes:
            return f"FSM mode {fsm_mode} not in allowed {sorted(self.allowed_modes)}"
        return None

    # ------------------------------------------------------------ API
    def heartbeat(self) -> dict:
        self.last_hb = time.monotonic()
        return {"ok": True}

    def gesture(self, name: str) -> dict:
        with self.lock:
            if name not in self.gestures:
                return self._refuse(f"gesture {name!r} not allowlisted/enabled/valid")
            if self.arm_maybe_held:
                return self._refuse("arm weight may be held from a crash: operator release required")
            why = self._precheck()
            if why:
                return self._refuse(why)
            clip, speed, _ = self.gestures[name]
            try:
                self.player.play(clip, speed)
            except Refused as exc:
                return self._refuse(str(exc))
            self.last_event = {"event": "gesture", "name": name, "t": time.time()}
            return {"ok": True, "dry_run": not self.live, "gesture": name}

    def step(self, direction: str) -> dict:
        with self.lock:
            if direction not in ("left", "right"):
                return self._refuse("dir must be left or right")
            if not self.walking_enabled:
                return self._refuse("walking is disabled (operator must enable it)")
            why = self._precheck()
            if why:
                return self._refuse(why)
            vy = self.step_vy if direction == "left" else -self.step_vy   # +y is the robot's left
            try:
                self.stepper.step(vy, self.step_s)
            except Refused as exc:
                return self._refuse(str(exc))
            self.last_event = {"event": "step", "dir": direction, "t": time.time()}
            return {"ok": True, "dry_run": not self.live, "vy": vy, "duration_s": self.step_s}

    def stop(self, reason: str = "api") -> dict:
        self.stepper.stop()
        self.player.stop()
        self.last_event = {"event": "stop", "reason": reason, "t": time.time()}
        return {"ok": True}

    def set_walking(self, enabled: bool) -> dict:
        self.walking_enabled = bool(enabled)
        if not enabled:
            self.stepper.stop()
        return {"ok": True, "walking_enabled": self.walking_enabled}

    def release_arms(self) -> dict:
        with self.lock:
            if self.player.active:
                return self._refuse("player active")
            self.player.release_from_unknown(ARMS)
            self.arm_maybe_held = False
            return {"ok": True}

    def status(self) -> dict:
        hb_age = time.monotonic() - self.last_hb if self.last_hb else None
        try:
            fsm_id, fsm_mode = self.b.fsm()
        except Exception as exc:
            fsm_id, fsm_mode = None, "error: %s" % exc
        return {"ok": True, "live": self.live, "fsm_id": fsm_id, "fsm_mode": fsm_mode, "walking_enabled": self.walking_enabled,
                "arm_phase": self.player.phase, "arm_weight": round(self.player.weight, 3),
                "stepping": self.stepper.active, "heartbeat_age_s": None if hb_age is None else round(hb_age, 3),
                "arm_maybe_held": self.arm_maybe_held, "gestures": sorted(self.gestures),
                "gesture_problems": self.gesture_problems, "allowed_fsm_ids": sorted(self.allowed_ids),
                "arm_jitter_ms": self.player.jitter_ms(), "last_event": self.last_event}

    def _watchdog(self) -> None:
        fired = False
        while not self._stop.wait(0.05):
            moving = self.player.active or self.stepper.active
            if not moving:
                fired = False
            elif not fired and time.monotonic() - self.last_hb > self.hb_timeout:
                fired = True          # once per motion: no repeated zero-velocity RPCs
                log.error("watchdog: heartbeat lost during motion, stopping")
                self.stop("watchdog")

    def close(self) -> None:
        self.stop("shutdown")
        self.player.wait(3.0)
        self.stepper.wait(3.0)
        self._stop.set()


def make_handler(srv: ActionServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _reply(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def _operator(self) -> bool:
            return secrets.compare_digest(self.headers.get("X-Operator-Token", ""), srv.token)

        def do_GET(self):
            if self.path == "/status":
                return self._reply(200, srv.status())
            self._reply(404, {"ok": False, "reason": "not found"})

        def do_POST(self):
            try:
                body = self._body()
            except ValueError:
                return self._reply(400, {"ok": False, "reason": "bad json"})
            p = self.path
            if p == "/heartbeat":
                res = srv.heartbeat()
            elif p == "/gesture":
                res = srv.gesture(str(body.get("name", "")))
            elif p == "/step":
                res = srv.step(str(body.get("dir", "")))
            elif p == "/stop":
                res = srv.stop()
            elif p.startswith("/operator/"):
                if not self._operator():
                    return self._reply(403, {"ok": False, "reason": "operator token required"})
                if p == "/operator/walking":
                    res = srv.set_walking(bool(body.get("enabled")))
                elif p == "/operator/release-arms":
                    res = srv.release_arms()
                else:
                    return self._reply(404, {"ok": False, "reason": "not found"})
            else:
                return self._reply(404, {"ok": False, "reason": "not found"})
            self._reply(200 if res.get("ok") else 409, res)

    return Handler


def serve(srv: ActionServer, host: str, port: int) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the action server must bind to loopback only")
    httpd = ThreadingHTTPServer((host, port), make_handler(srv))
    httpd.daemon_threads = True
    return httpd
