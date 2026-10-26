"""Robot backends. The action server talks only to this interface.

DryRunBackend simulates state and logs every command. UnitreeBackend publishes
LowCmd_ on rt/arm_sdk (position targets + PD gains, weight in motor 29) and calls
LocoClient.SetVelocity for small lateral steps. Nothing here can damp the robot or
remove torque: those calls are deliberately absent.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)


class Backend:
    live = False

    def read_q(self, joints: Sequence[int]) -> Tuple[np.ndarray, float]:
        """Measured joint positions and the monotonic time of that state."""
        raise NotImplementedError

    def send_arm(self, joints: Sequence[int], q: np.ndarray, kp: float, kd: float, weight: float) -> None:
        raise NotImplementedError

    def fsm(self) -> Tuple[Optional[int], Optional[int]]:
        """(fsm_id, fsm_mode); None where unknown."""
        raise NotImplementedError

    def set_velocity(self, vx: float, vy: float, vyaw: float, duration_s: float) -> None:
        raise NotImplementedError


class DryRunBackend(Backend):
    def __init__(self, fsm_id: Optional[int] = None, fsm_mode: Optional[int] = None,
                 initial_q: Optional[Dict[int, float]] = None):
        self.q = np.zeros(29)
        for j, v in (initial_q or {}).items():
            self.q[j] = v
        self.fsm_id, self.fsm_mode = fsm_id, fsm_mode
        self.weight = 0.0
        self.log: List[tuple] = []
        self._lock = threading.Lock()

    def read_q(self, joints):
        with self._lock:
            return self.q[list(joints)].copy(), time.monotonic()

    def send_arm(self, joints, q, kp, kd, weight):
        weight = float(np.clip(weight, 0.0, 1.0))
        with self._lock:
            idx = list(joints)
            # crude simulation: blend toward target in proportion to the weight
            self.q[idx] = self.q[idx] + weight * (np.asarray(q) - self.q[idx])
            self.weight = weight
            self.log.append(("arm", time.monotonic(), weight))

    def fsm(self):
        return self.fsm_id, self.fsm_mode

    def set_velocity(self, vx, vy, vyaw, duration_s):
        self.log.append(("vel", time.monotonic(), vx, vy, vyaw, duration_s))
        log.info("[dry-run] SetVelocity(%.2f, %.2f, %.2f, %.2f)", vx, vy, vyaw, duration_s)


class UnitreeBackend(Backend):
    live = True
    WEIGHT_JOINT = 29

    def __init__(self, iface: str, domain: int = 0, weight_joint: int = 29):
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,  # type: ignore
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # type: ignore
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # type: ignore
        from unitree_sdk2py.utils.crc import CRC  # type: ignore
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient  # type: ignore

        ChannelFactoryInitialize(domain, iface)
        self.WEIGHT_JOINT = weight_joint
        self._crc = CRC()
        self._cmd = unitree_hg_msg_dds__LowCmd_()
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        self._state = None
        self._state_t = 0.0
        self._lock = threading.Lock()
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        self.loco = LocoClient()
        self.loco.SetTimeout(3.0)
        self.loco.Init()

    def _on_state(self, msg) -> None:
        with self._lock:
            self._state = msg
            self._state_t = time.monotonic()

    def read_q(self, joints):
        with self._lock:
            if self._state is None:
                raise RuntimeError("no rt/lowstate received yet")
            return np.array([self._state.motor_state[j].q for j in joints]), self._state_t

    def send_arm(self, joints, q, kp, kd, weight):
        weight = float(np.clip(weight, 0.0, 1.0))
        cmd = self._cmd
        cmd.motor_cmd[self.WEIGHT_JOINT].q = weight
        for j, qj in zip(joints, q):
            m = cmd.motor_cmd[j]
            m.q = float(qj)
            m.dq = 0.0
            m.tau = 0.0
            m.kp = float(kp)
            m.kd = float(kd)
        cmd.crc = self._crc.Crc(cmd)
        self._pub.Write(cmd)

    def fsm(self):
        fsm_id = self._rpc_int("GetFsmId", 7001)
        fsm_mode = self._rpc_int("GetFsmMode", 7002)
        return fsm_id, fsm_mode

    def _rpc_int(self, method: str, api_id: int) -> Optional[int]:
        """VERIFY: GetFsmId/GetFsmMode API ids 7001/7002 per unitree_sdk2 g1 loco_api.hpp."""
        fn = getattr(self.loco, method, None)
        try:
            if fn is not None:
                ret = fn()
                if isinstance(ret, tuple):
                    code, val = ret[0], ret[1]
                    return int(val) if code == 0 else None
                return int(ret)
            try:
                self.loco._RegistApi(api_id, 0)
            except Exception:
                pass
            code, data = self.loco._Call(api_id, "{}")
            if code != 0:
                return None
            obj = json.loads(data) if isinstance(data, str) else data
            return int(obj.get("data")) if isinstance(obj, dict) else int(obj)
        except Exception as exc:
            log.warning("%s failed: %s", method, exc)
            return None

    def set_velocity(self, vx, vy, vyaw, duration_s):
        # Explicit, short duration: never the "continuous" variant.
        if hasattr(self.loco, "SetVelocity"):
            self.loco.SetVelocity(float(vx), float(vy), float(vyaw), float(duration_s))
        else:
            self.loco.Move(float(vx), float(vy), float(vyaw))
