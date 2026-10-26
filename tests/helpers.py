import json
import os
import socket
import tempfile
import threading

import numpy as np

import _paths  # noqa: F401


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start_mock_vlm(reply="Hello there! I am fine. [gesture:wave] Nice to meet you.", delay=0.0, first=0.0):
    import openai_server as vs
    port = free_port()
    srv = vs.make_server(vs.MockBackend(reply, delay, first), "127.0.0.1", port, "mock")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d/v1/chat/completions" % port


def write_valid_clip_and_manifest(tmp, enabled=True, llm_allowed=True):
    from g1action import LEFT_ARM, RIGHT_ARM
    from g1action.clips import Clip, save_clip
    n = 51                                  # 1.0 s at 50 Hz
    t = np.linspace(0, 1, n)
    F = np.zeros((n, 14))
    F[:, 7] = -0.4 * np.sin(np.pi * t) ** 2
    os.makedirs(os.path.join(tmp, "clips"), exist_ok=True)
    save_clip(Clip("wave", 50.0, LEFT_ARM + RIGHT_ARM, F), os.path.join(tmp, "clips", "wave.json"))
    manifest = {"urdf": os.path.join(_paths.FIX, "toy_g1.urdf"),
                "gestures": {"wave": {"enabled": enabled, "llm_allowed": llm_allowed,
                                      "file": "clips/wave.json", "speed_scale": 1.0}}}
    path = os.path.join(tmp, "manifest.yaml")
    import yaml
    with open(path, "w") as fh:
        yaml.safe_dump(manifest, fh)
    return path


def action_cfg(tmp, manifest, allowed=(801,)):
    return {"bind_host": "127.0.0.1", "bind_port": 0, "heartbeat_timeout_s": 0.5,
            "state_dir": os.path.join(tmp, "state"),
            "operator_token_file": os.path.join(tmp, "state", "operator_token"),
            "mode_check": {"allowed_fsm_ids": list(allowed), "allowed_fsm_modes": []},
            "arm": {"control_hz": 100, "kp": 60.0, "kd": 1.5, "weight_joint_index": 29,
                    "ramp_in_s": 0.1, "approach_s": 0.1, "ramp_out_s": 0.2, "max_clip_s": 4.0,
                    "max_speed_scale": 1.0, "cooldown_s": 0.2, "start_pose_tolerance_rad": 0.35,
                    "hold_joints": [12, 13, 14]},
            "loco": {"max_vy": 0.2, "max_duration_s": 2.0, "step_vy": 0.15, "step_duration_s": 0.5,
                     "resend_period_s": 0.1},
            "manifest": manifest}
