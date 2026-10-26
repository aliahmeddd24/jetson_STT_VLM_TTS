"""Safety behaviour of the action server against the dry-run backend."""
import json
import os
import re
import tempfile
import threading
import time
import unittest
import urllib.request

import numpy as np

import _paths  # noqa: F401
from g1action import ARMS
from g1action.backend import DryRunBackend
from g1action.server import ActionServer, serve
from helpers import action_cfg, write_valid_clip_and_manifest


class Base(unittest.TestCase):
    allowed = (801,)
    fsm = 801

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.manifest = write_valid_clip_and_manifest(self.tmp)
        self.b = DryRunBackend(fsm_id=self.fsm)
        self.srv = ActionServer(action_cfg(self.tmp, self.manifest, self.allowed), self.b)

    def tearDown(self):
        self.srv.close()

    def beat(self, seconds):
        stop = threading.Event()

        def run():
            end = time.monotonic() + seconds
            while time.monotonic() < end and not stop.is_set():
                self.srv.heartbeat()
                time.sleep(0.05)
        threading.Thread(target=run, daemon=True).start()
        time.sleep(0.06)
        return stop

    def arm_log(self):
        return [e for e in self.b.log if e[0] == "arm"]

    def vel_log(self):
        return [e for e in self.b.log if e[0] == "vel"]


class TestRefusals(Base):
    def test_no_heartbeat(self):
        r = self.srv.gesture("wave")
        self.assertFalse(r["ok"])
        self.assertIn("heartbeat", r["reason"])
        self.assertEqual(self.arm_log(), [])

    def test_unknown_gesture(self):
        self.beat(1)
        self.assertFalse(self.srv.gesture("backflip")["ok"])

    def test_walking_disabled_by_default(self):
        self.beat(1)
        r = self.srv.step("left")
        self.assertIn("walking is disabled", r["reason"])
        self.assertEqual(self.vel_log(), [])

    def test_bad_direction(self):
        self.srv.set_walking(True)
        self.beat(1)
        self.assertFalse(self.srv.step("forward")["ok"])


class TestEmptyFsmAllowlist(Base):
    allowed = ()

    def test_refuses_until_configured(self):
        self.beat(1)
        r = self.srv.gesture("wave")
        self.assertIn("allowed_fsm_ids is empty", r["reason"])


class TestWrongMode(Base):
    fsm = 1

    def test_refuses_in_wrong_fsm(self):
        self.beat(1)
        self.assertIn("not in allowed", self.srv.gesture("wave")["reason"])


class TestGesture(Base):
    def test_weight_ramps_from_measured_pose(self):
        self.b.q[ARMS] = 0.05                       # measured pose slightly off the clip start
        self.beat(3)
        self.assertTrue(self.srv.gesture("wave")["ok"])
        marker = os.path.join(self.tmp, "state", "arm_active")
        time.sleep(0.05)
        self.assertTrue(os.path.exists(marker))     # crash-detection marker while weight > 0
        self.srv.player.wait(5)
        w = [e[2] for e in self.arm_log()]
        self.assertEqual(w[0], 0.0)                  # starts at zero weight
        self.assertEqual(max(w), 1.0)
        self.assertEqual(w[-1], 0.0)                 # ends at zero weight
        steps = np.diff(w)
        self.assertLessEqual(np.max(np.abs(steps)), 0.2)   # never flips abruptly
        self.assertFalse(os.path.exists(marker))
        self.assertEqual(self.srv.player.phase, "idle")

    def test_one_at_a_time_and_cooldown(self):
        self.beat(3)
        self.assertTrue(self.srv.gesture("wave")["ok"])
        self.assertIn("another motion", self.srv.gesture("wave")["reason"])
        self.srv.player.wait(5)
        self.assertEqual(self.srv.gesture("wave")["reason"], "cooldown")

    def test_stop_ramps_out(self):
        self.beat(3)
        self.srv.gesture("wave")
        time.sleep(0.15)
        t0 = time.monotonic()
        self.srv.stop()
        self.srv.player.wait(5)
        self.assertLess(time.monotonic() - t0, 2.0)
        self.assertEqual(self.arm_log()[-1][2], 0.0)

    def test_watchdog_stops_on_lost_heartbeat(self):
        stop = self.beat(3)
        self.srv.gesture("wave")
        stop.set()                                   # orchestrator "dies"
        self.srv.player.wait(5)
        self.assertEqual(self.srv.last_event.get("reason") in ("watchdog", "finished", "stopped"), True)
        self.assertEqual(self.arm_log()[-1][2], 0.0)

    def test_far_from_start_pose_refused(self):
        self.b.q[ARMS] = 1.0
        self.beat(1)
        self.assertIn("from the clip start", self.srv.gesture("wave")["reason"])


class TestWatchdogLongClip(Base):
    def test_watchdog_cuts_motion_short(self):
        self.srv.player.ramp_in_s = 1.5                # make the motion long enough to observe
        stop = self.beat(5)
        self.assertTrue(self.srv.gesture("wave")["ok"])
        time.sleep(0.2)
        stop.set()
        t0 = time.monotonic()
        self.srv.player.wait(5)
        self.assertLess(time.monotonic() - t0, 1.2)    # 0.5 s timeout + ramp-out
        self.assertEqual(self.srv.last_event.get("reason"), "stopped")


class TestSteps(Base):
    def test_step_left_and_right(self):
        self.srv.set_walking(True)
        self.beat(3)
        r = self.srv.step("left")
        self.assertTrue(r["ok"])
        self.srv.stepper.wait(3)
        v = self.vel_log()
        self.assertTrue(all(e[2] == 0.0 and e[4] == 0.0 for e in v))   # lateral only
        self.assertTrue(all(abs(e[3]) <= 0.2 for e in v))
        self.assertTrue(all(e[5] <= 1.05 for e in v))                   # never continuous
        self.assertGreater(v[0][3], 0)                                   # left = +y
        self.assertEqual(v[-1][3], 0.0)                                  # ends with zero velocity
        time.sleep(0.25)
        self.b.log.clear()
        self.srv.step("right")
        self.srv.stepper.wait(3)
        self.assertLess(self.vel_log()[0][3], 0)

    def test_disable_walking_stops(self):
        self.srv.set_walking(True)
        self.beat(3)
        self.srv.step("left")
        self.srv.set_walking(False)
        self.srv.stepper.wait(3)
        self.assertEqual(self.vel_log()[-1][3], 0.0)


class TestCrashMarker(unittest.TestCase):
    def test_marker_blocks_until_release(self):
        tmp = tempfile.mkdtemp()
        man = write_valid_clip_and_manifest(tmp)
        os.makedirs(os.path.join(tmp, "state"))
        open(os.path.join(tmp, "state", "arm_active"), "w").close()
        b = DryRunBackend(fsm_id=801)
        srv = ActionServer(action_cfg(tmp, man), b)
        srv.heartbeat()
        self.assertIn("operator release", srv.gesture("wave")["reason"])
        self.assertTrue(srv.release_arms()["ok"])
        w = [e[2] for e in b.log if e[0] == "arm"]
        self.assertEqual(w[0], 1.0)
        self.assertEqual(w[-1], 0.0)
        srv.heartbeat()
        self.assertTrue(srv.gesture("wave")["ok"])
        srv.close()


class TestHTTP(Base):
    def setUp(self):
        super().setUp()
        self.httpd = serve(self.srv, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        super().tearDown()

    def post(self, path, body=None, token=None):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (self.port, path),
                                     json.dumps(body or {}).encode(), {"Content-Type": "application/json"})
        if token:
            req.add_header("X-Operator-Token", token)
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_operator_endpoints_need_token(self):
        self.assertEqual(self.post("/operator/walking", {"enabled": True})[0], 403)
        self.assertFalse(self.srv.walking_enabled)
        self.assertEqual(self.post("/operator/walking", {"enabled": True}, "wrong")[0], 403)
        code, body = self.post("/operator/walking", {"enabled": True}, self.srv.token)
        self.assertEqual((code, body["walking_enabled"]), (200, True))

    def test_api_flow(self):
        self.post("/heartbeat")
        code, body = self.post("/gesture", {"name": "wave"})
        self.assertEqual(code, 200)
        self.assertTrue(body["dry_run"])
        self.assertEqual(self.post("/stop")[0], 200)
        with urllib.request.urlopen("http://127.0.0.1:%d/status" % self.port, timeout=3) as r:
            st = json.loads(r.read())
        self.assertEqual(st["gestures"], ["wave"])
        self.assertFalse(st["live"])

    def test_token_file_private(self):
        mode = os.stat(os.path.join(self.tmp, "state", "operator_token")).st_mode & 0o777
        self.assertEqual(mode, 0o600)

    def test_refuses_non_loopback_bind(self):
        with self.assertRaises(ValueError):
            serve(self.srv, "0.0.0.0", 0)


class TestManifestStrictness(unittest.TestCase):
    def test_live_requires_urdf(self):
        from g1action.server import load_manifest
        tmp = tempfile.mkdtemp()
        man = write_valid_clip_and_manifest(tmp)
        import yaml
        d = yaml.safe_load(open(man))
        d["urdf"] = "/nonexistent.urdf"
        yaml.safe_dump(d, open(man, "w"))
        ok, problems, _ = load_manifest(man, strict=True)
        self.assertEqual(ok, {})
        self.assertIn("no URDF", problems["wave"])
        ok, problems, _ = load_manifest(man, strict=False)
        self.assertIn("wave", ok)

    def test_disabled_gesture_not_loaded(self):
        from g1action.server import load_manifest
        tmp = tempfile.mkdtemp()
        man = write_valid_clip_and_manifest(tmp, enabled=False)
        self.assertEqual(load_manifest(man)[0], {})

    def test_shipped_manifest_has_nothing_enabled(self):
        import yaml
        m = yaml.safe_load(open(os.path.join(_paths.ROOT, "gestures", "manifest.yaml")))
        self.assertFalse(any(g.get("enabled") for g in m["gestures"].values()))


class TestStaticSafety(unittest.TestCase):
    """The action server package must never contain damping/zero-torque/FSM-change calls,
    continuous moves, or non-loopback defaults."""

    def test_forbidden_calls_absent(self):
        pkg = os.path.join(_paths.ROOT, "services", "action_server", "g1action")
        bad = re.compile(r"\.(Damp|ZeroTorque|SetFsmId|StandUp|Squat|Sit|Lie2StandUp|HighStand|LowStand)\s*\("
                         r"|continous_move\s*=\s*True|continuous_move\s*=\s*True")
        for fn in os.listdir(pkg):
            if fn.endswith(".py"):
                src = open(os.path.join(pkg, fn)).read()
                self.assertIsNone(bad.search(src), "%s contains a forbidden call" % fn)

    def test_shipped_config_is_safe(self):
        import yaml
        c = yaml.safe_load(open(os.path.join(_paths.ROOT, "config", "action_server.yaml")))
        self.assertTrue(c["dry_run"])
        self.assertEqual(c["bind_host"], "127.0.0.1")
        self.assertEqual(c["mode_check"]["allowed_fsm_ids"], [])
        self.assertLessEqual(c["loco"]["max_vy"], 0.2)
        self.assertLessEqual(c["loco"]["max_duration_s"], 2.0)
        self.assertLessEqual(c["arm"]["ramp_out_s"], 2.0)
        self.assertLessEqual(c["arm"]["max_clip_s"], 4.0)
        self.assertLessEqual(c["arm"]["max_speed_scale"], 1.0)

    def test_live_needs_confirmation_phrase(self):
        from g1action import main as m
        os.environ.pop("G1_LIVE_CONFIRM", None)
        rc = m.main(["--config", os.path.join(_paths.ROOT, "config", "action_server.yaml"), "--live"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
