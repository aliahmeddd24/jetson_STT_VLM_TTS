import json
import os
import unittest

import numpy as np

import _paths  # noqa: F401
from g1action import RIGHT_ARM, LEFT_ARM
from g1action.clips import Clip, Rules, convert_avp_episode, resample, validate
from g1action.urdf import URDF

URDF_PATH = os.path.join(_paths.FIX, "toy_g1.urdf")
J = LEFT_ARM + RIGHT_ARM


def wave_clip(fps=50.0, seconds=2.0, amp=0.3):
    n = int(fps * seconds) + 1
    t = np.linspace(0, 1, n)
    F = np.zeros((n, 14))
    F[:, 3] = 0.3
    F[:, 10] = 0.3
    bump = np.sin(np.pi * t) ** 2
    F[:, 7] = -1.0 * bump
    F[:, 11] = amp * np.sin(2 * np.pi * 2 * t) * bump
    return Clip("wave", fps, J, F)


class TestURDF(unittest.TestCase):
    def setUp(self):
        self.u = URDF(URDF_PATH)

    def test_limits(self):
        self.assertEqual(self.u.limits("left_elbow_joint"), (-1.0, 2.0))

    def test_fk_arm_down(self):
        p = self.u.position_in("left_wrist_yaw_link", "torso_link", {})
        np.testing.assert_allclose(p, [0, 0.15, 0.25 - 0.05 - 0.15 - 0.08 - 0.04 - 0.03], atol=1e-9)

    def test_fk_shoulder_pitch_forward(self):
        p = self.u.position_in("left_wrist_yaw_link", "torso_link", {"left_shoulder_pitch_joint": -np.pi / 2})
        self.assertGreater(p[0], 0.3)        # arm raised forward -> +x


class TestValidate(unittest.TestCase):
    def setUp(self):
        self.u = URDF(URDF_PATH)
        self.r = Rules()

    def test_good_clip_passes(self):
        self.assertEqual(validate(wave_clip(), self.r, 1.0, self.u), [])

    def test_no_urdf_is_reported(self):
        errs = validate(wave_clip(), self.r, 1.0, None)
        self.assertTrue(any("no URDF" in e for e in errs))

    def test_start_end_mismatch(self):
        c = wave_clip()
        c.frames[-1, 0] += 0.2
        self.assertTrue(any("neutral" in e for e in validate(c, self.r, 1.0, self.u)))

    def test_too_long(self):
        self.assertTrue(any("duration" in e for e in validate(wave_clip(seconds=3.5), self.r, 0.8, self.u)))

    def test_too_fast(self):
        self.assertTrue(any("joint speed" in e for e in validate(wave_clip(seconds=0.4, amp=2.0), self.r, 1.0, self.u)))

    def test_speed_scale_above_one_rejected(self):
        self.assertTrue(any("speed_scale" in e for e in validate(wave_clip(), self.r, 1.5, self.u)))

    def test_joint_limit(self):
        c = wave_clip()
        c.frames[30:60, 3] = 2.1                      # elbow limit 2.0, margin 0.05
        self.assertTrue(any("left_elbow_joint" in e for e in validate(c, self.r, 1.0, self.u)))

    def test_bubble(self):
        c = wave_clip()
        bump = np.sin(np.linspace(0, np.pi, len(c.frames)))
        c.frames[:, 0] = 1.6 * bump                   # left arm swings BACK (behind the torso)
        self.assertTrue(any("bubble" in e for e in validate(c, self.r, 1.0, self.u)))

    def test_non_arm_joint_rejected(self):
        c = wave_clip()
        c.joints = [2] + c.joints[1:]
        self.assertTrue(any("only arm joints" in e for e in validate(c, self.r, 1.0, self.u)))

    def test_resample_speed(self):
        c = wave_clip(fps=50, seconds=2.0)
        self.assertEqual(len(resample(c, 50, 1.0)), 101)
        self.assertEqual(len(resample(c, 50, 0.5)), 201)


class TestConvert(unittest.TestCase):
    def test_avp_episode_to_valid_clip(self):
        ep = json.load(open(os.path.join(_paths.FIX, "avp_episode.json")))
        c = convert_avp_episode(ep, "wave", out_fps=50)
        self.assertEqual(c.joints, J)
        np.testing.assert_allclose(c.frames[0], c.frames[-1], atol=1e-9)
        self.assertEqual(validate(c, Rules(), 1.0, URDF(URDF_PATH)), [])

    def test_missing_arm_key(self):
        with self.assertRaises(KeyError):
            convert_avp_episode({"data": [{"actions": {}}, {"actions": {}}]}, "x")


if __name__ == "__main__":
    unittest.main()
