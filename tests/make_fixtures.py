"""Regenerates tests/fixtures/*. Synthetic data only: no real voices, no real robot clips."""
import json
import os
import wave

import numpy as np

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
AXES = {"shoulder_pitch": "0 1 0", "shoulder_roll": "1 0 0", "shoulder_yaw": "0 0 1", "elbow": "0 1 0",
        "wrist_roll": "1 0 0", "wrist_pitch": "0 1 0", "wrist_yaw": "0 0 1"}
OFFS = {"shoulder_pitch": "0 {y} 0.25", "shoulder_roll": "0 0 0", "shoulder_yaw": "0 0 -0.05",
        "elbow": "0 0 -0.15", "wrist_roll": "0 0 -0.08", "wrist_pitch": "0 0 -0.04", "wrist_yaw": "0 0 -0.03"}


def speechlike(seconds, sr=16000, level=0.15, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)            # 4 Hz syllable-ish envelope
    x = env * (0.6 * np.sin(2 * np.pi * 180 * t) + 0.4 * rng.standard_normal(t.size) * 0.5)
    return (level * x / np.max(np.abs(x))).astype(np.float32)


def write_wav(path, x, sr=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def utterance_wav(path, segments=((0.5, 0), (1.2, 1), (1.0, 0))):
    parts = [speechlike(d, seed=i) if s else np.zeros(int(d * 16000), np.float32)
             for i, (d, s) in enumerate(segments)]
    write_wav(path, np.concatenate(parts))


def toy_urdf():
    L = ['<robot name="toy_g1">', '<link name="pelvis"/>']
    prev = "pelvis"
    for j, ax, org in (("waist_yaw", "0 0 1", "0 0 0.1"), ("waist_roll", "1 0 0", "0 0 0"),
                       ("waist_pitch", "0 1 0", "0 0 0")):
        child = "torso_link" if j == "waist_pitch" else j + "_link"
        L += [f'<link name="{child}"/>',
              f'<joint name="{j}_joint" type="revolute"><parent link="{prev}"/><child link="{child}"/>'
              f'<origin xyz="{org}" rpy="0 0 0"/><axis xyz="{ax}"/>'
              f'<limit lower="-0.5" upper="0.5" effort="50" velocity="3"/></joint>']
        prev = child
    for side, y in (("left", 0.15), ("right", -0.15)):
        p = "torso_link"
        for n in ARM:
            child = f"{side}_{n}_link"
            lo, hi = (-1.0, 2.0) if n == "elbow" else (-2.5, 2.5)
            L += [f'<link name="{child}"/>',
                  f'<joint name="{side}_{n}_joint" type="revolute"><parent link="{p}"/><child link="{child}"/>'
                  f'<origin xyz="{OFFS[n].format(y=y)}" rpy="0 0 0"/><axis xyz="{AXES[n]}"/>'
                  f'<limit lower="{lo}" upper="{hi}" effort="20" velocity="3"/></joint>']
            p = child
    L.append("</robot>")
    return "\n".join(L)


def avp_episode(n=90, fps=30.0):
    """Shape of an avp_teleoperate episode (assumed format): a right-arm wave."""
    t = np.arange(n) / fps
    data = []
    for i, ti in enumerate(t):
        left = [0.0, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0]
        lift = min(1.0, ti / 0.8) if ti < 2.2 else max(0.0, (3.0 - ti) / 0.8)
        right = [-1.2 * lift, -0.3 * lift, 0.0, 0.9 * lift + 0.3 * (1 - lift),
                 0.25 * lift * np.sin(2 * np.pi * 1.2 * ti), 0.0, 0.0]
        right[1] = right[1] - 0.2 * (1 - lift)
        st = {"left_arm": {"qpos": left}, "right_arm": {"qpos": right}}
        data.append({"idx": i, "states": st, "actions": st})
    return {"info": {"version": "synthetic", "image": {"fps": fps}}, "text": {"goal": "wave"}, "data": data}


if __name__ == "__main__":
    os.makedirs(HERE, exist_ok=True)
    utterance_wav(os.path.join(HERE, "hello.wav"))
    with open(os.path.join(HERE, "toy_g1.urdf"), "w") as fh:
        fh.write(toy_urdf())
    with open(os.path.join(HERE, "avp_episode.json"), "w") as fh:
        json.dump(avp_episode(), fh)
    print("fixtures written to", HERE)
