#!/usr/bin/env python3
"""Play a short tone (or a WAV) through the G1 speaker via AudioClient.PlayStream, and
cycle the head LED. Runs inside the app image (needs unitree_sdk2py):

  docker compose run --rm --no-deps orchestrator python /app/scripts/speaker_test.py --iface eth0
"""
import argparse
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, "/app/services/orchestrator")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "orchestrator"))
from g1voice.audio import G1SpeakerSink, float_to_pcm  # noqa: E402
from g1voice.robot import G1Audio  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iface", default=os.environ.get("G1V__G1__NETWORK_INTERFACE", "eth0"))
    p.add_argument("--wav", default="", help="16 kHz mono 16-bit WAV (default: 1.5 s tone)")
    p.add_argument("--volume", type=int, default=None)
    a = p.parse_args()
    robot = G1Audio(0, a.iface)
    print("volume before:", robot.get_volume())
    if a.volume is not None:
        robot.set_volume(a.volume)
    for rgb in ([0, 80, 255], [255, 160, 0], [0, 255, 80], [0, 0, 0]):
        robot.led(rgb)
        time.sleep(0.4)
    if a.wav:
        with wave.open(a.wav) as w:
            pcm = w.readframes(w.getnframes())
    else:
        t = np.arange(int(16000 * 1.5)) / 16000.0
        pcm = float_to_pcm((0.2 * np.sin(2 * np.pi * 440 * t) * np.minimum(1, (1.5 - t) * 10)).astype(np.float32))
    sink = G1SpeakerSink(robot, "g1voice_test", chunk_ms=100, lead_ms=200)
    t0 = time.monotonic()
    sink.begin("test-%d" % int(time.time()))
    sink.write(pcm)
    sink.end()
    print("played %.2f s of audio in %.2f s wall time" % (len(pcm) / 32000.0, time.monotonic() - t0))
    print("Did it sound right (pitch/speed)? If too fast/slow, the 16 kHz assumption is wrong.")


if __name__ == "__main__":
    main()
