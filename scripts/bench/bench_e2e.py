#!/usr/bin/env python3
"""Phase 4/6: end-to-end latency over recorded WAVs, using the configured backends.

Each WAV is replayed in real time as the microphone; speech goes to a paced null sink
(timing as if played). Gestures go to the action server (dry-run by default) unless --no-actions.
--loop-seconds N repeats the set for a soak test.
"""
import argparse
import glob
import os
import time

import _common  # noqa: F401
from g1voice import config
from g1voice.main import build, wait_for_services


class Args:
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/settings.yaml")
    p.add_argument("--manifest", default="gestures/manifest.yaml")
    p.add_argument("--dir", default="tests/recordings")
    p.add_argument("--metrics", default="logs/e2e.jsonl")
    p.add_argument("--no-actions", action="store_true")
    p.add_argument("--loop-seconds", type=float, default=0)
    a = p.parse_args()
    wavs = sorted(glob.glob(os.path.join(a.dir, "*.wav")))
    if not wavs:
        raise SystemExit("no WAVs in %s (record some with scripts/record_utterances.py)" % a.dir)
    t_stop = time.monotonic() + a.loop_seconds
    first = True
    while True:
        for w in wavs:
            cfg = config.load(a.config, {"audio_in": {"source": "wav", "wav_path": w, "wav_realtime": True},
                                         "audio_out": {"sink": "null"}, "metrics": {"path": a.metrics}})
            if first:
                wait_for_services(cfg, 600)
                first = False
            args = Args()
            args.no_actions, args.manifest = a.no_actions, a.manifest
            orch = build(cfg, args)
            orch.metrics.event(0, "file", name=os.path.basename(w))
            orch.run(max_turns=None)
            orch.actions.close()
            orch.metrics.close()
            print("done", w, "turns", orch.turns_done)
        if not a.loop_seconds or time.monotonic() > t_stop:
            break
    print("metrics in", a.metrics, "-> python3 scripts/bench/report.py logs")


if __name__ == "__main__":
    main()
