#!/usr/bin/env python3
"""Phase 3: TTS time-to-first-audio for 3-8 word chunks and real-time factor.

Targets (non-blocking): first audio <= 0.7 s, RTF < 0.5.
"""
import argparse
import datetime
import time

import _common  # noqa: F401
from g1voice import config
from g1voice.tts import make_tts

SENTENCES = ["Hello there.", "Nice to meet you today.", "I can see a red box.",
             "That is a very good question.", "Let me think about that for a moment.",
             "The weather looks lovely outside today, doesn't it?",
             "My name is G1 and I am a humanoid robot.", "Sure, I can wave to you."]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/settings.yaml")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", default="logs/bench_tts.json")
    a = p.parse_args()
    cfg = config.load(a.config)
    tts = make_tts(cfg)
    list(tts.synth("Warm up."))
    rows = []
    for _ in range(a.rounds):
        for s in SENTENCES:
            t0 = time.monotonic()
            first, nbytes = None, 0
            for pcm in tts.synth(s):
                first = first or time.monotonic()
                nbytes += len(pcm)
            t1 = time.monotonic()
            audio_s = nbytes / 32000.0
            rows.append({"words": len(s.split()), "first_audio_s": round((first or t1) - t0, 3),
                         "synth_s": round(t1 - t0, 3), "audio_s": round(audio_s, 3),
                         "rtf": round((t1 - t0) / audio_s, 3) if audio_s else None})
            print("%d words: first %.3f s, RTF %s" % (rows[-1]["words"], rows[-1]["first_audio_s"], rows[-1]["rtf"]))
    out = {"kind": "tts", "backend": cfg.tts.backend, "date": datetime.datetime.now().isoformat(timespec="seconds"),
           "first_audio": _common.summary([r["first_audio_s"] for r in rows]),
           "rtf": _common.summary([r["rtf"] for r in rows]), "rows": rows}
    _common.write_json(a.out, out)


if __name__ == "__main__":
    main()
