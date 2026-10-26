#!/usr/bin/env python3
"""Phase 3: STT latency (end of audio -> final transcript) and a rough word error rate.

Streams each WAV in --dir at real time through the configured STT backend.
Target (non-blocking): <= 0.7 s for a ~3 s utterance. Transcripts are not saved unless --save-text.
"""
import argparse
import datetime
import glob
import os
import time
import wave

import _common  # noqa: F401
from g1voice import config
from g1voice.stt import make_stt


def wer(ref, hyp):
    norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c == " ").split()
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/settings.yaml")
    p.add_argument("--dir", default="tests/recordings")
    p.add_argument("--out", default="logs/bench_stt.json")
    p.add_argument("--save-text", action="store_true")
    a = p.parse_args()
    cfg = config.load(a.config)
    stt = make_stt(cfg)
    rows = []
    for path in sorted(glob.glob(os.path.join(a.dir, "*.wav"))):
        with wave.open(path) as w:
            pcm = w.readframes(w.getnframes())
        dur = len(pcm) / 32000.0
        s = stt.open_stream()
        step = 3200
        t0 = time.monotonic()
        for i in range(0, len(pcm), step):
            s.feed(pcm[i:i + step])
            delay = t0 + (i + step) / 32000.0 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        t_end = time.monotonic()
        text = s.finish()
        lat = time.monotonic() - t_end
        ref_path = path[:-4] + ".txt"
        e = wer(open(ref_path).read(), text) if os.path.exists(ref_path) else None
        rows.append({"file": os.path.basename(path), "audio_s": round(dur, 2), "final_latency_s": round(lat, 3),
                     "wer": None if e is None else round(e, 3), **({"text": text} if a.save_text else {})})
        print("%-16s %.1f s audio -> final in %.3f s  WER %s" % (rows[-1]["file"], dur, lat, rows[-1]["wer"]))
    out = {"kind": "stt", "backend": cfg.stt.backend, "date": datetime.datetime.now().isoformat(timespec="seconds"),
           "latency": _common.summary([r["final_latency_s"] for r in rows]),
           "wer": _common.summary([r["wer"] for r in rows]), "rows": rows}
    _common.write_json(a.out, out)


if __name__ == "__main__":
    main()
