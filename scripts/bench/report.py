#!/usr/bin/env python3
"""Summarise logs/ (bench_*.json, *.jsonl turn metrics) into markdown with PASS/MISS.
Stdlib only (host Python 3.8). Targets are the owner's, not guarantees."""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import summary  # noqa: E402

TARGETS = {  # name: (comparison, value, unit)
    "vlm_ttft_image_p50": ("<=", 2.5, "s"), "vlm_decode_tok_s_p50": (">=", 15.0, "tok/s"),
    "stt_final_latency_p50": ("<=", 0.7, "s"), "tts_first_audio_p50": ("<=", 0.7, "s"),
    "tts_rtf_p50": ("<", 0.5, ""), "e2e_reflex_p50": ("<=", 0.8, "s"),
    "e2e_text_first_audio_p50": ("<=", 2.0, "s"), "e2e_vision_first_audio_p50": ("<=", 3.5, "s"),
}


def verdict(name, v):
    if v is None:
        return "NO DATA"
    op, t, _ = TARGETS[name]
    ok = {"<=": v <= t, ">=": v >= t, "<": v < t}[op]
    return "PASS" if ok else "MISS"


def turns_from_jsonl(paths):
    rows = []
    for p in paths:
        turns = defaultdict(dict)
        for line in open(p):
            e = json.loads(line)
            d = turns[(p, e["turn"])]
            d.setdefault(e["event"], e)          # first occurrence wins
            if e["event"] == "command_issued":
                d.setdefault("_cmds", []).append(e)
        rows.extend(turns.values())
    out = {"stt": [], "text": [], "vision": [], "reflex": [], "first_token": []}
    for d in rows:
        end = d.get("speech_end", {}).get("t")
        if end is None:
            continue
        if "stt_final" in d:
            out["stt"].append(d["stt_final"]["t"] - end)
        if "vlm_first_token" in d:
            out["first_token"].append(d["vlm_first_token"]["t"] - end)
        if "first_audio" in d:
            vis = d.get("vlm_request", {}).get("vision")
            out["vision" if vis else "text"].append(d["first_audio"]["t"] - end)
        for c in d.get("_cmds", []):
            if c.get("action") in ("gesture", "step") and c.get("origin") != "llm":
                out["reflex"].append(c["t"] - end)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logdir", nargs="?", default="logs")
    p.add_argument("--only", default="", help="substring filter on file names")
    a = p.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(a.logdir, "*")) if a.only in os.path.basename(f))
    vals = {}
    print("| metric | p50 | p90 | n | target | result | source |\n|---|---|---|---|---|---|---|")

    def row(name, s, src):
        v = s.get("p50")
        vals[name] = v
        op, t, unit = TARGETS[name]
        print("| %s | %s | %s | %s | %s %s %s | %s | %s |" % (name, v, s.get("p90"), s.get("n"), op, t, unit,
                                                           verdict(name, v), os.path.basename(src)))

    for f in files:
        if not f.endswith(".json"):
            continue
        d = json.load(open(f))
        k = d.get("kind")
        if k == "vlm":
            row("vlm_ttft_image_p50", d["ttft_image"], f)
            row("vlm_decode_tok_s_p50", d["decode_tok_s"], f)
        elif k == "stt":
            row("stt_final_latency_p50", d["latency"], f)
        elif k == "tts":
            row("tts_first_audio_p50", d["first_audio"], f)
            row("tts_rtf_p50", d["rtf"], f)
    jl = [f for f in files if f.endswith(".jsonl")]
    if jl:
        t = turns_from_jsonl(jl)
        src = "%d jsonl file(s)" % len(jl)
        row("e2e_reflex_p50", summary(t["reflex"]), src)
        row("e2e_text_first_audio_p50", summary(t["text"]), src)
        row("e2e_vision_first_audio_p50", summary(t["vision"]), src)
        print("\nstages (s, from end of speech): stt_final %s | vlm_first_token %s"
              % (summary(t["stt"]), summary(t["first_token"])))
        print("note: reflex values < 0 mean the command fired on a partial before speech ended.")


if __name__ == "__main__":
    main()
