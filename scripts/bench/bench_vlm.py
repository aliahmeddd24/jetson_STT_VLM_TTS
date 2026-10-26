#!/usr/bin/env python3
"""Phase 2: VLM first-token latency and decode speed, text-only and with one image.

Run tegrastats alongside (scripts/tegrastats_log.sh) for memory. Targets (non-blocking):
first token <= 2.5 s with one image <= 448 px, >= 15 tok/s, <= 5 GB.
"""
import argparse
import base64
import datetime
import json
import time

import _common  # noqa: F401
import numpy as np
import requests


def test_image(px):
    import cv2
    img = np.full((px, px, 3), 235, np.uint8)
    cv2.rectangle(img, (40, 60), (200, 220), (30, 30, 200), -1)
    cv2.circle(img, (320, 300), 80, (40, 160, 40), -1)
    cv2.putText(img, "G1", (260, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5)
    return cv2.imencode(".jpg", img)[1].tobytes()


def load_image(path, px):
    if not path:
        return test_image(px)
    from g1voice.camera import downscale_jpeg
    return downscale_jpeg(open(path, "rb").read(), px)


def run(url, model, text, img, max_tokens):
    content = text if img is None else [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}},
        {"type": "text", "text": text}]
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0.0,
            "messages": [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": content}]}
    t0 = time.monotonic()
    t_first, n, server = None, 0, None
    with requests.post(url, json=body, stream=True, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                break
            obj = json.loads(d)
            server = obj.get("x_timing", server)
            for ch in obj.get("choices", []):
                if (ch.get("delta") or {}).get("content"):
                    n += 1
                    t_first = t_first or time.monotonic()
    t1 = time.monotonic()
    return {"ttft_s": (t_first or t1) - t0, "deltas": n, "total_s": t1 - t0,
            "client_tok_s": (n - 1) / (t1 - t_first) if t_first and n > 1 and t1 > t_first else None,
            "server": server}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    p.add_argument("--model", default="vila1.5-3b")
    p.add_argument("--image", default="")
    p.add_argument("--px", type=int, default=448)
    p.add_argument("-n", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--out", default="logs/bench_vlm.json")
    a = p.parse_args()
    img = load_image(a.image, a.px)
    run(a.url, a.model, "Say hello.", None, 8)          # warm-up
    res = {"text": [], "image": []}
    for i in range(a.n):
        res["text"].append(run(a.url, a.model, "Tell me a fun fact about robots in two sentences.", None, a.max_tokens))
        res["image"].append(run(a.url, a.model, "Describe what you see in one or two sentences.", img, a.max_tokens))
        print(i, "text ttft %.2f s | image ttft %.2f s, %s tok/s" % (
            res["text"][-1]["ttft_s"], res["image"][-1]["ttft_s"],
            (res["image"][-1]["server"] or {}).get("decode_tok_s")))
    out = {"kind": "vlm", "date": datetime.datetime.now().isoformat(timespec="seconds"), "args": vars(a),
           "ttft_text": _common.summary([r["ttft_s"] for r in res["text"]]),
           "ttft_image": _common.summary([r["ttft_s"] for r in res["image"]]),
           "decode_tok_s": _common.summary([(r["server"] or {}).get("decode_tok_s") or r["client_tok_s"]
                                            for r in res["text"] + res["image"]]),
           "runs": res}
    _common.write_json(a.out, out)
    print(json.dumps({k: out[k] for k in ("ttft_text", "ttft_image", "decode_tok_s")}, indent=2))


if __name__ == "__main__":
    main()
