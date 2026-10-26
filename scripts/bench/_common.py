import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "..", "..", "services", "orchestrator"),
          os.path.join(HERE, "..", "..", "services", "action_server"), "/app/services/orchestrator"):
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summary(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50": round(pct(xs, 50), 3), "p90": round(pct(xs, 90), 3),
            "mean": round(statistics.mean(xs), 3), "max": round(max(xs), 3)}


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    print("wrote", path)
