#!/usr/bin/env python3
"""Validate one clip (or all enabled gestures) with the rules in gestures/manifest.yaml."""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "action_server"))
from g1action.clips import Rules, load_clip, validate  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("clip", nargs="?", help="clip JSON; omit to check every gesture in the manifest")
    p.add_argument("--manifest", default="gestures/manifest.yaml")
    p.add_argument("--speed", type=float, default=1.0)
    a = p.parse_args()
    with open(a.manifest) as fh:
        m = yaml.safe_load(fh) or {}
    base = os.path.dirname(os.path.abspath(a.manifest))
    rules = Rules(**(m.get("rules") or {}))
    urdf = None
    if m.get("urdf"):
        up = os.path.join(base, m["urdf"])
        if os.path.exists(up):
            from g1action.urdf import URDF
            urdf = URDF(up)
        else:
            print(f"WARNING: URDF {up} not found: limits and bubble NOT checked")
    items = [(a.clip, a.speed)] if a.clip else [
        (os.path.join(base, g["file"]), float(g.get("speed_scale", 1.0)))
        for g in (m.get("gestures") or {}).values() if g]
    bad = 0
    for path, speed in items:
        if not os.path.exists(path):
            print(f"MISSING {path}")
            bad += 1
            continue
        errs = validate(load_clip(path), rules, speed, urdf)
        print(("VALID   " if not errs else "INVALID ") + path)
        for e in errs:
            print("   -", e)
        bad += bool(errs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
