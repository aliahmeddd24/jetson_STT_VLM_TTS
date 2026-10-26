#!/usr/bin/env python3
"""Convert an avp_teleoperate episode JSON into a gesture clip, then validate it.

Example:
  python scripts/convert_avp_clip.py ~/episode_0003/data.json --name wave \
      --start 30 --end 210 --out gestures/clips/wave.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "action_server"))
from g1action.clips import Rules, clamp_to_limits, convert_avp_episode, save_clip, validate  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode")
    p.add_argument("--name", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source", choices=["actions", "states"], default="actions",
                   help="use commanded targets (actions) or measured positions (states)")
    p.add_argument("--fps", type=float, default=None, help="recording fps (default: from file, else 30)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--smooth", type=int, default=5, help="moving-average window in frames")
    p.add_argument("--blend", type=float, default=0.5, help="seconds to blend from/to neutral")
    p.add_argument("--urdf", default="third_party/g1_description/g1_29dof.urdf")
    p.add_argument("--clamp", action="store_true", help="clamp to URDF limits minus margin")
    a = p.parse_args()

    with open(a.episode, "r", encoding="utf-8") as fh:
        ep = json.load(fh)
    clip = convert_avp_episode(ep, a.name, source=a.source, fps=a.fps, start=a.start, end=a.end,
                               smooth=a.smooth, blend_s=a.blend)
    urdf = None
    if os.path.exists(a.urdf):
        from g1action.urdf import URDF
        urdf = URDF(a.urdf)
        if a.clamp:
            clip = clamp_to_limits(clip, urdf, Rules().limits_margin_rad)
    save_clip(clip, a.out)
    print(f"wrote {a.out}: {len(clip.frames)} frames @ {clip.fps} Hz = {clip.duration_s:.2f} s")
    errs = validate(clip, Rules(), 1.0, urdf)
    for e in errs:
        print("PROBLEM:", e)
    print("VALID" if not errs else "NOT VALID (fix, trim with --start/--end, or --clamp)")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
