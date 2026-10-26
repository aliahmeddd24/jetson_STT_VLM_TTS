#!/usr/bin/env python3
"""Operator CLI for the action server (stdlib only; runs on the host's Python 3.8).

  python3 scripts/g1_operator.py status
  python3 scripts/g1_operator.py walking on|off
  python3 scripts/g1_operator.py release-arms
  python3 scripts/g1_operator.py gesture wave        # sends heartbeats while it runs
  python3 scripts/g1_operator.py step left
  python3 scripts/g1_operator.py stop

Gestures/steps need a fresh heartbeat; this tool sends one every 100 ms for the
duration of the command (and the motion), standing in for the orchestrator.
"""
import argparse
import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8090"


def call(path, body=None, token=None, method="POST"):
    data = json.dumps(body or {}).encode() if method == "POST" else None
    req = urllib.request.Request(URL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    if token:
        req.add_header("X-Operator-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}")


def heartbeat_for(seconds):
    stop = threading.Event()

    def run():
        end = time.monotonic() + seconds
        while not stop.is_set() and time.monotonic() < end:
            call("/heartbeat")
            time.sleep(0.1)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return stop


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["status", "walking", "release-arms", "gesture", "step", "stop"])
    p.add_argument("arg", nargs="?")
    p.add_argument("--token-file", default="state/operator_token")
    p.add_argument("--hold", type=float, default=8.0, help="seconds of heartbeat for gesture/step")
    a = p.parse_args()

    if a.cmd == "status":
        print(json.dumps(call("/status", method="GET"), indent=2))
        return 0
    if a.cmd == "stop":
        print(call("/stop"))
        return 0
    if a.cmd in ("walking", "release-arms"):
        token = open(a.token_file).read().strip()
        if a.cmd == "walking":
            if a.arg not in ("on", "off"):
                p.error("walking on|off")
            print(call("/operator/walking", {"enabled": a.arg == "on"}, token))
        else:
            print("This holds the MEASURED arm pose at full weight and ramps to zero.")
            if input("Robot standing, spotter present, e-stop in hand? type yes: ").strip() != "yes":
                return 1
            print(call("/operator/release-arms", {}, token))
        return 0
    stop = heartbeat_for(a.hold)
    time.sleep(0.15)
    try:
        if a.cmd == "gesture":
            print(call("/gesture", {"name": a.arg}))
        else:
            print(call("/step", {"dir": a.arg}))
        time.sleep(a.hold)
    except KeyboardInterrupt:
        print(call("/stop"))
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
