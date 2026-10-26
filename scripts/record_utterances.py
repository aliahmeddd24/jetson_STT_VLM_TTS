#!/usr/bin/env python3
"""Record prompted test utterances from the G1 mic into tests/recordings/ (WAV + .txt reference).

Privacy: these are voice recordings of real people. Record only consenting team members,
keep them on the robot, and delete them when testing is done (see docs/RUNBOOK.md).

  python3 scripts/record_utterances.py --iface eth0 --prompts scripts/bench/prompts.txt
"""
import argparse
import os
import socket
import struct
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(__file__))
from capture_g1_mic import iface_ip  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iface", default="eth0")
    p.add_argument("--prompts", default="scripts/bench/prompts.txt")
    p.add_argument("--outdir", default="tests/recordings")
    p.add_argument("--seconds", type=float, default=5.0)
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    prompts = [l.strip() for l in open(a.prompts) if l.strip() and not l.startswith("#")]
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", 5555))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                 socket.inet_aton("239.168.123.161") + socket.inet_aton(iface_ip(a.iface)))
    s.settimeout(0.2)
    for i, text in enumerate(prompts):
        input("\n[%d/%d] press Enter, wait 0.5 s, then say:\n    %s\n" % (i + 1, len(prompts), text))
        data = bytearray()
        end = time.monotonic() + a.seconds
        while time.monotonic() < end:
            try:
                data.extend(s.recvfrom(65536)[0])
            except socket.timeout:
                pass
        base = os.path.join(a.outdir, "utt_%02d" % (i + 1))
        with wave.open(base + ".wav", "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(bytes(data[: len(data) // 2 * 2]))
        with open(base + ".txt", "w") as fh:
            fh.write(text + "\n")
        print("saved", base + ".wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
