#!/usr/bin/env python3
"""Record the G1 microphone multicast to a WAV and report what the stream looks like.

Verifies the format assumption (16 kHz mono int16) by measuring the byte rate.
Stdlib only: runs on the Jetson's system Python 3.8 without installing anything.

  python3 scripts/capture_g1_mic.py --iface eth0 --seconds 5 --out logs/g1_mic.wav
"""
import argparse
import array
import fcntl
import math
import socket
import struct
import sys
import time
import wave


def iface_ip(name):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", name[:15].encode()))[20:24])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iface", default="eth0")
    p.add_argument("--group", default="239.168.123.161")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--out", default="logs/g1_mic.wav")
    p.add_argument("--rate", type=int, default=16000, help="sample rate written to the WAV header")
    a = p.parse_args()

    ip = iface_ip(a.iface)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", a.port))
    s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, socket.inet_aton(a.group) + socket.inet_aton(ip))
    s.settimeout(2.0)
    print("listening on %s:%d via %s (%s) for %.1f s ..." % (a.group, a.port, a.iface, ip, a.seconds))
    data, sizes = bytearray(), []
    t0 = t_first = None
    t_end = time.monotonic() + a.seconds
    while time.monotonic() < t_end:
        try:
            pkt, src = s.recvfrom(65536)
        except socket.timeout:
            print("no packets for 2 s (wrong interface? voice service off?)")
            continue
        if t_first is None:
            t_first = time.monotonic()
            print("first packet from", src)
        data.extend(pkt)
        sizes.append(len(pkt))
    dur = (time.monotonic() - t_first) if t_first else 0
    if not data:
        print("FAIL: no audio received")
        return 1
    if len(data) % 2:
        data = data[:-1]
    samples = array.array("h", bytes(data))
    if sys.byteorder == "big":
        samples.byteswap()
    peak = max(abs(x) for x in samples) / 32768.0
    rms = math.sqrt(sum(x * x for x in samples) / len(samples)) / 32768.0
    with wave.open(a.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(a.rate)
        w.writeframes(bytes(data))
    rate = len(data) / dur if dur else 0
    print("packets: %d, size min/median/max: %d/%d/%d bytes, %.1f packets/s"
          % (len(sizes), min(sizes), sorted(sizes)[len(sizes) // 2], max(sizes), len(sizes) / dur))
    print("byte rate: %.0f B/s -> %.0f samples/s if mono int16 (expect ~32000 B/s = 16 kHz)" % (rate, rate / 2))
    print("level: rms %.4f, peak %.3f" % (rms, peak))
    print("wrote %s; listen to it and confirm it sounds right (speed and pitch)." % a.out)
    if not 28000 < rate < 36000:
        print("WARNING: byte rate does not match 16 kHz mono int16. Do not build on this assumption.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
