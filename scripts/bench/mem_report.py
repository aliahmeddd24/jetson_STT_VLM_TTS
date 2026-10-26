#!/usr/bin/env python3
"""Summarise a timestamped tegrastats log: RAM/swap used, free headroom, temperatures.
Targets: >= 1.5 GB free headroom, steady-state swap < 500 MB. Stdlib only."""
import re
import sys

RAM = re.compile(r"RAM (\d+)/(\d+)MB")
SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
TEMP = re.compile(r"(\w+)@(-?[\d.]+)C")


def main(path):
    ram, swap, temps = [], [], {}
    for line in open(path):
        m = RAM.search(line)
        if m:
            ram.append((int(m.group(1)), int(m.group(2))))
        m = SWAP.search(line)
        if m:
            swap.append(int(m.group(1)))
        for name, v in TEMP.findall(line):
            temps.setdefault(name, []).append(float(v))
    if not ram:
        print("no RAM lines in", path)
        return 1
    total = ram[0][1]
    used = [u for u, _ in ram]
    tail = swap[len(swap) // 2:] or [0]          # "steady state" = second half of the run
    min_free = total - max(used)
    print("| metric | value | target | result |\n|---|---|---|---|")
    print("| RAM used max / total | %d / %d MB | | |" % (max(used), total))
    print("| min free headroom | %d MB | >= 1536 MB | %s |" % (min_free, "PASS" if min_free >= 1536 else "MISS"))
    print("| swap used, steady state (max of 2nd half) | %d MB | < 500 MB | %s |"
          % (max(tail), "PASS" if max(tail) < 500 else "MISS"))
    for name, v in sorted(temps.items()):
        print("| temp %s max | %.1f C | | |" % (name, max(v)))
    print("\nsamples: %d from %s" % (len(ram), path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
