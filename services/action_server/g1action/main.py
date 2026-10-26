"""python -m g1action.main --config config/action_server.yaml [--live]"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

import yaml

from .backend import DryRunBackend
from .server import ActionServer, serve

LIVE_PHRASE = "spotter-present-estop-in-hand"
log = logging.getLogger("g1action")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/action_server.yaml")
    p.add_argument("--live", action="store_true",
                   help=f"drive the real robot; also requires G1_LIVE_CONFIRM={LIVE_PHRASE}")
    p.add_argument("--dry-run-fsm-id", type=int, default=None, help="simulated FSM id in dry-run")
    p.add_argument("--cpus", default="", help="pin to these CPU ids, e.g. 7 or 6,7")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with open(a.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if a.cpus:
        os.sched_setaffinity(0, {int(c) for c in a.cpus.split(",")})

    live = a.live and not cfg.get("dry_run_forced", False)
    if live:
        if os.environ.get("G1_LIVE_CONFIRM") != LIVE_PHRASE:
            log.error("--live refused: set G1_LIVE_CONFIRM=%s only when the robot is ready, a spotter "
                      "is present and the e-stop/remote is in hand.", LIVE_PHRASE)
            return 2
        from .backend import UnitreeBackend
        iface = os.environ.get("G1_IFACE") or cfg.get("network_interface", "eth0")
        backend = UnitreeBackend(iface,
                                 weight_joint=cfg["arm"]["weight_joint_index"])
        log.warning("LIVE MODE: commands will move the robot")
    else:
        backend = DryRunBackend(fsm_id=a.dry_run_fsm_id)
        log.info("dry-run mode (no robot commands are sent)")

    srv = ActionServer(cfg, backend)
    httpd = serve(srv, cfg.get("bind_host", "127.0.0.1"), int(cfg.get("bind_port", 8090)))
    stop = threading.Event()

    def _sig(*_):
        stop.set()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    log.info("action server on %s:%s (live=%s) gestures=%s", cfg.get("bind_host"), cfg.get("bind_port"),
             live, sorted(srv.gestures))
    try:
        httpd.serve_forever()
    finally:
        srv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
