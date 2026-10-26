"""Timestamped JSONL event log. Uses time.monotonic() for latency and wall time for humans."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any


class Metrics:
    def __init__(self, path: str):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def event(self, turn: int, event: str, /, t: float = None, **fields: Any) -> None:
        """turn and event are positional-only so any keyword (e.g. name=...) can be a field."""
        rec = {"turn": turn, "event": event, "t": time.monotonic() if t is None else t,
               "wall": time.time()}
        rec.update(fields)
        with self._lock:
            self._fh.write(json.dumps(rec) + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()
