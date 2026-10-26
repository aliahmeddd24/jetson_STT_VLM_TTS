"""Client for the local action server, plus the heartbeat that arms its watchdog."""
from __future__ import annotations

import logging
import threading
from typing import Optional

log = logging.getLogger(__name__)


class ActionClient:
    def __init__(self, url: str, heartbeat_hz: float = 10.0, timeout_s: float = 0.3):
        import requests
        self.s = requests.Session()
        self.url = url.rstrip("/")
        self.period = 1.0 / heartbeat_hz
        self.timeout = timeout_s
        self._stop = threading.Event()
        self._hb: Optional[threading.Thread] = None

    def start_heartbeat(self) -> None:
        self._hb = threading.Thread(target=self._beat, daemon=True)
        self._hb.start()

    def close(self) -> None:
        self._stop.set()

    def _beat(self) -> None:
        import requests
        hb = requests.Session()   # separate session: never queued behind a slow command
        while not self._stop.wait(self.period):
            try:
                hb.post(self.url + "/heartbeat", timeout=self.timeout)
            except Exception:
                pass

    def _post(self, path: str, body: dict) -> dict:
        try:
            r = self.s.post(self.url + path, json=body, timeout=self.timeout)
            data = r.json() if r.content else {}
            data["http_status"] = r.status_code
            return data
        except Exception as exc:
            log.warning("action server %s failed: %s", path, exc)
            return {"ok": False, "error": str(exc)}

    def gesture(self, name: str) -> dict:
        return self._post("/gesture", {"name": name})

    def step(self, direction: str) -> dict:
        return self._post("/step", {"dir": direction})

    def stop(self) -> dict:
        return self._post("/stop", {})

    def status(self) -> dict:
        try:
            return self.s.get(self.url + "/status", timeout=self.timeout).json()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


class NullActions:
    """Used with --no-actions; records calls for tests."""

    def __init__(self):
        self.calls = []

    def start_heartbeat(self) -> None: ...
    def close(self) -> None: ...

    def gesture(self, name: str) -> dict:
        self.calls.append(("gesture", name))
        return {"ok": True, "dry_run": True}

    def step(self, direction: str) -> dict:
        self.calls.append(("step", direction))
        return {"ok": True, "dry_run": True}

    def stop(self) -> dict:
        self.calls.append(("stop", ""))
        return {"ok": True}

    def status(self) -> dict:
        return {"ok": True}
