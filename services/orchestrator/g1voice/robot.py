"""Thin adapter around unitree_sdk2_python for audio, LED and volume.

Only the orchestrator's *audio/LED* calls live here. Motion lives exclusively in
the action server. ChannelFactoryInitialize may only be called once per process.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional, Sequence

log = logging.getLogger(__name__)
_init_lock = threading.Lock()
_initialized = False


def init_dds(domain: int, iface: str) -> None:
    global _initialized
    with _init_lock:
        if _initialized:
            return
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # type: ignore
        ChannelFactoryInitialize(domain, iface)
        _initialized = True


class G1Audio:
    def __init__(self, domain: int, iface: str, timeout_s: float = 3.0):
        init_dds(domain, iface)
        from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient  # type: ignore
        self.client = AudioClient()
        self.client.SetTimeout(timeout_s)
        self.client.Init()
        self._lock = threading.Lock()
        self._bytes_ok: Optional[bool] = None

    def play_stream(self, app_name: str, stream_id: str, pcm: bytes) -> None:
        with self._lock:
            # Different SDK revisions take bytes or a list of ints; detect once.
            if self._bytes_ok is not False:
                try:
                    code = self.client.PlayStream(app_name, stream_id, pcm)
                    self._bytes_ok = True
                    self._check(code, "PlayStream")
                    return
                except TypeError:
                    self._bytes_ok = False
            code = self.client.PlayStream(app_name, stream_id, list(pcm))
            self._check(code, "PlayStream")

    def play_stop(self, app_name: str) -> None:
        with self._lock:
            try:
                self._check(self.client.PlayStop(app_name), "PlayStop")
            except Exception as exc:  # never let stop raise
                log.warning("PlayStop failed: %s", exc)

    def led(self, rgb: Sequence[int]) -> None:
        with self._lock:
            try:
                self._check(self.client.LedControl(int(rgb[0]), int(rgb[1]), int(rgb[2])), "LedControl")
            except Exception as exc:
                log.debug("LED failed: %s", exc)

    def set_volume(self, volume: int) -> None:
        with self._lock:
            self._check(self.client.SetVolume(int(volume)), "SetVolume")

    def get_volume(self):
        with self._lock:
            return self.client.GetVolume()

    @staticmethod
    def _check(ret, what: str) -> None:
        code = ret[0] if isinstance(ret, tuple) else ret
        if code not in (0, None):
            log.warning("%s returned code %s", what, code)


class NullRobot:
    """Stand-in when running without the robot (tests, WAV mode)."""

    def play_stream(self, *a, **k) -> None: ...
    def play_stop(self, *a, **k) -> None: ...
    def led(self, rgb) -> None: ...
    def set_volume(self, v) -> None: ...
    def get_volume(self): return None
