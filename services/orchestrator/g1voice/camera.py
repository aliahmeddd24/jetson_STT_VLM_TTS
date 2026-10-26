"""Latest-frame-only camera buffer. Frames are attached only when the router asks."""
from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)


class FrameBuffer:
    def __init__(self, source: str, *, zmq_endpoint: str = "", http_url: str = "",
                 v4l2_device=0, poll_s: float = 0.1):
        self.source = source
        self.zmq_endpoint, self.http_url, self.v4l2_device = zmq_endpoint, http_url, v4l2_device
        self.poll_s = poll_s
        self._jpeg: Optional[bytes] = None
        self._t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "FrameBuffer":
        if self.source == "none":
            return self
        target = {"zmq": self._run_zmq, "http": self._run_http, "v4l2": self._run_v4l2}[self.source]
        self._thread = threading.Thread(target=self._guard, args=(target,), daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _guard(self, fn) -> None:
        while not self._stop.is_set():
            try:
                fn()
            except Exception as exc:
                log.warning("camera source %s error: %s (retrying in 2 s)", self.source, exc)
                self._stop.wait(2.0)

    def _put(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg, self._t = jpeg, time.monotonic()

    def _run_zmq(self) -> None:
        # e.g. avp_teleoperate image_server: PUB socket sending JPEG bytes. The newest
        # frame wins (CONFLATE keeps only the last message).
        import zmq  # type: ignore
        ctx = zmq.Context.instance()
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.RCVTIMEO, 2000)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        s.connect(self.zmq_endpoint)
        try:
            while not self._stop.is_set():
                try:
                    self._put(s.recv())
                except zmq.Again:
                    log.debug("no camera frame for 2 s")
        finally:
            s.close(0)

    def _run_http(self) -> None:
        import requests
        while not self._stop.is_set():
            r = requests.get(self.http_url, timeout=2)
            r.raise_for_status()
            self._put(r.content)
            self._stop.wait(self.poll_s)

    def _run_v4l2(self) -> None:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(self.v4l2_device)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video device {self.v4l2_device}")
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("frame grab failed")
                ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    self._put(enc.tobytes())
                self._stop.wait(self.poll_s)
        finally:
            cap.release()

    def latest_b64(self, max_px: int = 448, max_age_s: float = 1.0) -> Optional[str]:
        with self._lock:
            jpeg, t = self._jpeg, self._t
        if jpeg is None or time.monotonic() - t > max_age_s:
            return None
        return base64.b64encode(downscale_jpeg(jpeg, max_px)).decode("ascii")


def downscale_jpeg(jpeg: bytes, max_px: int) -> bytes:
    """Resize so the longest side is <= max_px. Returns input unchanged if OpenCV is missing."""
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError:
        return jpeg
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jpeg
    h, w = img.shape[:2]
    scale = max_px / float(max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return enc.tobytes() if ok else jpeg
