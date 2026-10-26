"""Audio sources and sinks. All PCM is 16 kHz mono signed 16-bit little-endian."""
from __future__ import annotations

import fcntl
import logging
import socket
import struct
import threading
import time
import wave
from typing import Iterator, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2


def pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def rms(pcm: bytes) -> float:
    x = pcm_to_float(pcm)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def resample(x: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
    """Band-limited resample when soxr is installed, linear interpolation otherwise."""
    if src_hz == dst_hz or x.size == 0:
        return x
    try:
        import soxr  # type: ignore
        return soxr.resample(x, src_hz, dst_hz).astype(np.float32)
    except ImportError:
        n_out = int(round(x.size * dst_hz / src_hz))
        t_out = np.linspace(0, x.size - 1, n_out)
        return np.interp(t_out, np.arange(x.size), x).astype(np.float32)


def iface_ipv4(ifname: str) -> str:
    """IPv4 address of a Linux network interface (SIOCGIFADDR)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", ifname[:15].encode()))
        return socket.inet_ntoa(packed[20:24])
    finally:
        s.close()


# ---------------------------------------------------------------- sources
class AudioSource:
    """Yields (pcm_frame, t_monotonic) with frames of exactly frame_samples samples."""

    def frames(self, frame_samples: int) -> Iterator[Tuple[bytes, float]]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class G1MulticastMic(AudioSource):
    """G1 microphone stream: UDP multicast of raw PCM (VERIFY format with capture_g1_mic.py)."""

    def __init__(self, group: str, port: int, local_ip: str, timeout_s: float = 1.0):
        self.group, self.port, self.local_ip = group, port, local_ip
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", port))
        mreq = socket.inet_aton(group) + socket.inet_aton(local_ip)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.sock.settimeout(timeout_s)
        self._closed = False
        log.info("joined multicast %s:%d on %s", group, port, local_ip)

    def frames(self, frame_samples: int) -> Iterator[Tuple[bytes, float]]:
        need = frame_samples * BYTES_PER_SAMPLE
        buf = bytearray()
        while not self._closed:
            try:
                data, _ = self.sock.recvfrom(65536)
            except socket.timeout:
                log.warning("no mic packets for 1 s")
                continue
            except OSError:
                if self._closed:
                    return
                raise
            buf.extend(data)
            now = time.monotonic()
            while len(buf) >= need:
                frame = bytes(buf[:need])
                del buf[:need]
                yield frame, now

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass


class WavSource(AudioSource):
    """Replays a 16 kHz mono 16-bit WAV, optionally paced in real time, then trailing silence."""

    def __init__(self, path: str, realtime: bool = True, tail_silence_s: float = 1.5):
        self.path, self.realtime, self.tail = path, realtime, tail_silence_s
        with wave.open(path, "rb") as w:
            if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
                raise ValueError(f"{path}: need 16 kHz mono 16-bit, got "
                                 f"{w.getframerate()} Hz {w.getnchannels()} ch {8 * w.getsampwidth()} bit")
            self.pcm = w.readframes(w.getnframes())

    def frames(self, frame_samples: int) -> Iterator[Tuple[bytes, float]]:
        need = frame_samples * BYTES_PER_SAMPLE
        data = self.pcm + b"\x00" * int(self.tail * SAMPLE_RATE * BYTES_PER_SAMPLE)
        frame_s = frame_samples / SAMPLE_RATE
        t0 = time.monotonic()
        for i, off in enumerate(range(0, len(data) - need + 1, need)):
            if self.realtime:
                delay = t0 + i * frame_s - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            yield data[off:off + need], time.monotonic()


class NullSource(AudioSource):
    def frames(self, frame_samples: int) -> Iterator[Tuple[bytes, float]]:
        silence = b"\x00" * frame_samples * BYTES_PER_SAMPLE
        while True:
            time.sleep(frame_samples / SAMPLE_RATE)
            yield silence, time.monotonic()


# ---------------------------------------------------------------- sinks
class AudioSink:
    """One utterance = begin() .. write()* .. end(). stop() aborts playback."""

    def begin(self, stream_id: str) -> None: ...
    def write(self, pcm: bytes) -> None: ...
    def end(self) -> None: ...
    def stop(self) -> None: ...


class PacedSink(AudioSink):
    """Splits audio into chunk_ms pieces and keeps at most lead_ms ahead of real time."""

    def __init__(self, chunk_ms: int = 100, lead_ms: int = 200):
        self.chunk_bytes = int(SAMPLE_RATE * chunk_ms / 1000) * BYTES_PER_SAMPLE
        self.lead_s = lead_ms / 1000.0
        self._play_until = 0.0
        self._pending = bytearray()
        self._stopped = threading.Event()

    def begin(self, stream_id: str) -> None:
        self.stream_id = stream_id
        self._pending.clear()
        self._stopped.clear()
        self._play_until = max(self._play_until, time.monotonic())

    def write(self, pcm: bytes) -> None:
        self._pending.extend(pcm)
        while len(self._pending) >= self.chunk_bytes and not self._stopped.is_set():
            self._push(bytes(self._pending[:self.chunk_bytes]))
            del self._pending[:self.chunk_bytes]

    def end(self) -> None:
        if self._pending and not self._stopped.is_set():
            self._push(bytes(self._pending))
        self._pending.clear()
        wait = self._play_until - time.monotonic()
        if wait > 0 and not self._stopped.is_set():
            self._stopped.wait(wait)

    def stop(self) -> None:
        self._stopped.set()
        self._pending.clear()
        self._play_until = time.monotonic()

    def _push(self, chunk: bytes) -> None:
        ahead = self._play_until - time.monotonic()
        if ahead > self.lead_s:
            self._stopped.wait(ahead - self.lead_s)
        self._send(chunk)
        dur = len(chunk) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
        self._play_until = max(self._play_until, time.monotonic()) + dur

    def _send(self, chunk: bytes) -> None:
        raise NotImplementedError

    @property
    def busy_until(self) -> float:
        return self._play_until


class WavSink(PacedSink):
    """Writes everything played to a WAV file (for offline tests). Optionally paced."""

    def __init__(self, path: str, paced: bool = False, **kw):
        super().__init__(**kw)
        self.paced = paced
        self.path = path
        self._w = wave.open(path, "wb")
        self._w.setnchannels(1)
        self._w.setsampwidth(2)
        self._w.setframerate(SAMPLE_RATE)
        self.utterances = 0

    def begin(self, stream_id: str) -> None:
        super().begin(stream_id)
        self.utterances += 1

    def _push(self, chunk: bytes) -> None:
        if self.paced:
            super()._push(chunk)
        else:
            self._send(chunk)

    def _send(self, chunk: bytes) -> None:
        self._w.writeframes(chunk)

    def close(self) -> None:
        self._w.close()


class NullSink(PacedSink):
    def _send(self, chunk: bytes) -> None:
        pass


class G1SpeakerSink(PacedSink):
    """Plays through AudioClient.PlayStream on one stream id per utterance."""

    def __init__(self, robot, app_name: str, **kw):
        super().__init__(**kw)
        self.robot, self.app_name = robot, app_name

    def _send(self, chunk: bytes) -> None:
        self.robot.play_stream(self.app_name, self.stream_id, chunk)

    def stop(self) -> None:
        super().stop()
        self.robot.play_stop(self.app_name)
