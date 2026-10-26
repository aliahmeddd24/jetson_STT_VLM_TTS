"""Text-to-speech backends. Interface: TTS.synth(text) -> iterator of 16 kHz int16 PCM chunks."""
from __future__ import annotations

import logging
import math
from typing import Iterator

import numpy as np

from .audio import float_to_pcm

log = logging.getLogger(__name__)


class TTS:
    def synth(self, text: str) -> Iterator[bytes]:
        raise NotImplementedError


class RivaTTS(TTS):
    def __init__(self, uri: str, voice: str, language: str = "en-US", sample_rate: int = 16000):
        import riva.client  # type: ignore
        self.rc = riva.client
        self.svc = riva.client.SpeechSynthesisService(riva.client.Auth(uri=uri))
        self.voice, self.language, self.sr = voice, language, sample_rate

    def synth(self, text: str) -> Iterator[bytes]:
        for resp in self.svc.synthesize_online(
                text, self.voice, self.language, sample_rate_hz=self.sr,
                encoding=self.rc.AudioEncoding.LINEAR_PCM):
            if resp.audio:
                yield resp.audio


class PiperHttpTTS(TTS):
    """services/tts_piper/server.py: POST {"text": ...} -> chunked raw 16 kHz PCM."""

    def __init__(self, url: str):
        import requests
        self.requests, self.url = requests, url

    def synth(self, text: str) -> Iterator[bytes]:
        with self.requests.post(self.url, json={"text": text}, stream=True, timeout=30) as r:
            r.raise_for_status()
            carry = b""
            for chunk in r.iter_content(chunk_size=3200):
                chunk = carry + chunk
                cut = len(chunk) - (len(chunk) % 2)
                carry = chunk[cut:]
                if cut:
                    yield chunk[:cut]


class MockTTS(TTS):
    """A soft tone, ~0.25 s per word, delivered in 100 ms chunks."""

    def __init__(self, seconds_per_word: float = 0.25):
        self.spw = seconds_per_word

    def synth(self, text: str) -> Iterator[bytes]:
        n = max(1, len(text.split()))
        t = np.arange(int(16000 * self.spw * n)) / 16000.0
        x = 0.1 * np.sin(2 * math.pi * 220.0 * t)
        pcm = float_to_pcm(x.astype(np.float32))
        for i in range(0, len(pcm), 3200):
            yield pcm[i:i + 3200]


def make_tts(cfg) -> TTS:
    b = cfg.tts.backend
    if b == "riva":
        return RivaTTS(cfg.tts.riva_uri, cfg.tts.voice, cfg.tts.language, cfg.tts.sample_rate)
    if b == "piper_http":
        return PiperHttpTTS(cfg.tts.piper_url)
    if b == "mock":
        return MockTTS()
    raise ValueError(f"unknown tts backend {b!r}")
