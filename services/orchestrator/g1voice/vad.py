"""Streaming voice activity detection with Silero (ONNX, CPU).

Loop design follows the Reachy assistant: RMS pre-filter before the model,
512-sample chunks at 16 kHz, a lookback buffer so the first syllable is not
lost, and silence-duration endpointing.
"""
from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional

import numpy as np

from .audio import SAMPLE_RATE, pcm_to_float

log = logging.getLogger(__name__)


class SileroVAD:
    """Supports Silero v5 (input/state/sr + 64-sample context) and v4 (input/sr/h/c)."""

    def __init__(self, model_path: str):
        import onnxruntime as ort  # CPU on purpose
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(model_path, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        names = {i.name for i in self.sess.get_inputs()}
        self.v5 = "state" in names
        self.reset()

    def reset(self) -> None:
        if self.v5:
            self.state = np.zeros((2, 1, 128), dtype=np.float32)
            self.context = np.zeros((1, 64), dtype=np.float32)
        else:
            self.h = np.zeros((2, 1, 64), dtype=np.float32)
            self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def __call__(self, x: np.ndarray) -> float:
        x = x.reshape(1, -1).astype(np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        if self.v5:
            inp = np.concatenate([self.context, x], axis=1)
            out, self.state = self.sess.run(None, {"input": inp, "state": self.state, "sr": sr})
            self.context = inp[:, -64:]
        else:
            out, self.h, self.c = self.sess.run(None, {"input": x, "sr": sr, "h": self.h, "c": self.c})
        return float(np.asarray(out).reshape(-1)[0])


@dataclass
class VadEvent:
    kind: str            # "start" | "audio" | "end" | "cancel"
    pcm: bytes = b""
    t: float = 0.0       # monotonic time of the frame that triggered the event
    speech_end_t: float = 0.0   # for "end": estimated time the user actually stopped talking


class Endpointer:
    """Turns per-frame speech probabilities into utterance events."""

    def __init__(self, model: Callable[[np.ndarray], float], *, threshold: float = 0.5,
                 rms_floor: float = 0.004, chunk_samples: int = 512, lookback_ms: int = 300,
                 min_speech_ms: int = 250, end_silence_ms: int = 500, max_utterance_s: float = 15.0):
        self.model = model
        self.thr = threshold
        self.neg_thr = max(threshold - 0.15, 0.01)
        self.rms_floor = rms_floor
        self.frame_s = chunk_samples / SAMPLE_RATE
        self.lookback: Deque[bytes] = collections.deque(maxlen=max(1, int(lookback_ms / 1000 / self.frame_s)))
        self.min_speech = min_speech_ms / 1000.0
        self.end_silence = end_silence_ms / 1000.0
        self.max_utt = max_utterance_s
        self.in_speech = False
        self.speech_s = 0.0
        self.silence_s = 0.0
        self.utt_s = 0.0
        self.last_voiced_t = 0.0

    def reset(self) -> None:
        self.in_speech = False
        self.speech_s = self.silence_s = self.utt_s = 0.0
        self.lookback.clear()
        if hasattr(self.model, "reset"):
            self.model.reset()

    def process(self, pcm: bytes, t: float) -> List[VadEvent]:
        x = pcm_to_float(pcm)
        level = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        prob = self.model(x) if level >= self.rms_floor else 0.0
        events: List[VadEvent] = []
        if not self.in_speech:
            if prob >= self.thr:
                self.in_speech = True
                self.speech_s = self.frame_s
                self.silence_s = 0.0
                self.utt_s = self.frame_s * (len(self.lookback) + 1)
                self.last_voiced_t = t
                events.append(VadEvent("start", b"".join(self.lookback) + pcm, t))
                self.lookback.clear()
            else:
                self.lookback.append(pcm)
            return events

        self.utt_s += self.frame_s
        events.append(VadEvent("audio", pcm, t))
        if prob >= self.neg_thr:
            self.speech_s += self.frame_s
            self.silence_s = 0.0
            self.last_voiced_t = t
        else:
            self.silence_s += self.frame_s
        if self.silence_s >= self.end_silence or self.utt_s >= self.max_utt:
            kind = "end" if self.speech_s >= self.min_speech else "cancel"
            events.append(VadEvent(kind, b"", t, speech_end_t=self.last_voiced_t))
            self.in_speech = False
            self.speech_s = self.silence_s = self.utt_s = 0.0
        return events


class EnergyVAD:
    """Model-free stand-in (tests, or if the Silero file is missing): RMS -> pseudo-probability."""

    def __init__(self, on_level: float = 0.02):
        self.on_level = on_level

    def reset(self) -> None:
        pass

    def __call__(self, x: np.ndarray) -> float:
        level = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        return min(1.0, level / self.on_level * 0.5)
