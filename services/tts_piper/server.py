#!/usr/bin/env python3
"""Piper TTS over HTTP: POST /synthesize {"text": ...} -> chunked raw 16 kHz mono int16 PCM.

Runs on CPU by default (acceptable: only the first sentence must finish before
audio starts). --cuda uses onnxruntime-gpu if a working build is installed.
Supports piper-tts 1.2 (synthesize_stream_raw) and 1.3+ (synthesize -> AudioChunk).
"""
import argparse
import json
import logging
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "orchestrator"))
from g1voice.audio import resample  # noqa: E402

log = logging.getLogger("piper")
OUT_HZ = 16000


class Piper:
    def __init__(self, model: str, cuda: bool):
        from piper.voice import PiperVoice  # type: ignore
        self.voice = PiperVoice.load(model, config_path=model + ".json", use_cuda=cuda)
        self.sr = int(self.voice.config.sample_rate)
        self.lock = threading.Lock()

    def stream(self, text: str):
        with self.lock:
            if hasattr(self.voice, "synthesize_stream_raw"):          # piper-tts 1.2
                for raw in self.voice.synthesize_stream_raw(text):
                    yield self._to16k(np.frombuffer(raw, dtype="<i2"), self.sr)
            else:                                                       # piper-tts 1.3+
                for chunk in self.voice.synthesize(text):
                    yield self._to16k(np.frombuffer(chunk.audio_int16_bytes, dtype="<i2"), chunk.sample_rate)

    @staticmethod
    def _to16k(pcm: np.ndarray, sr: int) -> bytes:
        x = pcm.astype(np.float32) / 32768.0
        y = resample(x, sr, OUT_HZ)
        return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()


def make_handler(piper: Piper):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            log.debug(fmt, *a)

        def do_GET(self):
            body = b'{"ok": true}'
            self.send_response(200 if self.path == "/health" else 404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/synthesize":
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            text = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))).get("text", "")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Sample-Rate", str(OUT_HZ))
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for pcm in piper.stream(text):
                    if pcm:
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(pcm), pcm))
                        self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="models/piper/en_US-lessac-medium.onnx")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--cuda", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    piper = Piper(a.model, a.cuda)
    list(piper.stream("Warm up."))
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(piper))
    log.info("piper on %s:%d (voice sr=%d, cuda=%s)", a.host, a.port, piper.sr, a.cuda)
    srv.serve_forever()


if __name__ == "__main__":
    main()
