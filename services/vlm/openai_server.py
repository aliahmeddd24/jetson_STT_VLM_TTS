#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server (SSE streaming) for the VLM.

Backends:
  nanollm  NanoLLM (MLC + TensorRT vision encoder), e.g. VILA1.5-3B, inside dustynv/nano_llm r35
  mock     canned reply with configurable per-token delay (tests, dry runs)

Python 3.8, standard library only, so it runs unchanged in the nano_llm r35 image.
One request at a time (one GPU model); concurrent requests queue on a lock.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator, List, Optional, Tuple

log = logging.getLogger("vlm")


# ---------------------------------------------------------------- message parsing
def split_messages(messages: List[dict]) -> Tuple[str, List[Tuple[str, str]], str, Optional[bytes]]:
    """-> (system, [(role, text) history], last_user_text, last_user_image_bytes)."""
    system = ""
    turns: List[Tuple[str, str]] = []
    image: Optional[bytes] = None
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        texts: List[str] = []
        if isinstance(content, str):
            texts.append(content)
        else:
            for part in content or []:
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "image_url" and i == len(messages) - 1:
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:") and "," in url:
                        image = base64.b64decode(url.split(",", 1)[1])
        text = " ".join(t for t in texts if t)
        if role == "system":
            system = (system + "\n" + text).strip()
        else:
            turns.append((role, text))
    if not turns or turns[-1][0] != "user":
        raise ValueError("last message must be from the user")
    return system, turns[:-1], turns[-1][1], image


# ---------------------------------------------------------------- backends
class MockBackend:
    def __init__(self, reply: str, token_delay_s: float, first_token_delay_s: float):
        self.reply, self.delay, self.first = reply, token_delay_s, first_token_delay_s

    def generate(self, system, history, text, image, max_tokens, temperature) -> Iterator[str]:
        time.sleep(self.first)
        reply = self.reply
        if image is not None:
            reply = "I can see an image of %d bytes. %s" % (len(image), reply)
        for i, w in enumerate(reply.split(" ")[:max_tokens]):
            if i:
                time.sleep(self.delay)
            yield (" " if i else "") + w


class NanoLLMBackend:
    def __init__(self, model: str, api: str, quantization: str, max_context_len: int, vision_api: str):
        from nano_llm import ChatHistory, NanoLLM  # type: ignore
        self.ChatHistory = ChatHistory
        t0 = time.time()
        self.model = NanoLLM.from_pretrained(model, api=api, quantization=quantization,
                                             max_context_len=max_context_len, vision_api=vision_api)
        log.info("model loaded in %.1f s", time.time() - t0)

    def _append(self, chat, role: str, **kw) -> None:
        try:
            chat.append(role=role, **kw)
        except TypeError:   # older/newer signatures: append(role, msg)
            chat.append(role, next(iter(kw.values())))

    def generate(self, system, history, text, image, max_tokens, temperature) -> Iterator[str]:
        from PIL import Image  # type: ignore
        chat = self.ChatHistory(self.model, system_prompt=system or None)
        try:
            for role, msg in history:
                self._append(chat, "bot" if role == "assistant" else "user", msg=msg)
            if image is not None:
                self._append(chat, "user", image=Image.open(io.BytesIO(image)).convert("RGB"))
            self._append(chat, "user", msg=text)
            embedding, _ = chat.embed_chat()
            reply = self.model.generate(embedding, streaming=True, kv_cache=chat.kv_cache,
                                        stop_tokens=chat.template.stop, max_new_tokens=max_tokens,
                                        do_sample=temperature > 0, temperature=max(temperature, 1e-3))
            for token in reply:
                yield token
        finally:
            try:
                chat.reset()
            except Exception:
                pass


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    backend = None
    lock = threading.Lock()
    model_name = "vlm"

    def log_message(self, fmt, *args):
        log.debug(fmt, *args)

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/v1/health"):
            self._json(200, {"ok": True})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.model_name, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            system, history, text, image = split_messages(req.get("messages", []))
        except Exception as exc:
            return self._json(400, {"error": str(exc)})
        max_tokens = int(req.get("max_tokens") or 128)
        temperature = float(req.get("temperature") if req.get("temperature") is not None else 0.4)
        stream = bool(req.get("stream"))
        cid = "chatcmpl-" + uuid.uuid4().hex[:12]

        with self.lock:
            t0 = time.monotonic()
            t_first = None
            n = 0
            pieces: List[str] = []
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
            try:
                for tok in self.backend.generate(system, history, text, image, max_tokens, temperature):
                    if t_first is None:
                        t_first = time.monotonic()
                    n += 1
                    pieces.append(tok)
                    if stream:
                        ev = {"id": cid, "object": "chat.completion.chunk", "model": self.model_name,
                              "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]}
                        self._chunk(b"data: " + json.dumps(ev).encode() + b"\n\n")
            except (BrokenPipeError, ConnectionResetError):
                log.info("client disconnected")
                return
            except Exception as exc:
                log.exception("generation failed")
                if not stream:
                    return self._json(500, {"error": str(exc)})
            t_end = time.monotonic()
            timing = {"ttft_s": round((t_first or t_end) - t0, 4), "tokens": n,
                      "decode_tok_s": round((n - 1) / (t_end - t_first), 2) if t_first and n > 1 and t_end > t_first else None,
                      "image": image is not None}
            log.info("request done: %s", timing)
            if stream:
                ev = {"id": cid, "object": "chat.completion.chunk", "model": self.model_name,
                      "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                      "usage": {"completion_tokens": n}, "x_timing": timing}
                self._chunk(b"data: " + json.dumps(ev).encode() + b"\n\n")
                self._chunk(b"data: [DONE]\n\n")
                self._chunk(b"")
            else:
                self._json(200, {"id": cid, "object": "chat.completion", "model": self.model_name,
                                 "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": "".join(pieces)}}],
                                 "usage": {"completion_tokens": n}, "x_timing": timing})


def make_server(backend, host: str, port: int, model_name: str) -> ThreadingHTTPServer:
    Handler.backend = backend
    Handler.model_name = model_name
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["nanollm", "mock"], default="nanollm")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", default="/models/VILA1.5-3b")
    p.add_argument("--model-name", default="vila1.5-3b")
    p.add_argument("--api", default="mlc")
    p.add_argument("--quantization", default="q4f16_ft")
    p.add_argument("--max-context-len", type=int, default=1536)
    p.add_argument("--vision-api", default="auto", help="auto | hf | trt")
    p.add_argument("--warmup", action="store_true", help="run one text request at startup")
    p.add_argument("--mock-reply", default="Hello! I am a mock model. [gesture:wave]")
    p.add_argument("--mock-token-delay", type=float, default=0.03)
    p.add_argument("--mock-first-delay", type=float, default=0.2)
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.backend == "mock":
        backend = MockBackend(a.mock_reply, a.mock_token_delay, a.mock_first_delay)
    else:
        backend = NanoLLMBackend(a.model, a.api, a.quantization, a.max_context_len, a.vision_api)
    if a.warmup:
        list(backend.generate("You are helpful.", [], "Say hi.", None, 8, 0.0))
    srv = make_server(backend, a.host, a.port, a.model_name)
    log.info("listening on %s:%d (backend=%s)", a.host, a.port, a.backend)
    srv.serve_forever()


if __name__ == "__main__":
    main()
