"""OpenAI-compatible streaming chat client (images as base64 data URLs)."""
from __future__ import annotations

import json
import logging
import threading
from typing import Dict, Iterator, List, Optional

log = logging.getLogger(__name__)


def build_messages(system: str, history: List[Dict[str, str]], user_text: str,
                   image_b64_jpeg: Optional[str]) -> List[dict]:
    msgs: List[dict] = [{"role": "system", "content": system}]
    msgs.extend(history)
    if image_b64_jpeg:
        msgs.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64_jpeg}},
            {"type": "text", "text": user_text},
        ]})
    else:
        msgs.append({"role": "user", "content": user_text})
    return msgs


class VLMClient:
    def __init__(self, url: str, model: str, max_tokens: int = 96, temperature: float = 0.4,
                 timeout_s: float = 30.0):
        import requests
        self.requests = requests
        self.url, self.model = url, model
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout_s

    def stream(self, messages: List[dict], cancel: Optional[threading.Event] = None) -> Iterator[str]:
        body = {"model": self.model, "messages": messages, "stream": True,
                "max_tokens": self.max_tokens, "temperature": self.temperature}
        with self.requests.post(self.url, json=body, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if cancel is not None and cancel.is_set():
                    return
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    obj = json.loads(data)
                except ValueError:
                    log.warning("bad SSE line: %r", data[:80])
                    continue
                for choice in obj.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta


class MockVLM:
    """In-process mock (no HTTP): echoes a canned reply word by word."""

    def __init__(self, reply: str = "Hello! I am running in mock mode. [gesture:wave]"):
        self.reply = reply

    def stream(self, messages: List[dict], cancel: Optional[threading.Event] = None) -> Iterator[str]:
        for i, w in enumerate(self.reply.split(" ")):
            if cancel is not None and cancel.is_set():
                return
            yield (" " if i else "") + w


def make_vlm(cfg):
    if cfg.vlm.backend == "mock":
        return MockVLM()
    return VLMClient(cfg.vlm.url, cfg.vlm.model, cfg.vlm.max_tokens, cfg.vlm.temperature,
                     cfg.vlm.timeout_s)
