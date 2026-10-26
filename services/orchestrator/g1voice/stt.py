"""Speech-to-text backends behind one streaming interface.

Interface: STT.open_stream(on_partial) -> STTStream with feed(pcm), finish() -> str, abort().
Input is 16 kHz mono int16 PCM. Partials are delivered on a background thread.
"""
from __future__ import annotations

import io
import logging
import queue
import threading
import wave
from typing import Callable, Iterator, List, Optional

log = logging.getLogger(__name__)
PartialCb = Optional[Callable[[str], None]]
_END = object()


class STTStream:
    def feed(self, pcm: bytes) -> None: raise NotImplementedError
    def finish(self, timeout_s: float = 10.0) -> str: raise NotImplementedError
    def abort(self) -> None: ...


class STT:
    supports_partials = False

    def open_stream(self, on_partial: PartialCb = None) -> STTStream:
        raise NotImplementedError


# ------------------------------------------------------------------ Riva
class RivaSTT(STT):
    supports_partials = True

    def __init__(self, uri: str, language: str = "en-US"):
        import riva.client  # type: ignore
        self.rc = riva.client
        self.auth = riva.client.Auth(uri=uri)
        self.asr = riva.client.ASRService(self.auth)
        self.config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                language_code=language,
                max_alternatives=1,
                enable_automatic_punctuation=True,
                sample_rate_hertz=16000,
                audio_channel_count=1,
            ),
            interim_results=True,
        )

    def open_stream(self, on_partial: PartialCb = None) -> STTStream:
        return _RivaStream(self, on_partial)


class _RivaStream(STTStream):
    def __init__(self, owner: RivaSTT, on_partial: PartialCb):
        self.q: "queue.Queue" = queue.Queue()
        self.finals: List[str] = []
        self.on_partial = on_partial
        self.done = threading.Event()
        self.error: Optional[BaseException] = None
        self._aborted = False
        self.thread = threading.Thread(target=self._run, args=(owner,), daemon=True)
        self.thread.start()

    def _chunks(self) -> Iterator[bytes]:
        while True:
            item = self.q.get()
            if item is _END:
                return
            yield item

    def _run(self, owner: RivaSTT) -> None:
        try:
            for resp in owner.asr.streaming_response_generator(
                    audio_chunks=self._chunks(), streaming_config=owner.config):
                for result in resp.results:
                    if not result.alternatives:
                        continue
                    text = result.alternatives[0].transcript.strip()
                    if result.is_final:
                        if text:
                            self.finals.append(text)
                    elif self.on_partial and text and not self._aborted:
                        self.on_partial(" ".join(self.finals + [text]))
        except BaseException as exc:  # surfaced in finish()
            self.error = exc
        finally:
            self.done.set()

    def feed(self, pcm: bytes) -> None:
        self.q.put(pcm)

    def finish(self, timeout_s: float = 10.0) -> str:
        self.q.put(_END)
        if not self.done.wait(timeout_s):
            raise TimeoutError("Riva ASR did not return a final transcript")
        if self.error and not self._aborted:
            raise RuntimeError(f"Riva ASR error: {self.error}")
        return " ".join(self.finals).strip()

    def abort(self) -> None:
        self._aborted = True
        self.q.put(_END)


# ------------------------------------------------- whisper.cpp HTTP server
class WhisperHttpSTT(STT):
    """whisper.cpp `whisper-server` (POST /inference, multipart). No partials."""

    def __init__(self, url: str):
        import requests
        self.requests = requests
        self.url = url

    def open_stream(self, on_partial: PartialCb = None) -> STTStream:
        return _BufferedStream(self._transcribe)

    def _transcribe(self, pcm: bytes) -> str:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(pcm)
        r = self.requests.post(self.url, files={"file": ("u.wav", buf.getvalue(), "audio/wav")},
                               data={"response_format": "json", "temperature": "0.0"}, timeout=30)
        r.raise_for_status()
        return str(r.json().get("text", "")).strip()


class _BufferedStream(STTStream):
    def __init__(self, fn: Callable[[bytes], str]):
        self.fn = fn
        self.buf = bytearray()

    def feed(self, pcm: bytes) -> None:
        self.buf.extend(pcm)

    def finish(self, timeout_s: float = 10.0) -> str:
        return self.fn(bytes(self.buf))

    def abort(self) -> None:
        self.buf.clear()


# ------------------------------------------------------------------ mock
class MockSTT(STT):
    """Returns scripted transcripts in order (tests / dry runs). Emits the text as one partial."""

    supports_partials = True

    def __init__(self, transcripts: Optional[List[str]] = None, default: str = "hello robot"):
        self.transcripts = list(transcripts or [])
        self.default = default

    def open_stream(self, on_partial: PartialCb = None) -> STTStream:
        text = self.transcripts.pop(0) if self.transcripts else self.default
        return _MockStream(text, on_partial)


class _MockStream(STTStream):
    def __init__(self, text: str, on_partial: PartialCb):
        self.text, self.on_partial, self.sent = text, on_partial, False

    def feed(self, pcm: bytes) -> None:
        if not self.sent and self.on_partial:
            self.sent = True
            self.on_partial(self.text)

    def finish(self, timeout_s: float = 10.0) -> str:
        return self.text


def make_stt(cfg) -> STT:
    b = cfg.stt.backend
    if b == "riva":
        return RivaSTT(cfg.stt.riva_uri, cfg.stt.language)
    if b == "whisper_http":
        return WhisperHttpSTT(cfg.stt.whisper_url)
    if b == "mock":
        return MockSTT()
    raise ValueError(f"unknown stt backend {b!r}")
