"""Streaming text processing between the VLM and TTS.

GestureTagFilter strips [gesture:name] tags from a token stream (tags may be
split across tokens) and reports allowlisted gestures. SentenceChunker flushes
text to TTS at sentence punctuation, or after N words (3 for the first chunk,
8 afterwards) so audio can start while generation continues.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, List, Optional, Tuple

TAG_RE = re.compile(r"\[\s*gesture\s*:\s*([A-Za-z0-9_\-]+)\s*\]")
_MAX_TAG = 40
_SENTENCE_END = (".", "!", "?", ";", ":")
_MD = re.compile(r"[*_#`~>|]+")


class GestureTagFilter:
    def __init__(self, allowed: Iterable[str], on_gesture: Optional[Callable[[str], None]] = None,
                 on_rejected: Optional[Callable[[str], None]] = None):
        self.allowed = set(allowed)
        self.on_gesture = on_gesture
        self.on_rejected = on_rejected
        self.buf = ""
        self.fired: List[str] = []

    def feed(self, text: str) -> str:
        """Returns text safe to speak; holds back anything that may be a partial tag."""
        self.buf += text
        out = []
        while self.buf:
            i = self.buf.find("[")
            if i < 0:
                out.append(self.buf)
                self.buf = ""
                break
            out.append(self.buf[:i])
            self.buf = self.buf[i:]
            j = self.buf.find("]")
            if j < 0:
                if len(self.buf) > _MAX_TAG:      # not a tag: release the bracket as text
                    out.append(self.buf[0])
                    self.buf = self.buf[1:]
                    continue
                break                              # wait for more tokens
            candidate = self.buf[:j + 1]
            self.buf = self.buf[j + 1:]
            m = TAG_RE.fullmatch(candidate)
            if m:
                self._handle(m.group(1).lower())
            else:
                out.append(candidate)
        return "".join(out)

    def flush(self) -> str:
        rest = self.buf
        self.buf = ""
        m = TAG_RE.search(rest)
        if m:
            self._handle(m.group(1).lower())
            rest = TAG_RE.sub("", rest)
        return rest

    def _handle(self, name: str) -> None:
        if name in self.allowed and name not in self.fired:
            self.fired.append(name)
            if self.on_gesture:
                self.on_gesture(name)
        elif name not in self.allowed and self.on_rejected:
            self.on_rejected(name)


def clean_for_speech(text: str) -> str:
    text = _MD.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceChunker:
    def __init__(self, first_words: int = 3, later_words: int = 8):
        self.first_words, self.later_words = first_words, later_words
        self.buf = ""
        self.emitted = 0

    def _limit(self) -> int:
        return self.first_words if self.emitted == 0 else self.later_words

    def feed(self, text: str) -> List[str]:
        self.buf += text
        out: List[str] = []
        while True:
            chunk = self._take()
            if chunk is None:
                break
            out.append(chunk)
        return out

    def _take(self) -> Optional[str]:
        # 1) sentence punctuation followed by whitespace
        m = re.search(r"[.!?;:](?=\s)", self.buf)
        # 2) N complete words (a word is complete once whitespace follows it)
        words = list(re.finditer(r"\S+(?=\s)", self.buf))
        cut = None
        if m:
            cut = m.end()
        if len(words) >= self._limit():
            wcut = words[self._limit() - 1].end()
            cut = wcut if cut is None else min(cut, wcut)
        if cut is None:
            return None
        chunk = clean_for_speech(self.buf[:cut])
        self.buf = self.buf[cut:]
        if not chunk:
            return self._take() if self.buf.strip() else None
        self.emitted += 1
        return chunk

    def flush(self) -> List[str]:
        chunk = clean_for_speech(self.buf)
        self.buf = ""
        if chunk:
            self.emitted += 1
            return [chunk]
        return []


def needs_vision(text: str, keywords: Iterable[str]) -> bool:
    t = " " + re.sub(r"[^a-z0-9' ]+", " ", text.lower()) + " "
    return any((" " + k.lower() + " ") in t for k in keywords)


def has_wake_word(text: str, wake_words: Iterable[str]) -> bool:
    t = text.lower()
    return any(re.search(r"\b" + re.escape(w.lower()) + r"\b", t) for w in wake_words)


class CommandGrammar:
    """Maps (partial or final) transcripts to allowlisted commands.

    Stop phrases never require the wake word. Motion requires it when configured.
    Each action fires at most once per utterance (call reset() per utterance).
    """

    def __init__(self, rules: Iterable[dict], stop_phrases: Iterable[str], wake_words: Iterable[str],
                 require_wake_word_for_motion: bool = True):
        self.rules: List[Tuple[re.Pattern, str, str]] = [
            (re.compile(r["pattern"], re.I), r["action"], str(r.get("arg", ""))) for r in rules]
        self.stop_res = [re.compile(r"\b" + re.escape(p) + r"\b", re.I) for p in stop_phrases]
        self.wake_words = list(wake_words)
        self.require_wake = require_wake_word_for_motion
        self.fired: set = set()

    def reset(self) -> None:
        self.fired = set()

    def match(self, text: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if any(r.search(text) for r in self.stop_res) and ("stop", "") not in self.fired:
            self.fired.add(("stop", ""))
            out.append(("stop", ""))
            return out              # never combine stop with motion
        if self.require_wake and not has_wake_word(text, self.wake_words):
            return out
        for rx, action, arg in self.rules:
            if rx.search(text) and (action, arg) not in self.fired:
                self.fired.add((action, arg))
                out.append((action, arg))
        return out

    def is_command_only(self, text: str, max_extra_words: int = 2) -> bool:
        """True if the utterance is basically just a command (skip the LLM turn)."""
        t = re.sub(r"[^a-z ]+", " ", text.lower())
        for w in self.wake_words:
            t = re.sub(r"\b" + re.escape(w.lower()) + r"\b", " ", t)
        for rx, _, _ in self.rules:
            t = rx.sub(" ", t)
        for rx in self.stop_res:
            t = rx.sub(" ", t)
        filler = {"please", "now", "the", "a", "hey", "ok", "okay", "can", "you", "to"}
        rest = [w for w in t.split() if w not in filler]
        return len(rest) <= max_extra_words
