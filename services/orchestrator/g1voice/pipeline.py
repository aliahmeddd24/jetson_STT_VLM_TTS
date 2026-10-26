"""Turn state machine: mic -> VAD -> (command lane | conversation lane) -> TTS -> speaker.

Threads:
  audio thread   reads the mic, runs VAD, feeds the STT stream, drives the command lane
  turn worker    finishes STT, routes, streams the VLM, chunks text
  tts thread     (per turn) synthesises chunks and plays them, so speech overlaps generation
Half-duplex: mic frames are dropped while the robot speaks (plus a short tail).
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from typing import Iterable, List, Optional

from .textproc import CommandGrammar, GestureTagFilter, SentenceChunker, has_wake_word, needs_vision
from .vlm import build_messages

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, cfg, *, source, sink, robot, stt, vlm, tts, actions, camera, metrics,
                 endpointer, llm_gestures: Iterable[str]):
        self.cfg = cfg
        self.source, self.sink, self.robot = source, sink, robot
        self.stt, self.vlm, self.tts = stt, vlm, tts
        self.actions, self.camera, self.metrics = actions, camera, metrics
        self.ep = endpointer
        self.llm_gestures = list(llm_gestures)
        c = cfg.commands
        self.grammar = CommandGrammar(c.rules, c.stop_phrases, c.wake_words,
                                      c.require_wake_word_for_motion)
        self._glock = threading.Lock()
        self.turn_q: "queue.Queue" = queue.Queue()
        self.busy = threading.Event()
        self.cancel = threading.Event()
        self.speaking = False
        self.mute_until = 0.0
        self.tail_s = cfg.audio_out.half_duplex_tail_ms / 1000.0
        self.utt = 0
        self.utt_commands: dict = {}
        self.stream = None
        self.history: collections.deque = collections.deque(maxlen=4)
        self.turns_done = 0
        self.stop_event = threading.Event()
        gl = ", ".join("[gesture:%s]" % g for g in self.llm_gestures) or "(none)"
        self.system_prompt = cfg.vlm.system_prompt.replace("{gestures}", gl)

    # ------------------------------------------------------------ helpers
    def _led(self, state: str) -> None:
        if self.cfg.led.enabled:
            rgb = self.cfg.led.colors.get(state)
            if rgb is not None:
                self.robot.led(rgb)

    def _privacy_text(self, text: str) -> dict:
        return {"text": text} if self.cfg.privacy.log_transcripts else {"words": len(text.split())}

    def _muted(self) -> bool:
        return self.speaking or time.monotonic() < self.mute_until

    # ------------------------------------------------------------ command lane
    def _exec_command(self, action: str, arg: str, turn: int, origin: str) -> None:
        t = time.monotonic()
        self.utt_commands.setdefault(turn, []).append((action, arg))
        if action == "stop":
            self.cancel.set()
            self.sink.stop()
            res = self.actions.stop()
        elif action == "gesture":
            res = self.actions.gesture(arg)
        elif action == "step":
            res = self.actions.step(arg)
        else:
            log.warning("unknown action %s", action)
            return
        ok = bool(res.get("ok"))
        self.metrics.event(turn, "command_issued", t=t, action=action, arg=arg, origin=origin,
                           ok=ok, reason=res.get("error") or res.get("reason"))
        if action != "stop":
            self._led("moving" if ok else "rejected")

    def _on_partial(self, text: str, utt: int) -> None:
        if not self.cfg.commands.enabled or not self.cfg.commands.use_partials:
            return
        with self._glock:
            if utt != self.utt:
                return
            cmds = self.grammar.match(text)
        for action, arg in cmds:
            self._exec_command(action, arg, utt, "partial")

    # ------------------------------------------------------------ audio thread
    def run(self, max_turns: Optional[int] = None, drain_timeout_s: float = 60.0) -> None:
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        self._led("idle")
        frame = int(self.cfg.vad.chunk_samples)
        try:
            for pcm, t in self.source.frames(frame):
                if self.stop_event.is_set() or (max_turns and self.turns_done >= max_turns):
                    break
                if self._muted():
                    if self.ep.in_speech:
                        self._abort_utterance()
                    continue
                for ev in self.ep.process(pcm, t):
                    self._on_vad(ev)
        finally:
            deadline = time.monotonic() + drain_timeout_s
            while (self.busy.is_set() or not self.turn_q.empty()) and time.monotonic() < deadline:
                time.sleep(0.05)
            self.turn_q.put(None)
            worker.join(timeout=5)

    def _abort_utterance(self) -> None:
        if self.stream is not None:
            self.stream.abort()
            self.stream = None
        self.ep.reset()

    def _on_vad(self, ev) -> None:
        if ev.kind == "start":
            with self._glock:
                self.utt += 1
                self.grammar.reset()
                utt = self.utt
            self.stream = self.stt.open_stream(on_partial=lambda txt, u=utt: self._on_partial(txt, u))
            self.stream.feed(ev.pcm)
            self.metrics.event(utt, "speech_start", t=ev.t)
            if not self.busy.is_set():
                self._led("listening")
        elif ev.kind == "audio" and self.stream is not None:
            self.stream.feed(ev.pcm)
        elif ev.kind == "cancel":
            if self.stream is not None:
                self.stream.abort()
            self.stream = None
            self.metrics.event(self.utt, "speech_cancel", t=ev.t)
            if not self.busy.is_set():
                self._led("idle")
        elif ev.kind == "end" and self.stream is not None:
            item = (self.stream, self.utt, ev.speech_end_t, ev.t)
            self.stream = None
            if self.busy.is_set():
                # A turn is in progress: this utterance only feeds the command lane.
                threading.Thread(target=self._command_only, args=item, daemon=True).start()
            else:
                self.busy.set()
                self.turn_q.put(item)

    def _command_only(self, stream, utt, speech_end_t, detected_t) -> None:
        try:
            text = stream.finish()
        except Exception as exc:
            log.warning("STT (command-only) failed: %s", exc)
            return
        with self._glock:
            cmds = self.grammar.match(text) if utt == self.utt else []
        for action, arg in cmds:
            self._exec_command(action, arg, utt, "final")

    # ------------------------------------------------------------ turn worker
    def _worker(self) -> None:
        while True:
            item = self.turn_q.get()
            if item is None:
                return
            try:
                self._turn(*item)
            except Exception as exc:
                log.exception("turn failed")
                self.metrics.event(item[1], "turn_error", error="%s: %s" % (type(exc).__name__, exc))
            finally:
                self.turns_done += 1
                self.busy.clear()
                self._led("idle")

    def _turn(self, stream, turn: int, speech_end_t: float, detected_t: float) -> None:
        m = self.metrics
        self.cancel.clear()
        m.event(turn, "speech_end", t=speech_end_t)
        m.event(turn, "vad_endpoint", t=detected_t)
        self._led("thinking")
        try:
            text = stream.finish()
        except Exception as exc:
            m.event(turn, "stt_error", error=str(exc))
            return
        m.event(turn, "stt_final", **self._privacy_text(text))
        if not text:
            return

        if self.cfg.commands.enabled:
            with self._glock:
                cmds = self.grammar.match(text) if turn == self.utt else []
            for action, arg in cmds:
                self._exec_command(action, arg, turn, "final")
        if self.utt_commands.get(turn) and self.grammar.is_command_only(text):
            m.event(turn, "turn_done", reason="command_only")
            return
        conv = self.cfg.conversation
        if conv.require_wake_word and not has_wake_word(text, conv.wake_words):
            m.event(turn, "turn_done", reason="no_wake_word")
            return
        if self.cancel.is_set():
            m.event(turn, "turn_done", reason="cancelled")
            return

        image = None
        if needs_vision(text, self.cfg.router.vision_keywords) and self.camera is not None:
            image = self.camera.latest_b64(self.cfg.vlm.max_image_px, self.cfg.camera.max_age_s)
            m.event(turn, "frame", attached=image is not None)
        messages = build_messages(self.system_prompt, list(self.history), text, image)
        m.event(turn, "vlm_request", vision=image is not None)
        reply = self._speak_stream(turn, messages)
        if reply:
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": reply})
        m.event(turn, "turn_done", reason="ok" if reply else "empty")

    def _speak_stream(self, turn: int, messages: List[dict]) -> str:
        m = self.metrics
        q: "queue.Queue" = queue.Queue()
        tts_thread = threading.Thread(target=self._tts_loop, args=(turn, q), daemon=True)
        tts_thread.start()
        gf = GestureTagFilter(self.llm_gestures,
                              on_gesture=lambda g: self._exec_command("gesture", g, turn, "llm"),
                              on_rejected=lambda g: m.event(turn, "gesture_rejected", name=g))
        ch = SentenceChunker(self.cfg.chunker.first_chunk_words, self.cfg.chunker.later_chunk_words)
        spoken: List[str] = []
        n_deltas = 0

        def push(chunks):
            for c in chunks:
                if n_deltas and not spoken:
                    m.event(turn, "first_chunk_ready", words=len(c.split()))
                spoken.append(c)
                q.put(c)

        try:
            for delta in self.vlm.stream(messages, self.cancel):
                if n_deltas == 0:
                    m.event(turn, "vlm_first_token")
                n_deltas += 1
                push(ch.feed(gf.feed(delta)))
                if self.cancel.is_set():
                    break
            push(ch.feed(gf.flush()))
            push(ch.flush())
            m.event(turn, "vlm_done", deltas=n_deltas)
        except Exception as exc:
            m.event(turn, "vlm_error", error=str(exc))
            if not spoken:
                q.put("Sorry, I could not think of an answer.")
        finally:
            q.put(None)
            tts_thread.join()
        return " ".join(spoken)

    def _tts_loop(self, turn: int, q: "queue.Queue") -> None:
        started = False
        first_audio = True
        try:
            while True:
                text = q.get()
                if text is None:
                    break
                if self.cancel.is_set():
                    continue
                self.metrics.event(turn, "tts_request", words=len(text.split()))
                try:
                    for pcm in self.tts.synth(text):
                        if self.cancel.is_set():
                            break
                        if not started:
                            self.speaking = True
                            self.sink.begin("%d-%d" % (turn, int(time.time() * 1000)))
                            self._led("speaking")
                            started = True
                        if first_audio:
                            self.metrics.event(turn, "first_audio")
                            first_audio = False
                        self.sink.write(pcm)
                except Exception as exc:
                    self.metrics.event(turn, "tts_error", error=str(exc))
            if started and not self.cancel.is_set():
                self.sink.end()
                self.metrics.event(turn, "audio_done")
        finally:
            self.speaking = False
            self.mute_until = time.monotonic() + self.tail_s
