"""End-to-end orchestrator test: WAV mic -> energy VAD -> mock STT -> mock VLM over HTTP ->
gesture tag -> chunker -> mock TTS -> WAV speaker, with timing metrics."""
import json
import os
import tempfile
import unittest
import wave

import _paths  # noqa: F401
from g1voice import config
from g1voice.actions import NullActions
from g1voice.audio import WavSink, WavSource
from g1voice.metrics import Metrics
from g1voice.pipeline import Orchestrator
from g1voice.robot import NullRobot
from g1voice.stt import MockSTT
from g1voice.tts import MockTTS
from g1voice.vad import Endpointer, EnergyVAD
from g1voice.vlm import VLMClient
from helpers import start_mock_vlm


class FakeCamera:
    def __init__(self):
        self.calls = 0

    def latest_b64(self, max_px, max_age):
        self.calls += 1
        return "ZmFrZQ=="

    def stop(self):
        pass


def run_once(transcript, reply="Hello there! I am fine. [gesture:wave] Nice to meet you.", overrides=None):
    tmp = tempfile.mkdtemp()
    srv, url = start_mock_vlm(reply)
    cfg = config.load(os.path.join(_paths.ROOT, "config", "settings.yaml"), overrides or {}, env={})
    metrics_path = os.path.join(tmp, "m.jsonl")
    out_wav = os.path.join(tmp, "out.wav")
    actions, cam = NullActions(), FakeCamera()
    orch = Orchestrator(
        cfg, source=WavSource(os.path.join(_paths.FIX, "hello.wav"), realtime=False),
        sink=WavSink(out_wav), robot=NullRobot(), stt=MockSTT([transcript]),
        vlm=VLMClient(url, "mock"), tts=MockTTS(0.05), actions=actions, camera=cam,
        metrics=Metrics(metrics_path), endpointer=Endpointer(EnergyVAD(), end_silence_ms=400),
        llm_gestures=["wave"])
    orch.run(drain_timeout_s=10)
    orch.sink.close()
    orch.metrics.close()
    srv.shutdown()
    events = [json.loads(l) for l in open(metrics_path)]
    with wave.open(out_wav) as w:
        frames = w.getnframes()
    return orch, actions, cam, events, frames


class TestPipeline(unittest.TestCase):
    def test_conversation_turn(self):
        orch, actions, cam, events, frames = run_once("Hello robot, how are you?")
        names = [e["event"] for e in events]
        for n in ("speech_start", "speech_end", "stt_final", "vlm_request", "vlm_first_token",
                  "first_chunk_ready", "first_audio", "audio_done", "turn_done"):
            self.assertIn(n, names)
        order = [names.index(n) for n in ("speech_end", "stt_final", "vlm_first_token", "first_audio")]
        self.assertEqual(order, sorted(order))
        self.assertIn(("gesture", "wave"), actions.calls)              # from the [gesture:wave] tag
        self.assertGreater(frames, 16000 * 0.05 * 5)                    # speech was produced
        self.assertEqual(cam.calls, 0)                                  # no vision keyword
        stt = [e for e in events if e["event"] == "stt_final"][0]
        self.assertNotIn("text", stt)                                   # privacy default
        self.assertEqual(orch.turns_done, 1)

    def test_vision_turn_attaches_frame(self):
        orch, actions, cam, events, _ = run_once("What am I holding?")
        self.assertEqual(cam.calls, 1)
        self.assertTrue([e for e in events if e["event"] == "vlm_request"][0]["vision"])
        self.assertIn("image of", " ".join(orch.history[-1]["content"] for _ in [0]))

    def test_command_only_skips_llm(self):
        orch, actions, cam, events, frames = run_once("Robot, wave.")
        names = [e["event"] for e in events]
        self.assertIn(("gesture", "wave"), actions.calls)
        self.assertNotIn("vlm_request", names)
        cmd = [e for e in events if e["event"] == "command_issued"][0]
        self.assertEqual(cmd["origin"], "partial")                      # reflex lane
        self.assertEqual(frames, 0)

    def test_motion_without_wake_word_is_ignored(self):
        orch, actions, cam, events, _ = run_once("wave", reply="Okay.")
        self.assertNotIn(("gesture", "wave"), [c for c in actions.calls])

    def test_stop_cancels(self):
        orch, actions, cam, events, frames = run_once("stop")
        self.assertIn(("stop", ""), actions.calls)
        self.assertNotIn("vlm_request", [e["event"] for e in events])

    def test_unknown_llm_gesture_rejected(self):
        orch, actions, cam, events, _ = run_once("Hello robot", reply="Sure! [gesture:backflip] Done.")
        self.assertNotIn(("gesture", "backflip"), actions.calls)
        self.assertIn("gesture_rejected", [e["event"] for e in events])

    def test_log_transcripts_opt_in(self):
        _, _, _, events, _ = run_once("Hello robot", overrides={"privacy": {"log_transcripts": True}})
        self.assertEqual([e for e in events if e["event"] == "stt_final"][0]["text"], "Hello robot")


class TestReport(unittest.TestCase):
    def test_report_from_metrics(self):
        import report
        _, _, _, events, _ = run_once("Hello robot, how are you?")
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, "e2e.jsonl")
        with open(p, "w") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")
        t = report.turns_from_jsonl([p])
        self.assertEqual(len(t["text"]), 1)
        self.assertGreater(t["text"][0], 0)


if __name__ == "__main__":
    unittest.main()
