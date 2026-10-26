import os
import tempfile
import unittest

import numpy as np

import _paths  # noqa: F401
from g1voice import config
from g1voice.audio import WavSource, float_to_pcm, resample
from g1voice.vad import Endpointer, EnergyVAD, SileroVAD


class TestConfig(unittest.TestCase):
    def test_env_and_overrides(self):
        path = os.path.join(_paths.ROOT, "config", "settings.yaml")
        cfg = config.load(path, {"vlm": {"max_tokens": 7}},
                          env={"G1V__VAD__THRESHOLD": "0.65", "G1V__STT__BACKEND": "mock"})
        self.assertAlmostEqual(cfg.vad.threshold, 0.65)
        self.assertEqual(cfg.stt.backend, "mock")
        self.assertEqual(cfg.vlm.max_tokens, 7)
        self.assertEqual(cfg.led.colors.listening, [0, 80, 255])

    def test_defaults_are_safe(self):
        cfg = config.load(os.path.join(_paths.ROOT, "config", "settings.yaml"), env={})
        self.assertTrue(cfg.commands.require_wake_word_for_motion)
        self.assertFalse(cfg.privacy.log_transcripts)
        self.assertFalse(cfg.privacy.save_audio)


class TestEndpointer(unittest.TestCase):
    def frames(self, x, n=512):
        pcm = float_to_pcm(x)
        return [(pcm[i:i + 2 * n], i / 32000.0) for i in range(0, len(pcm) - 2 * n + 1, 2 * n)]

    def run_ep(self, x, **kw):
        ep = Endpointer(EnergyVAD(), **kw)
        evs = []
        for pcm, t in self.frames(x):
            evs += ep.process(pcm, t)
        return evs

    def test_utterance_start_and_end_with_lookback(self):
        sr = 16000
        x = np.concatenate([np.zeros(sr), 0.2 * np.sin(np.arange(sr) * 0.1), np.zeros(sr)]).astype(np.float32)
        evs = self.run_ep(x, lookback_ms=300, end_silence_ms=500)
        kinds = [e.kind for e in evs]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds.count("end"), 1)
        self.assertGreater(len(evs[0].pcm), 512 * 2 * 5)          # lookback included
        end = [e for e in evs if e.kind == "end"][0]
        self.assertAlmostEqual(end.speech_end_t, 2.0, delta=0.1)   # last voiced frame
        self.assertGreaterEqual(end.t - end.speech_end_t, 0.45)    # endpoint delay ~ end_silence

    def test_short_blip_is_cancelled(self):
        x = np.concatenate([np.zeros(8000), 0.3 * np.ones(1024), np.zeros(16000)]).astype(np.float32)
        kinds = [e.kind for e in self.run_ep(x, min_speech_ms=250)]
        self.assertIn("cancel", kinds)
        self.assertNotIn("end", kinds)

    def test_max_utterance(self):
        x = (0.2 * np.sin(np.arange(16000 * 4) * 0.1)).astype(np.float32)
        kinds = [e.kind for e in self.run_ep(x, max_utterance_s=1.0)]
        self.assertGreaterEqual(kinds.count("end"), 2)

    def test_rms_floor_skips_model(self):
        calls = []
        ep = Endpointer(lambda x: calls.append(1) or 0.0, rms_floor=0.01)
        ep.process(float_to_pcm(np.zeros(512, np.float32)), 0.0)
        self.assertEqual(calls, [])

    @unittest.skipUnless(os.path.exists(os.path.join(_paths.ROOT, "models", "silero_vad.onnx")), "no Silero model")
    def test_silero_model_runs(self):
        m = SileroVAD(os.path.join(_paths.ROOT, "models", "silero_vad.onnx"))
        p = m(np.zeros(512, np.float32))
        self.assertTrue(0.0 <= p <= 1.0)


class TestAudio(unittest.TestCase):
    def test_wav_source_frames(self):
        src = WavSource(os.path.join(_paths.FIX, "hello.wav"), realtime=False, tail_silence_s=0.1)
        frames = list(src.frames(512))
        self.assertTrue(all(len(f) == 1024 for f, _ in frames))

    def test_wav_source_rejects_wrong_rate(self):
        import wave
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            with wave.open(f.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(22050)
                w.writeframes(b"\0\0" * 100)
            with self.assertRaises(ValueError):
                WavSource(f.name)

    def test_resample_length(self):
        y = resample(np.zeros(22050, np.float32), 22050, 16000)
        self.assertAlmostEqual(len(y), 16000, delta=2)


if __name__ == "__main__":
    unittest.main()
