"""Entry point: python -m g1voice.main --config config/settings.yaml [options]"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys

import yaml

from . import config as config_mod
from .audio import G1MulticastMic, G1SpeakerSink, NullSink, NullSource, WavSink, WavSource, iface_ipv4
from .camera import FrameBuffer
from .metrics import Metrics
from .pipeline import Orchestrator
from .robot import G1Audio, NullRobot
from .stt import make_stt
from .tts import make_tts
from .vad import Endpointer, EnergyVAD, SileroVAD
from .vlm import make_vlm

log = logging.getLogger("g1voice")


def load_llm_gestures(manifest_path: str):
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return [name for name, g in (data.get("gestures") or {}).items()
            if g and g.get("enabled") and g.get("llm_allowed")]


def build(cfg, args):
    needs_robot = cfg.audio_in.source == "g1_multicast" or cfg.audio_out.sink == "g1_speaker"
    robot = NullRobot()
    if needs_robot:
        robot = G1Audio(cfg.g1.dds_domain, cfg.g1.network_interface)
        if cfg.audio_out.volume is not None:
            robot.set_volume(cfg.audio_out.volume)

    src = cfg.audio_in.source
    if src == "g1_multicast":
        ip = cfg.g1.local_ip or iface_ipv4(cfg.g1.network_interface)
        source = G1MulticastMic(cfg.audio_in.multicast_group, cfg.audio_in.multicast_port, ip)
    elif src == "wav":
        source = WavSource(cfg.audio_in.wav_path, cfg.audio_in.wav_realtime)
    else:
        source = NullSource()

    kw = dict(chunk_ms=cfg.audio_out.chunk_ms, lead_ms=cfg.audio_out.lead_ms)
    if cfg.audio_out.sink == "g1_speaker":
        sink = G1SpeakerSink(robot, cfg.audio_out.app_name, **kw)
    elif cfg.audio_out.sink == "wav":
        sink = WavSink(cfg.audio_out.wav_path, paced=False, **kw)
    else:
        sink = NullSink(**kw)

    if cfg.vad.backend == "energy":
        model = EnergyVAD()
    elif os.path.exists(cfg.vad.model_path):
        model = SileroVAD(cfg.vad.model_path)
    else:
        log.warning("Silero model %s missing: using energy VAD (much worse)", cfg.vad.model_path)
        model = EnergyVAD()
    v = cfg.vad
    endpointer = Endpointer(model, threshold=v.threshold, rms_floor=v.rms_floor,
                            chunk_samples=v.chunk_samples, lookback_ms=v.lookback_ms,
                            min_speech_ms=v.min_speech_ms, end_silence_ms=v.end_silence_ms,
                            max_utterance_s=v.max_utterance_s)

    if args.no_actions:
        from .actions import NullActions
        actions = NullActions()
    else:
        from .actions import ActionClient
        actions = ActionClient(cfg.action_server.url, cfg.action_server.heartbeat_hz)
        actions.start_heartbeat()

    c = cfg.camera
    camera = FrameBuffer(c.source, zmq_endpoint=c.zmq_endpoint, http_url=c.http_url,
                         v4l2_device=c.v4l2_device).start()

    return Orchestrator(cfg, source=source, sink=sink, robot=robot, stt=make_stt(cfg),
                        vlm=make_vlm(cfg), tts=make_tts(cfg), actions=actions, camera=camera,
                        metrics=Metrics(cfg.metrics.path), endpointer=endpointer,
                        llm_gestures=load_llm_gestures(args.manifest))


def wait_for_services(cfg, timeout_s: float) -> None:
    """Block until Riva's gRPC port and the VLM health endpoint answer (or timeout)."""
    import socket
    import time
    import urllib.request
    from urllib.parse import urlparse

    checks = []
    if cfg.stt.backend == "riva" or cfg.tts.backend == "riva":
        host, port = cfg.stt.riva_uri.rsplit(":", 1)
        checks.append(("riva", lambda: socket.create_connection((host, int(port)), 1).close()))
    if cfg.vlm.backend == "openai":
        u = urlparse(cfg.vlm.url)
        health = "%s://%s/health" % (u.scheme, u.netloc)
        checks.append(("vlm", lambda: urllib.request.urlopen(health, timeout=2).read()))
    deadline = time.monotonic() + timeout_s
    for name, fn in checks:
        while True:
            try:
                fn()
                log.info("%s is up", name)
                break
            except Exception as exc:
                if time.monotonic() > deadline:
                    raise RuntimeError("%s not reachable after %.0f s: %s" % (name, timeout_s, exc))
                time.sleep(2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/settings.yaml")
    p.add_argument("--manifest", default="gestures/manifest.yaml")
    p.add_argument("--wav", help="use a WAV file as the microphone")
    p.add_argument("--out-wav", help="write speech to a WAV file instead of the G1 speaker")
    p.add_argument("--mock", action="store_true", help="mock STT, VLM and TTS")
    p.add_argument("--no-actions", action="store_true", help="do not contact the action server")
    p.add_argument("--max-turns", type=int, default=None)
    p.add_argument("--wait", type=float, default=0, help="seconds to wait for Riva/VLM at startup")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    over = {}
    if args.wav:
        over.setdefault("audio_in", {}).update(source="wav", wav_path=args.wav)
    if args.out_wav:
        over.setdefault("audio_out", {}).update(sink="wav", wav_path=args.out_wav)
    if args.mock:
        over.update(stt={"backend": "mock"}, vlm={"backend": "mock"}, tts={"backend": "mock"},
                    vad={"backend": "energy"})
    cfg = config_mod.load(args.config, over)

    if args.wait:
        wait_for_services(cfg, args.wait)
    orch = build(cfg, args)
    signal.signal(signal.SIGTERM, lambda *_: orch.stop_event.set())
    try:
        orch.run(max_turns=args.max_turns)
    except KeyboardInterrupt:
        orch.stop_event.set()
    finally:
        if hasattr(orch.sink, "close"):
            orch.sink.close()
        orch.actions.close()
        orch.camera.stop()
        orch.metrics.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
