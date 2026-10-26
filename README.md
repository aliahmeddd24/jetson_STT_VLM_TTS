# g1-voice

Fully local, low-latency voice and vision assistant for a **Unitree G1 EDU** on its Jetson Orin NX
16 GB (JetPack 5.1.1), plus a safe, dry-run-by-default action server for gestures and small steps.

```
G1 mic (UDP multicast) ─► orchestrator ─► Silero VAD (CPU)
                              ├─► command lane: partial transcripts ─► grammar ─► action server (127.0.0.1:8090)
                              └─► conversation: final STT ─► router ─► VLM (+frame) ─► chunker ─► TTS ─► G1 PlayStream
G1 camera ─► latest-frame buffer (attached only when the question needs sight)
```

| Stage | Default | GPU | Fallback |
|---|---|---|---|
| STT + TTS | NVIDIA Riva embedded 2.14.0 (gRPC :50051) | yes | whisper.cpp CUDA (`whisper` profile) / Piper (`piper` profile) |
| VLM | NanoLLM + VILA1.5-3B, OpenAI API on :8080 | yes | llama.cpp + Qwen2.5-VL-3B GGUF (`llamacpp` profile) |
| VAD, router, grammar, chunker | orchestrator container | no | |
| Gestures / steps | action server container, host network, own CPU core | no | |

## Workflow

1. **Workstation (internet):** `cp .env.example .env`, export `NGC_API_KEY` (for Riva), then
   `make bundle`. This clones pinned sources, downloads models, builds and saves arm64 images into
   `offline-bundle/`. Budget roughly 60–100 GB of disk and a few hours (QEMU builds).
2. **Copy** the whole directory over Ethernet: `rsync -a --exclude .venv-ws ./ unitree@192.168.123.164:~/g1-voice/`
3. **Jetson (offline):** `make install`, then follow **docs/RUNBOOK.md** phase by phase
   (`make discover`, `make gpu-sanity`, `make mic-capture`, `make riva-deploy`, `make up`, benchmarks).

`make help` lists every target. No step installs anything on the Jetson host.

## Status (be precise about what has and has not been verified)

| Component | Code | Tested in development sandbox (x86, no GPU, no robot) | Verified on the Jetson / robot |
|---|---|---|---|
| Orchestrator (VAD, routing, grammar, chunker, gesture tags, half-duplex, metrics) | complete | yes: unit + end-to-end with mock STT/TTS and HTTP mock VLM | **not yet** |
| G1 mic multicast / PlayStream / LED adapter | complete | no (needs robot) | **not yet**: `make mic-capture`, `scripts/speaker_test.py` |
| Riva STT/TTS clients | complete | no (needs Riva) | **not yet** |
| VLM OpenAI server (NanoLLM backend) | complete | mock backend yes; NanoLLM no | **not yet**: riskiest piece |
| Action server (dry-run backend, safety logic, HTTP, watchdog) | complete | yes: 25 safety tests | **not yet** |
| Unitree backend (arm_sdk, LocoClient) | complete | no (needs robot) | **not yet** |
| Clip converter / validator (URDF FK) | complete | yes, on synthetic data and a toy URDF | needs a real clip + real URDF |
| Offline bundle / install scripts | complete | syntax only (no Docker in sandbox) | **not yet** |
| Benchmarks / report | complete | report logic yes | numbers: **none measured yet** |

All targets in docs/BENCHMARKS.md are currently `NOT MEASURED`.

## Docs
CLAUDE.md (constraints for future sessions) · docs/ARCHITECTURE.md · docs/RUNBOOK.md ·
docs/DECISIONS.md · docs/BENCHMARKS.md · docs/ENVIRONMENT.md · NOTICE (attribution)
