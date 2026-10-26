# CLAUDE.md: standing context for every session in this repo

## What this is
Local voice + vision assistant for a Unitree G1 EDU, running on its development computer (PC2):
G1 mic (UDP multicast) -> Silero VAD (CPU) -> STT -> VLM (+ camera frame on demand) -> sentence
chunker -> TTS -> G1 speaker (AudioClient.PlayStream), plus a local-only action server for
allowlisted arm gestures and small lateral steps. Map: README.md; details: docs/.

## Target machine (reported 2026-10-21; re-verify with `make discover`)
- Jetson Orin NX 16 GB, sm_87, on the G1 carrier ("PC2", 192.168.123.164).
- L4T 35.3.1 = JetPack 5.1.1. CUDA 11.4, cuDNN 8.6, TensorRT 8.5.2. Ubuntu 20.04, glibc 2.31.
- System Python 3.8.10 (numpy 1.17.4): do not touch. Host scripts here are stdlib-only for that reason.
- Docker 28.1.1 with the NVIDIA runtime registered (not the default runtime).
- NO internet on the Jetson (no github/pypi/huggingface/ghcr). Everything comes from `make bundle`
  on a workstation, copied over Ethernet.
- G1 mic/speaker are NOT local sound devices. Other programs run on this Jetson: never disturb them.
- Only r35 images run here (r36 do not). Nothing that needs CUDA >= 12 natively.

## Hard constraints (never violate; if a step would, stop and ask the owner)
1. No reflash; no L4T/JetPack/kernel/bootloader changes; no apt upgrade/dist-upgrade; no host
   CUDA/cuDNN/TensorRT changes; no changes to system Python.
2. Everything in containers or under ~/g1-voice. Never install into system paths.
3. Host-wide config (Docker daemon.json default runtime, nvpmodel, jetson_clocks, firewall, routes,
   auto-start) only with explicit approval, showing the exact command and the rollback first.
   Never stop or reconfigure the owner's other programs or containers.
4. No robot motion unless the owner said IN THIS SESSION that the robot is ready, a spotter is
   present and the remote/e-stop is in hand. The action server defaults to dry-run.
5. Never claim something works without a measured run. Log command, config, date and result
   (docs/BENCHMARKS.md, docs/DECISIONS.md).
6. Time-box risky builds (llama.cpp, whisper.cpp, ONNX Runtime) to ~2 h; record and fall back.

## Action-server safety rules (services/action_server/)
- Local-only API (127.0.0.1:8090; serve() refuses other binds). Dry-run unless `--live` AND
  G1_LIVE_CONFIRM=spotter-present-estop-in-hand.
- The LLM never produces joint values or velocities; only allowlisted `[gesture:name]` tags from
  gestures/manifest.yaml (enabled + llm_allowed + clip passes validation against the URDF).
- Gestures: position targets + PD gains on rt/arm_sdk; weight in motor 29, ramped from the
  MEASURED pose, never flipped; ramp-out <= 2 s on finish/stop. Clips: arms only, same start/end
  pose, URDF limits minus 0.05 rad, inside the torso bubble, speed scale <= 1, <= 4 s, one at a
  time, cooldown.
- Steps: SetVelocity(0, vy, 0, <=1 s) re-sent, |vy| <= 0.2, <= 2 s, never continuous; requires
  the operator walking flag (token-protected endpoint), default off.
- Never Damp / ZeroTorque / FSM changes from software (tests/test_action_server.py greps for them).
- Every motion needs a fresh orchestrator heartbeat; watchdog stops motion if it is > 0.5 s old.
- FSM ids are NOT hard-coded: config mode_check.allowed_fsm_ids is empty until the owner confirms
  them on the robot (GET /status shows the current fsm_id/fsm_mode). Docs conflict (200 vs 500).
- Crash marker state/arm_active blocks gestures until `scripts/g1_operator.py release-arms`.

## Completion rule
Measurements are targets, not gates: measure, record PASS/MISS in docs/BENCHMARKS.md, try the
next fallback once within the time box, record the shortfall in docs/DECISIONS.md, move on.
Only a hard-constraint violation, possible robot motion, or a component that cannot run at all
(then stub it behind its interface) may halt progress.

## Working here
- Tests: `make test-local` (anywhere with python3+numpy+pyyaml+requests) or `make test` (app image).
- Interfaces that must stay stable: STT 16 kHz PCM in -> partial/final text; VLM OpenAI-compatible
  streaming chat with base64 images on :8080; TTS text -> 16 kHz int16 PCM chunks.
- Privacy defaults (GDPR): no transcripts, audio or frames persisted (config/settings.yaml privacy).
- Status of each component (built / tested where / measured): README.md "Status".
