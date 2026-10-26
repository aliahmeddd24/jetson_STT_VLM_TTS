# Decisions

Format: what was decided, why, and how to undo it. The newest entries go at the bottom. Entries
D1–D20 were made while writing the repo in a development sandbox (x86, no GPU, no Docker, no
network, no robot), on 2026-10-26. Nothing below has been run on the Jetson yet.

## Build and packaging
- **D1. Everything ships as images built on the workstation.** The Jetson has no internet, and the
  owner forbids host installs. Our images (`app`, `piper`, `vlm`, `builder`) are built for
  linux/arm64 under QEMU on the workstation and saved as tarballs. The vlm image only adds a COPY on
  top of dustynv/nano_llm, so it needs no emulation.
- **D2. CUDA compilation happens on the Jetson, offline.** whisper.cpp and llama.cpp are built with
  `docker build --network=none` from a `builder` image (l4t-jetpack r35.4.1 + gcc-10 + pip cmake)
  that was itself prepared on the workstation. libcuda is linked from the toolkit stubs, and
  `GGML_CUDA_NO_VMM=ON` is set because VMM is not expected on L4T 35 / CUDA 11.4 (VERIFY). Each
  build is time-boxed to 2 h.
- **D3. One Python image for the orchestrator and the action server.** The HTTP servers use the
  standard library only, so there is no web-framework dependency. Two containers keep them
  isolated at run time. Piper is a separate build target so a piper-phonemize wheel problem cannot
  break the main image.
- **D4. Host-side scripts are stdlib-only and Python 3.8.** These are mic capture, recording, the
  operator CLI, the report and the memory report. `tests/test_compat.py` enforces Python 3.8
  syntax and that no script shadows a stdlib module.
- **D5. Source refs default to branch heads, and the resolved SHAs are recorded** in
  `offline-bundle/BUNDLE_MANIFEST.md`. I could not check which llama.cpp/whisper.cpp revisions
  still build with CUDA 11.4. If the head fails, pin an older tag via `LLAMA_REF` / `WHISPER_REF`
  and rebuild.

## Riva
- **D6. Docker's default runtime is not changed.** `riva.sh prepare` patches a private copy of the
  quickstart: ASR+TTS en-US only, Orin platform, and `--runtime nvidia` added to every `docker run`.
  The daemon.json change is documented as an [approval] fallback (RUNBOOK G), with the warning
  that restarting Docker restarts the owner's containers.
- **D7. Offline Riva models (UNVERIFIED approach).** `riva_models.sh` runs `riva_init.sh` on the
  workstation with stub `docker`, `ngc` and `uname` on PATH, captures the model names it asks for,
  and downloads them with the ngc CLI. On the Jetson, `riva.sh deploy-offline` runs `riva-deploy`
  per RMIR. If either step does not match the 2.14.0 quickstart's internals, use temporary
  connection sharing (RUNBOOK F) with the stock `riva_init.sh`.

## VLM
- **D8. nano_llm image tag.** The bundle tries `NANO_LLM_IMAGE` and then several r35 tags, and
  records the one that pulls. Whether VILA1.5-3B runs in an r35 NanoLLM image is the riskiest open
  question (NanoLLM's VLM docs list only JetPack 6). It is measured first in Phase 2.
- **D9. Stateless VLM server.** The orchestrator sends the last 2 exchanges as text history. The
  image goes only on the current user message. One request at a time on the GPU.

## Orchestrator behaviour
- **D10. A wake word is required for motion by default** (`commands.require_wake_word_for_motion:
  true`, wake word "robot"). This is a deliberate tightening beyond the original spec: overheard
  speech in a demo room should not move the arms. "Stop" never needs it. Set it to false to restore
  the original behaviour.
- **D11. Half-duplex: the mic is ignored while speaking.** Consequence: a spoken "stop" is not
  heard while the robot is talking. It is heard during thinking and between turns. The remote,
  `scripts/g1_operator.py stop` and the watchdog always work. Barge-in with a stop-word detector
  needs an echo measurement on the robot first (RUNBOOK C).
- **D12. Command-only utterances skip the LLM** ("robot, wave"). Longer utterances run both lanes.
- **D13. The endpointing window dominates STT latency.** Latency is measured from the last voiced
  frame, so `vad.end_silence_ms` (500 ms) is part of every end-to-end number. The sandbox mock run
  showed speech_end → final transcript ≈ 0.51 s with an instant mock STT. Tune 300–400 ms against
  false endpoints on real recordings.
- **D14. Privacy defaults.** No transcripts, audio or frames are written by default. Only timings
  and word counts are logged.

## Action server
- **D15. FSM ids are not hard-coded.** `allowed_fsm_ids` is empty, so every motion is refused until
  the owner reads the ids on the robot (`status` shows `fsm_id`/`fsm_mode`) and confirms them.
- **D16. Mode check via LocoClient RPC (GetFsmId/GetFsmMode, API 7001/7002), not the
  `rt/sportmodestate` topic.** The spec named the topic, but I could not confirm its G1 message
  type and fields offline. The RPC ids come from unitree_sdk2's `g1/loco` headers (VERIFY). This is
  a swap point inside `UnitreeBackend.fsm()`.
- **D17. Gesture execution details.** Waist joints (12–14) are held at their measured pose with the
  same gains. kp=60 and kd=1.5 follow Unitree's arm_sdk example (VERIFY for your firmware). The
  clip is refused if the arms are more than 0.35 rad from its start pose.
- **D18. Python arm player.** The control loop runs at 50 Hz with tick-error statistics in
  `/status` (`arm_jitter_ms`). If p99 exceeds about 5 ms with the AI load running, move the player
  to C++ (unitree_sdk2) behind the same backend interface. CPU pinning to core 7 is the first
  mitigation.
- **D19. Crash recovery is manual.** A marker file while the weight may be > 0 means gestures are
  blocked after a crash until an operator runs `release-arms`. Nothing moves automatically on
  restart.
- **D20. Gesture clips do not ship.** The manifest lists `wave` and `nod_arms` disabled. Test clips
  exist only as synthetic fixtures. The avp_teleoperate format used by the converter (frames under
  `data`, `{actions|states}.{left_arm|right_arm}.qpos`) is an assumption until checked on a real
  episode.

## Bugs found by the local test-suite (fixed before delivery)
- Config key mismatch (`chunker.first_words` vs `first_chunk_words`): every conversation turn would
  have failed. Turn failures are now also written to the metrics log as `turn_error`.
- `Metrics.event(name=...)` collided with its own parameter: rejected-gesture events crashed the
  VLM stream. Its parameters are now positional-only.
- Clip validator crashed on clips containing non-arm joints, instead of rejecting them.
- `scripts/operator.py` shadowed Python's `operator` module, which would have crashed the operator
  CLI on the Jetson. It was renamed to `g1_operator.py`.
- The action-server watchdog re-sent stop every 50 ms during ramp-out. It now fires once per motion.
- The runbook's live procedure would have collided with the dry-run server on port 8090. It now
  uses the `compose.live.yaml` overlay (`make live` / `make dry`).

## Open items to verify on the robot (also tagged VERIFY in code/config)
Mic multicast format · PlayStream argument type (bytes vs list; the adapter tries both) ·
interface name · FSM ids/modes and API ids 7001/7002 · arm_sdk gains · URDF filename for your G1
revision · Riva voice name · nano_llm r35 tag and VILA1.5 support · Riva offline model path ·
camera source (zmq image server, http or v4l2) and its endpoint · avp_teleoperate JSON layout.
