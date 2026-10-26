# Architecture

## Processes and ports

| Process | Container / image | Network | Port | GPU | CPU pinning |
|---|---|---|---|---|---|
| Riva speech (ASR+TTS) | `riva-speech:2.14.0-l4t-aarch64`, started by NVIDIA's `riva_start.sh` (patched copy) | published | 50051 | yes | none |
| VLM | `g1voice/vlm:r35` (FROM dustynv/nano_llm r35) | host | 127.0.0.1:8080 | yes | none |
| Orchestrator | `g1voice/app` | host (multicast + DDS) | none | no | `ORCH_CPUSET` (5-6) |
| Action server | `g1voice/app` | host (DDS) | 127.0.0.1:8090 | no | `ACTION_CPUSET` (7) |
| Fallbacks | `g1voice/whisper:r35`, `g1voice/piper`, `g1voice/llamacpp:r35` | host | 8091 / 8092 / 8080 | whisper, llama.cpp | none |

Riva's own script publishes 50051 on all interfaces of PC2 (the robot's internal network).
Everything we start binds to loopback.

## Orchestrator threads (services/orchestrator/g1voice/pipeline.py)

- **Audio thread**: reads 512-sample frames (32 ms) from the mic source. It drops them while the robot
  speaks (half-duplex, plus `half_duplex_tail_ms`). It runs the `Endpointer` (RMS pre-filter →
  Silero → hysteresis → silence endpointing with a 300 ms lookback) and feeds the streaming STT.
  Partial transcripts go to the **command lane** immediately (reflex path).
- **Turn worker**: finishes STT, re-checks commands on the final text, and skips the LLM for
  command-only utterances. It then applies the optional conversation wake word, routes vision
  (keyword router → latest camera frame ≤ 448 px, ≤ 1 s old), and streams the VLM. Tokens pass
  through `GestureTagFilter` (strips `[gesture:x]`, fires allowlisted ones once) and then
  `SentenceChunker` (flush at sentence punctuation, or after 3 words for the first chunk and
  8 afterwards).
- **TTS thread (per turn)**: synthesises chunks while generation continues. It pushes 100 ms PCM
  pieces to `PlayStream` on one stream id per utterance, at most 200 ms ahead of real time.
- **Heartbeat thread**: `POST /heartbeat` to the action server at 10 Hz.

A spoken "stop" (no wake word needed) cancels generation, stops playback, and calls `/stop`.
While a turn is in progress, new utterances only feed the command lane.

## Stable interfaces (swap any component behind them)

| Interface | Contract | Implementations |
|---|---|---|
| STT | `open_stream(on_partial) → feed(pcm16k) … finish() → text` | `RivaSTT` (partials), `WhisperHttpSTT`, `MockSTT` |
| VLM | OpenAI `POST /v1/chat/completions`, `stream: true`, images as `data:image/jpeg;base64` | NanoLLM wrapper, llama-server, mock |
| TTS | `synth(text) → iterator of 16 kHz int16 PCM` | `RivaTTS`, `PiperHttpTTS`, `MockTTS` |
| Audio in / out | `frames(n) → (pcm, t)` / `begin, write, end, stop` | G1 multicast / PlayStream, WAV, null |
| Actions | `POST /gesture`, `/step`, `/stop`, `/heartbeat`, `GET /status` | action server; `NullActions` |

## Safety model (services/action_server/g1action)

1. **Who can ask for motion.** The grammar asks only on transcripts containing the wake word
   (default on), except "stop". The LLM can ask only through tags naming gestures that are
   `enabled`, `llm_allowed` and validated.
2. **What the server will do.** Clips are joint position targets for the 14 arm joints. Waist
   joints are held at the measured pose. The validator checks: arms only, start = end pose, URDF
   limits − 0.05 rad, peak speed ≤ 2.5 rad/s, ≤ 4 s at speed scale ≤ 1, and wrist links inside a
   box in the torso frame (URDF forward kinematics). In live mode a clip without a URDF check is
   refused.
3. **Preconditions for every motion.** A fresh heartbeat (≤ 0.5 s), no other motion running, the
   cooldown elapsed, no crash marker, and the current FSM id in the owner-confirmed allowlist
   (empty by default, so everything is refused). Steps also need the operator `walking` flag
   (token in `state/operator_token`, mode 0600, not mounted into the orchestrator).
4. **How motion is executed.**
   - Gestures: weight 0 → 1 while holding the measured pose, a smoothstep approach to the clip
     start, playback at 50 Hz, then weight 1 → 0 holding the last target (≤ 2 s).
   - Steps: `SetVelocity(0, ±0.15, 0, ≤1 s)` re-sent every 0.3 s for ≤ 2 s, then zero velocity.
     Never continuous.
5. **How motion stops.** `/stop`, a lost heartbeat (watchdog, once per motion), or shutdown
   trigger zero velocity plus an arm ramp-out. There is no damping or zero-torque code path. A
   static test enforces this.
6. **After a crash.** `state/arm_active` exists while the weight may be > 0. On restart, gestures
   stay blocked until an operator runs `release-arms`, which holds the measured pose at weight 1
   and ramps to 0.

## Data and privacy
By default no audio, no frames and no transcripts are written. `logs/metrics.jsonl` holds only
timestamps, word counts and event names. Opt-in switches live under `privacy:` in
config/settings.yaml. Benchmark recordings (`tests/recordings/`) are personal data. Consult the
Data Privacy Officer before recording anyone outside the team.
