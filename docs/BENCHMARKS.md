# Benchmarks

**Nothing in this file has been measured on the target yet.** Targets are the owner's, are
non-blocking, and are reported as PASS / MISS. Generate rows with `make report` (reads `logs/`),
and record the date, config and image versions with each run.

## Results

| Metric | Target | Result | Status | Date / config / command |
|---|---|---|---|---|
| GPU sanity (sm_87, CUDA in r35 container) | runs | | NOT MEASURED | `make gpu-sanity` |
| VLM first token, 1 image ≤ 448 px (p50) | ≤ 2.5 s | | NOT MEASURED | `make bench-vlm` |
| VLM decode | ≥ 15 tok/s | | NOT MEASURED | `make bench-vlm` |
| VLM memory | ≤ 5 GB | | NOT MEASURED | tegrastats during bench-vlm |
| STT end of speech → final, ~3 s utterance (p50) | ≤ 0.7 s | | NOT MEASURED | `make bench-stt` |
| STT word error rate on ~20 G1-mic recordings | sanity | | NOT MEASURED | `make bench-stt` |
| TTS first audio, 3–8 words (p50) | ≤ 0.7 s | | NOT MEASURED | `make bench-tts` |
| TTS real-time factor | < 0.5 | | NOT MEASURED | `make bench-tts` |
| Reflex: end of speech → motion command (p50, 20 turns) | ≤ 0.8 s | | NOT MEASURED | `make bench-e2e` |
| Text turn: end of speech → first audio (p50) | ≤ 2.0 s | | NOT MEASURED | `make bench-e2e` |
| Vision turn: end of speech → first audio (p50) | ≤ 3.5 s | | NOT MEASURED | `make bench-e2e` |
| Free headroom with all services + owner's programs | ≥ 1.5 GB | | NOT MEASURED | `make soak` |
| Steady-state swap | < 500 MB | | NOT MEASURED | `make soak` |
| 30-min soak: temps, errors | stable | | NOT MEASURED | `make soak` |
| Arm control loop jitter (p99) | < ~5 ms at 50 Hz | | NOT MEASURED | `GET /status` → `arm_jitter_ms` |

## How latency is defined
All end-to-end numbers start at **`speech_end`**: the last voiced VAD frame, i.e. when the person
actually stopped talking. So they include the endpointing silence window (`vad.end_silence_ms`,
default 500 ms). "First audio" is when the first PCM chunk is handed to `PlayStream`. The G1's own
playback buffering comes on top of that; estimate it once with a phone recording if needed.
Reflex values can be negative when a partial transcript triggers the command before speech ends.

## Development-sandbox smoke run (NOT representative)
x86 container with no GPU, mock STT/TTS and an HTTP mock VLM (0.2 s artificial first token), using
`tests/fixtures/hello.wav`: text turn speech_end → first audio 0.75 s, of which about 0.5 s is the
VAD silence window. This shows the plumbing and timing instrumentation work. It says nothing about
the real stack.
