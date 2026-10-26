# Runbook

Work through the phases in order. Each phase logs what it measured. A MISS never blocks the next
phase (see the completion rule in CLAUDE.md). Commands marked **[approval]** change something
host-wide: run them only with the owner's explicit OK, and keep the rollback at hand.

## A. Workstation (has internet)

Prerequisites: Docker 24+ (BuildKit), git, curl, python3 with venv, 60–100 GB free disk. For Riva,
an NGC account, an API key and the `ngc` CLI. Never paste the key into files or chats: export it
in the shell only.

```bash
cp .env.example .env            # adjust G1_IFACE later on the Jetson
export NGC_API_KEY=...          # shell only
make bundle                     # = scripts/offline/fetch_all.sh
```
The script prints a FAILED list at the end and writes `offline-bundle/BUNDLE_MANIFEST.md` (source
SHAs, image tags, model files). Typical failures and what to do:
- **No nano_llm r35 tag pulled.** Look up the tags on hub.docker.com/r/dustynv/nano_llm, set
  `NANO_LLM_IMAGE` in .env, and re-run with `SKIP_MODELS=1`.
- **Riva RMIR capture empty.** Plan for section F (temporary connection sharing) for `riva_init`,
  or use the Piper/whisper fallbacks.
- **piper image failed.** It only affects the TTS fallback.

Copy everything (only the robot's Ethernet is needed):
```bash
rsync -a --info=progress2 --exclude .venv-ws ./ unitree@192.168.123.164:~/g1-voice/
```

## B. Jetson, phases 0–1 (read-only, then GPU sanity)

```bash
cd ~/g1-voice
make install          # docker load + model check; no host changes
make discover         # Phase 0: read-only; output in logs/environment-*.md
```
Copy the findings into docs/ENVIRONMENT.md. Set `G1_IFACE` in `.env` to the 192.168.123.x
interface (both containers read it). Check that ports 8080, 8090, 8091, 8092 and 50051 are free.

```bash
make gpu-sanity       # Phase 1: expects sm_87, prints fp16 TFLOP/s, PASS
```

## C. G1 audio I/O (verify the assumptions before building on them)

```bash
make mic-capture      # 5 s from 239.168.123.161:5555 -> logs/g1_mic.wav
```
The script reports the byte rate (expected about 32,000 B/s for 16 kHz mono int16). Copy the WAV
to a laptop and listen. If pitch or speed is wrong, stop: the format assumption is wrong.
```bash
docker compose run --rm --no-deps orchestrator python /app/scripts/speaker_test.py --iface $G1_IFACE
```
This cycles the head LED and plays a 1.5 s tone through `PlayStream`. Then test whether the robot
hears itself: run `make mic-capture` while the speaker test plays, and listen. If it does, keep
half-duplex (default) and consider raising `half_duplex_tail_ms`.

## D. Phase 3: Riva (STT + TTS)

```bash
make riva-prepare     # patched private copy in third_party/riva_qs (ASR+TTS en-US, orin, --runtime nvidia)
make riva-deploy      # builds TensorRT engines from bundled RMIRs (offline; can take 30+ min)
make riva-start && bash services/riva/riva.sh status
```
If `riva-deploy` has no RMIRs or fails, you have two options. One is section F plus
`bash services/riva/riva.sh init`. The other is to use the fallbacks (section H). If
`riva_start.sh` still insists on Docker's default runtime, see section G **[approval]**.

Check the voice name: `docker exec riva-speech ls /data/models | grep -i tts`. Set `tts.voice` if
it is not `English-US.Female-1`. Then measure:
```bash
make bench-tts
python3 scripts/record_utterances.py --iface $G1_IFACE   # ~20 prompted WAVs from the G1 mic (consenting team members only)
make bench-stt
```

## E. Phase 2: VLM

```bash
bash scripts/tegrastats_log.sh logs/teg-vlm.log &        # memory while loading and benchmarking
COMPOSE_PROFILES=nanollm docker compose up -d vlm
docker compose logs -f vlm
```
The first start converts VILA1.5-3B to MLC q4f16_ft and builds the TensorRT vision engine. Both
are cached in `models/nano_llm_data` and take a long time once. Wait for `listening on
127.0.0.1:8080`, then:
```bash
make bench-vlm
kill %1; python3 scripts/bench/mem_report.py logs/teg-vlm.log
```
If NanoLLM does not run on L4T 35.3.1, record the error in DECISIONS.md and try these in order:
1. `--vision-api hf` (edit the `command` in compose.yaml).
2. A smaller `--max-context-len`.
3. The llama.cpp fallback (section H).

## F. Temporary internet for the Jetson via the workstation [approval]

Use this only if hand-copying models is impractical (Riva `init`, NanoLLM first run needing an
extra file). The example assumes the workstation is 192.168.123.222 on the robot LAN with
internet on `wlan0`.

Workstation:
```bash
sudo sysctl -w net.ipv4.ip_forward=1
sudo iptables -t nat -A POSTROUTING -s 192.168.123.0/24 -o wlan0 -j MASQUERADE
```
Jetson (temporary, lost at reboot). Record `ip route` first:
```bash
sudo ip route add default via 192.168.123.222 dev $G1_IFACE metric 50
sudo resolvectl dns $G1_IFACE 1.1.1.1 2>/dev/null || echo "DNS: check /etc/resolv.conf handling first"
```
Rollback:
```bash
sudo ip route del default via 192.168.123.222 dev $G1_IFACE metric 50; sudo resolvectl revert $G1_IFACE   # Jetson
sudo iptables -t nat -D POSTROUTING -s 192.168.123.0/24 -o wlan0 -j MASQUERADE; sudo sysctl -w net.ipv4.ip_forward=0   # workstation
```

## G. Docker default runtime = nvidia (only if Riva requires it) [approval]

**Warning:** restarting Docker restarts every container, including the owner's other programs.
Schedule it.
```bash
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak-g1voice
sudo python3 - <<'PY'
import json; p="/etc/docker/daemon.json"; d=json.load(open(p)); d["default-runtime"]="nvidia"; json.dump(d, open(p,"w"), indent=2)
PY
sudo systemctl restart docker
```
Rollback: `sudo cp /etc/docker/daemon.json.bak-g1voice /etc/docker/daemon.json && sudo systemctl restart docker`.

## H. Fallbacks

| Swap | Do |
|---|---|
| STT → whisper.cpp | `make build-native` (time-boxed 2 h), add `whisper` to `COMPOSE_PROFILES`, set `stt.backend: whisper_http` |
| TTS → Piper (CPU) | add `piper` to `COMPOSE_PROFILES`, set `tts.backend: piper_http` |
| VLM → llama.cpp | `make build-native`, set `COMPOSE_PROFILES=llamacpp`, `vlm.model` any string; keep images ≤ 448 px |
| Anything → mock | `make up-mock` (no GPU, no robot): verifies plumbing only |

Record every swap in DECISIONS.md with the reason and the measured numbers.

## I. Phases 4–6: run it

```bash
make up                  # Riva + VLM + action server (DRY-RUN) + orchestrator
make logs
make bench-e2e           # replay the recordings through the full stack
make report              # paste into docs/BENCHMARKS.md
make soak                # 30 min, tegrastats, latency report
```
Talk to the robot. The head LED shows the state: blue listening, amber thinking, green speaking,
magenta moving, red rejected.

## J. Gestures

1. Record the motion with avp_teleoperate, starting and ending in the standing arm pose. Copy the
   episode JSON to the Jetson.
2. Convert and validate it inside the app image (it needs numpy and pyyaml, which the host
   Python must not get). Put the episode in `gestures/`, then:
   ```bash
   docker compose run --rm --no-deps -v "$PWD/gestures:/work/gestures" orchestrator \
     python /app/scripts/convert_avp_clip.py gestures/episode.json --name wave --out gestures/clips/wave.json
   docker compose run --rm --no-deps orchestrator python /app/scripts/validate_clip.py
   ```
   Fix any PROBLEM lines by trimming (`--start/--end`), smoothing, or `--clamp`. The **first real
   episode** also verifies the converter's format assumptions: compare the plotted joints with
   what was recorded.
3. Set `enabled: true` in gestures/manifest.yaml, then `docker compose restart action_server`.
   `python3 scripts/g1_operator.py status` should now list the gesture with no
   `gesture_problems`.
4. Test in dry-run with `python3 scripts/g1_operator.py gesture wave`. The FSM allowlist is empty,
   so this is refused until step K.

## K. Going live (only with robot ready, spotter present, remote/e-stop in hand)

1. Put the robot into the intended standing mode with the remote. Switch the action server to live
   **for this session only**:
   ```bash
   export G1_LIVE_CONFIRM=spotter-present-estop-in-hand
   make live            # compose.live.yaml overlay; refuses to start without the variable
   ```
2. Run `python3 scripts/g1_operator.py status`. Read `fsm_id` / `fsm_mode` in Locked Stance and in
   Movement Control. Cross-check them against the unitree_sdk2 example for your firmware
   (`third_party/unitree_sdk2`). Write the confirmed ids into `mode_check.allowed_fsm_ids`
   **only after this check**, then restart the live server.
3. With the spotter ready, run `python3 scripts/g1_operator.py gesture wave`. Ctrl-C sends
   `/stop`. Watch `arm_jitter_ms` in status; if p99 is above about 5 ms, see DECISIONS D18.
4. Steps: `python3 scripts/g1_operator.py walking on`, then `python3 scripts/g1_operator.py step left`.
   Keep the area clear: there is no obstacle sensing.
5. Only then let the orchestrator drive it: `docker compose up -d orchestrator` (the live action
   server keeps running). When done: `unset G1_LIVE_CONFIRM; make dry`.

If anything crashed mid-gesture, `status` shows `arm_maybe_held: true`. With the spotter ready,
run `python3 scripts/g1_operator.py release-arms`.

## L. Auto-start [approval]
`docker compose -f compose.yaml -f compose.autostart.yaml up -d`. Roll back with
`docker compose -f compose.yaml up -d`. Riva has its own restart policy from NVIDIA's script.

## M. Housekeeping
- **Privacy:** delete `tests/recordings/` and any opt-in transcript, audio or frame logs when
  testing ends. Do not copy them off the robot without the Data Privacy Officer's agreement.
- **Uninstall:** `make down`, `docker rmi g1voice/app:latest g1voice/vlm:r35 g1voice/piper:latest g1voice/builder:r35 g1voice/whisper:r35 g1voice/llamacpp:r35`,
  remove the Riva image and `~/g1-voice`. Nothing else on the host was changed.

## N. Troubleshooting
| Symptom | Check |
|---|---|
| `no mic packets for 1 s` | wrong interface/IP; voice service off; `ip maddr` shows 239.168.123.161? |
| DDS calls time out | `network_interface` in both configs; `CYCLONEDDS_URI` not overriding; robot reachable |
| `turn_error` / `vlm_error` in logs/metrics.jsonl | the error text is in the event; check `docker compose logs vlm` |
| VLM out of memory | `--max-context-len 1024`, stop fallbacks, check the owner's processes in tegrastats |
| Every gesture refused | `status.last_event.reason`; allowlist, heartbeat, URDF path, crash marker |
