# g1-voice. Run `make help`. Workstation targets need internet; Jetson targets never do.
SHELL := /bin/bash
-include .env
export
COMPOSE := docker compose
APP_RUN := $(COMPOSE) run --rm --no-deps
NOW := $(shell date +%Y%m%d-%H%M%S)

.PHONY: help
help:  ## list targets
	@grep -hE '^[a-zA-Z0-9_.-]+:.*## ' $(MAKEFILE_LIST) | sed -E 's/:.*## /\t/' | column -t -s $$'\t'

.env:
	cp .env.example .env

# ---------------------------------------------------------------- workstation (internet)
.PHONY: bundle
bundle: .env  ## [workstation] fetch sources/models, build+save all images -> offline-bundle/
	bash scripts/offline/fetch_all.sh

.PHONY: test-local
test-local:  ## [anywhere] unit + integration tests (needs python3, numpy, pyyaml, requests)
	python3 -m unittest discover -s tests -v

# ---------------------------------------------------------------- jetson (offline)
.PHONY: install
install: .env  ## [jetson] load images from offline-bundle/ and place models
	bash scripts/offline/install_bundle.sh

.PHONY: discover
discover:  ## [jetson] Phase 0 read-only discovery -> logs/environment-*.md
	bash scripts/discover_env.sh | tee logs/environment-$(NOW).md

.PHONY: gpu-sanity
gpu-sanity:  ## [jetson] Phase 1 CUDA check in an r35 container
	bash scripts/gpu_sanity.sh

.PHONY: build-native
build-native:  ## [jetson] build whisper.cpp + llama.cpp CUDA images offline (time-boxed fallbacks)
	bash scripts/offline/build_native.sh

.PHONY: riva-prepare riva-deploy riva-start riva-stop
riva-prepare:  ## [jetson] patched private copy of the Riva quickstart
	bash services/riva/riva.sh prepare
riva-deploy:  ## [jetson] build Riva TensorRT engines from bundled RMIRs
	bash services/riva/riva.sh deploy-offline
riva-start:  ## [jetson] start Riva (ASR+TTS, :50051)
	bash services/riva/riva.sh start
riva-stop:
	bash services/riva/riva.sh stop

.PHONY: up down logs ps
up: .env  ## [jetson] start Riva + VLM + action server (dry-run) + orchestrator
	@if [[ ",$(COMPOSE_PROFILES)," != *",mock,"* ]] && grep -q '^\s*backend: "riva"' config/settings.yaml; then bash services/riva/riva.sh start; fi
	$(COMPOSE) up -d
	$(COMPOSE) ps
down:  ## stop our containers (never touches other programs)
	$(COMPOSE) --profile nanollm --profile llamacpp --profile mock --profile piper --profile whisper down
	-@bash services/riva/riva.sh stop 2>/dev/null || true
logs:
	$(COMPOSE) logs -f --tail=100
ps:
	$(COMPOSE) ps

.PHONY: live dry
live:  ## [jetson] action server LIVE (needs G1_LIVE_CONFIRM; spotter + e-stop!)
	$(COMPOSE) -f compose.yaml -f compose.live.yaml up -d --force-recreate action_server
	python3 scripts/g1_operator.py status
dry:  ## [jetson] action server back to dry-run
	G1_LIVE_CONFIRM= $(COMPOSE) up -d --force-recreate action_server

.PHONY: up-mock
up-mock:  ## [jetson/any docker] full pipeline with mock STT/VLM/TTS, WAV in/out, no robot
	COMPOSE_PROFILES=mock $(COMPOSE) up -d action_server vlm_mock
	$(APP_RUN) -e G1V__STT__BACKEND=mock -e G1V__TTS__BACKEND=mock -e G1V__VAD__BACKEND=energy orchestrator \
	  python -m g1voice.main --config config/settings.yaml --wav tests/fixtures/hello.wav \
	  --out-wav logs/mock_out.wav --max-turns 1 --wait 60

.PHONY: test
test:  ## [jetson] run the test-suite inside the app image
	$(APP_RUN) orchestrator python -m unittest discover -s /app/tests -v

# ---------------------------------------------------------------- measurements
.PHONY: mic-capture
mic-capture:  ## [jetson] record 5 s from the G1 mic to logs/g1_mic.wav (host python, stdlib)
	python3 scripts/capture_g1_mic.py --iface $(G1_IFACE) --seconds 5 --out logs/g1_mic.wav

.PHONY: bench-vlm bench-stt bench-tts bench-e2e report soak
bench-vlm:  ## Phase 2 VLM: TTFT, tok/s (+ run scripts/tegrastats_log.sh for memory)
	$(APP_RUN) orchestrator python /app/scripts/bench/bench_vlm.py --out logs/bench_vlm-$(NOW).json
bench-stt:  ## Phase 3 STT latency on WAVs in tests/recordings/
	$(APP_RUN) orchestrator python /app/scripts/bench/bench_stt.py --dir tests/recordings --out logs/bench_stt-$(NOW).json
bench-tts:  ## Phase 3 TTS first-audio + RTF
	$(APP_RUN) orchestrator python /app/scripts/bench/bench_tts.py --out logs/bench_tts-$(NOW).json
bench-e2e:  ## Phase 6 end-to-end over recorded WAVs (no robot needed)
	$(APP_RUN) orchestrator python /app/scripts/bench/bench_e2e.py --dir tests/recordings --metrics logs/e2e-$(NOW).jsonl
report:  ## latency/benchmark summary -> stdout (paste into docs/BENCHMARKS.md)
	python3 scripts/bench/report.py logs
soak:  ## Phase 6 30-minute soak with tegrastats
	bash scripts/soak.sh 1800
