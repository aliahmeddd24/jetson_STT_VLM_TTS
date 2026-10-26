"""g1voice: local STT -> VLM -> TTS orchestrator for the Unitree G1 (PC2).

Design ideas (VAD loop, streaming sentence chunker, OpenAI-style VLM client,
YAML-overridable config) follow NVIDIA-AI-IOT/reachy-mini-jetson-assistant
(Apache-2.0). This is an independent re-implementation; see NOTICE.
"""
__version__ = "0.1.0"
