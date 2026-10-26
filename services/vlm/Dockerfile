# VLM service: dustynv/nano_llm (r35, L4T 35.x) + our stdlib-only OpenAI wrapper.
# Only COPY instructions, so it builds on an x86 workstation without QEMU.
ARG NANO_LLM_IMAGE=dustynv/nano_llm:24.5-r35.4.1
FROM ${NANO_LLM_IMAGE}
COPY openai_server.py /opt/g1voice/openai_server.py
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "/opt/g1voice/openai_server.py"]
CMD ["--backend", "nanollm", "--model", "/models/VILA1.5-3b", "--host", "127.0.0.1", "--port", "8080", "--warmup"]
