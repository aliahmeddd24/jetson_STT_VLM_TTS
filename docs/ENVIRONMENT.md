# Environment

> Status: **owner-reported values (compatibility check 2026-10-21); not yet re-verified by this repo.**
> Run `make discover` on the Jetson (read-only) and replace the "verified" column with its output.

| Item | Reported | Verified (date, `make discover`) |
|---|---|---|
| Module | Jetson Orin NX 16 GB, sm_87, G1 carrier, PC2 192.168.123.164 | |
| L4T / JetPack | R35.3.1 / 5.1.1 | |
| CUDA / cuDNN / TensorRT | 11.4 / 8.6 / 8.5.2 (`nvidia-jetpack` meta-package absent) | |
| OS | Ubuntu 20.04, glibc 2.31, kernel 5.10 | |
| Python | 3.8.10, numpy 1.17.4 (do not touch) | |
| RAM / swap / disk | 15 GB (13.8 GB free idle) / 7.5 GB / ~1.6 TB free | |
| Docker | 28.1.1, NVIDIA runtime registered, default runtime: ? | |
| Internet | none (github, pypi, huggingface, ghcr unreachable) | |
| Free ports | 8080, 8090 (also check 8091, 8092, 50051, 8000-8002) | |
| G1 network interface name | ? (the NIC on 192.168.123.0/24) → set `G1_IFACE` in .env | |
| Unitree SDK on host | ? | |
| Camera | ? (RealSense on PC2 in later batches, else on the locomotion computer) | |
| Mic stream | expected 239.168.123.161:5555, 16 kHz mono int16 | `make mic-capture` |
| Free memory with owner's programs running (tegrastats, a few minutes) | ? | |
| Robot firmware version | ? (≥ 1.5.1 needed for rt/sportmodestate) | |
| FSM id/mode in Locked Stance and Movement Control | ? | `scripts/g1_operator.py status` with the robot in each mode |
