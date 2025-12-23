CUDA MPI LLaMA Scheduler (llama.cpp + GGUF)

This project demonstrates a minimal, production-style GPU inference pipeline built on top of llama.cpp, combining:

CUDA-accelerated GGUF model inference

OpenAI-compatible HTTP server

Multi-rank / multi-client request scheduling

Latency & throughput benchmarking

It is designed to run even on low-VRAM NVIDIA GPUs (1 GB) and focuses on real systems engineering, not synthetic benchmarks.

🚀 Features

✅ llama.cpp HTTP server (/v1/chat/completions)

✅ CUDA GPU inference (tested on NVIDIA GeForce 940M, CC 5.0)

✅ GGUF quantized models (Q4_K_M fits in 1 GB VRAM)

✅ Multi-rank scheduler (MPI-style load generator)

✅ Latency metrics: mean, p50, p95, p99

✅ Throughput metrics: tokens/sec

✅ Prompt cache enabled

✅ No Docker, no Python, no frameworks

🧠 Model

Model: Gemma-3-1B-IT

Format: GGUF

Quantization: Q4_K_M

Size: ~762 MiB

Context Length: 32k

Backend: llama.cpp (CUDA)

🖥️ Hardware Tested
Component	Details
GPU	NVIDIA GeForce 940M
VRAM	1 GB
CUDA	12.8
Compute Capability	5.0
CPU	Intel (4 threads)
OS	Xubuntu 22.04
🔧 Build
./scripts/build.sh


This uses:

CMake

Ninja

Header-only nlohmann/json

▶️ Run llama.cpp Server (GPU)
./llama-server \
  -m gemma-3-1b-it-Q4_K_M.gguf \
  --port 8090 \
  --mlock


The server exposes:

POST http://127.0.0.1:8090/v1/chat/completions

⚙️ Run Scheduler (2 Ranks)
./scripts/run_2ranks.sh


Example output (GPU-backed):

=== llama-server latency (ms) ===
mean=3342.553 p50=3317.200 p95=4010.026 p99=4129.743

=== throughput ===
tokens=539 tokens/sec=16.12 ok=10 err=0

📊 Observations

GPU VRAM usage ~850–900 MiB (stable)

~10–12 tokens/sec on 1 GB VRAM

Prompt cache improves repeated request latency

Scheduler correctly overlaps requests

No CPU fallback when CUDA is enabled

🎯 Why This Project Matters

This project demonstrates real-world GPU systems engineering:

Running modern LLMs on constrained hardware

Understanding memory fitting, KV cache sizing, and prompt reuse

Measuring tail latency, not just averages

Building inference infrastructure, not just demos

It is directly relevant to roles involving:

GPU performance engineering

LLM inference systems

CUDA + C++ optimization

Edge / low-resource AI deployment

📚 Documentation

[QUICKSTART.md](QUICKSTART.md) - Get started quickly with step-by-step setup and testing

[CHANGELOG.md](CHANGELOG.md) - Recent changes and improvements

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture and design

[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) - Detailed project status and verification report

[docs/GPU_USAGE_GUIDE.md](docs/GPU_USAGE_GUIDE.md) - GPU usage verification and optimization guide

[docs/LOG_ANALYSIS.md](docs/LOG_ANALYSIS.md) - Performance analysis and benchmarking results