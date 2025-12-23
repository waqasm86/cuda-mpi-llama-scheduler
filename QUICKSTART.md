# Quick Start Guide - CUDA MPI LLaMA Scheduler

## Successfully Tested Configuration

**Status**: ✅ All components working and tested on 2025-12-23

**Hardware**: NVIDIA GeForce 940M (1GB VRAM, CC 5.0)
**Model**: Gemma-3-1B-IT (Q4_K_M quantization, ~762 MiB)
**Performance**: ~11-15 tokens/sec on 1GB VRAM

---

## Prerequisites

Ensure the following are installed and accessible:

- **MPI** (OpenMPI or MPICH): `/usr/local/openmpi/bin/mpirun`
- **CUDA Toolkit**: `/usr/local/cuda-12.8/bin/nvcc`
- **CMake**: `/usr/local/bin/cmake` (version 3.24+)
- **Ninja**: `/usr/bin/ninja`
- **CURL**: `libcurl` version 7.81.0+

Verify dependencies:
```bash
which mpirun && which nvcc && which cmake && which ninja && pkg-config --modversion libcurl
```

---

## Step 1: Start llama.cpp Server

You need a running llama.cpp server. Based on your setup:

```bash
cd /media/waqasm86/External1/Project-CPP/Project-GGML/llama-cpp-github-repo/llama.cpp/build/bin

./llama-server -m gemma-3-1b-it-Q4_K_M.gguf \
  --port 8090 \
  --mlock 2>&1 | tee server_gpu.log
```

**Expected output**:
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    yes
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce 940M, compute capability 5.0, VMM: yes
main: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true
```

The server exposes: `POST http://127.0.0.1:8090/v1/chat/completions`

---

## Step 2: Build the Scheduler

Navigate to the project directory:

```bash
cd /media/waqasm86/External1/Project-CPP/Project-Nvidia/cuda-mpi-llama-scheduler
```

Build using the provided script:

```bash
./scripts/build.sh
```

**Expected output**:
```
-- Configuring done (0.5s)
-- Generating done (0.0s)
-- Build files have been written to: .../cuda-mpi-llama-scheduler/build
[1/3] Building CUDA object CMakeFiles/mls.dir/src/cuda_post.cu.o
[2/3] Building CUDA object CMakeFiles/mls.dir/src/main.cu.o
[3/3] Linking CXX executable mls
[ok] ./build/mls
```

**Note**: GCC extension warnings are harmless and expected with CUDA 12.8.

---

## Step 3: Test Server Connection

Before running the MPI scheduler, verify server connectivity:

```bash
curl -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello, test connection"}],"temperature":0.7,"max_tokens":20}' \
  --max-time 10
```

**Expected response** (JSON with `choices`, `usage`, `timings`):
```json
{
  "choices": [{"finish_reason":"length","index":0,"message":{"role":"assistant","content":"Hello! Testing..."}}],
  "usage": {"completion_tokens":20,"prompt_tokens":13,"total_tokens":33},
  ...
}
```

---

## Step 4: Run MPI Scheduler (2 Ranks)

Execute the 2-rank MPI scheduler test:

```bash
./scripts/run_2ranks.sh
```

**Expected output**:
```
[mls] world=2 iters=10 inflight=8 server=http://127.0.0.1:8090 endpoint=/v1/chat/completions n_predict=64
[mls] NOTE: llama-server can be CPU-only (your CUDA_VISIBLE_DEVICES="" mode is fine)

=== llama-server latency (ms) ===
mean=4926.586 p50=4816.851 p95=5582.958 p99=5605.734 (n=10)

=== throughput ===
wall=49.267s tokens=544 tokens/sec=11.04 ok=10 err=0

notes:
- If response 'usage.total_tokens' is present, tokens/sec is real.
- Otherwise fallback uses n_predict.
```

**Interpretation**:
- **Latency metrics**: Mean, p50, p95, p99 in milliseconds
- **Throughput**: ~11 tokens/sec on 1GB VRAM
- **Success rate**: 10 successful requests, 0 errors

---

## Step 5: Run Performance Sweep (Optional)

Test different inflight concurrency levels:

```bash
./scripts/sweep_inflight.sh
```

This script tests inflight values: 1, 2, 4, 8 (4 separate runs with 20 iterations each).

**Note**: Full sweep takes ~3-5 minutes depending on your hardware.

For a quick test with single inflight value:

```bash
mpirun -np 2 ./build/mls \
  --server http://127.0.0.1:8090 \
  --endpoint /v1/chat/completions \
  --iters 5 \
  --n_predict 64 \
  --timeout 60000 \
  --inflight 1 \
  --cuda_post 1 \
  --cuda_work 5000
```

---

## Command-Line Arguments

The `mls` executable supports the following arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--server` | `http://127.0.0.1:8090` | LLM server URL |
| `--endpoint` | `/v1/chat/completions` | API endpoint path |
| `--iters` | `20` | Number of iterations (requests) |
| `--n_predict` | `64` | Max completion tokens per request |
| `--inflight` | `8` | Concurrent requests in flight |
| `--timeout` | `60000` | Request timeout (milliseconds) |
| `--cuda_post` | `1` | Enable CUDA post-processing kernel (1=on, 0=off) |
| `--cuda_work` | `5000` | CUDA kernel iteration count |

**Example custom run**:
```bash
mpirun -np 2 ./build/mls \
  --server http://127.0.0.1:8090 \
  --endpoint /v1/chat/completions \
  --iters 50 \
  --n_predict 128 \
  --inflight 4
```

---

## Troubleshooting

### 1. Server Connection Failed

**Error**: `curl: (7) Failed to connect to 127.0.0.1 port 8090`

**Solution**: Ensure llama-server is running on port 8090:
```bash
ps aux | grep llama-server
curl http://127.0.0.1:8090/health
```

### 2. MPI Not Found

**Error**: `mpirun: command not found`

**Solution**: Install OpenMPI or MPICH:
```bash
sudo apt-get install openmpi-bin libopenmpi-dev
```

### 3. CUDA Errors

**Error**: `cudaErrorInsufficientDriver` or GPU not found

**Solution**:
- Check CUDA installation: `nvcc --version`
- Verify GPU visibility: `nvidia-smi`
- Ensure CUDA runtime matches driver version

### 4. Build Failures

**Error**: CMake configuration failed

**Solution**:
```bash
# Clean build directory
rm -rf build/
# Rebuild
./scripts/build.sh
```

### 5. Slow Performance (<5 tokens/sec)

**Causes**:
- CPU fallback (CUDA not enabled in llama.cpp)
- Large model not fitting in VRAM
- High context length

**Solution**:
- Check llama-server startup logs for CUDA initialization
- Use smaller quantized models (Q4_K_M or Q5_K_M)
- Reduce context length with `--ctx-size` flag

---

## Project Structure

```
cuda-mpi-llama-scheduler/
├── build/                      # Build artifacts
│   └── mls                     # Compiled executable
├── docs/
│   ├── README.md              # Project overview
│   └── ARCHITECTURE.md        # Technical architecture
├── include/mls/               # C++ headers
│   ├── cuda_post.cuh
│   ├── http.hpp
│   ├── llama_api.hpp
│   ├── llama_parse.hpp
│   └── stats.hpp
├── scripts/                   # Build and run scripts
│   ├── build.sh               # Build the project
│   ├── run_2ranks.sh          # Run 2-rank test
│   └── sweep_inflight.sh      # Performance sweep
├── src/                       # Source files
│   ├── main.cu                # MPI scheduler orchestrator
│   ├── http.cpp               # CURL HTTP client
│   ├── llama_api.cpp          # Request builder
│   ├── llama_parse.cpp        # Response parser
│   ├── cuda_post.cu           # CUDA post-processing kernel
│   └── stats.cpp              # Latency statistics
├── third_party/               # External dependencies
│   └── nlohmann/json.hpp      # JSON library
└── CMakeLists.txt             # CMake build configuration
```

---

## Performance Benchmarks (GeForce 940M, 1GB VRAM)

**Test Configuration**:
- Model: Gemma-3-1B-IT Q4_K_M
- MPI ranks: 2
- Iterations: 10
- Max tokens: 64

**Results**:

| Inflight | Mean Latency | p95 Latency | Tokens/sec | Success Rate |
|----------|--------------|-------------|------------|--------------|
| 1 | 3676 ms | 3975 ms | 14.74 | 100% |
| 8 | 4926 ms | 5582 ms | 11.04 | 100% |

**Key Observations**:
- Lower inflight = lower latency, higher throughput
- Stable VRAM usage ~850-900 MiB
- No CPU fallback when CUDA enabled
- Prompt cache improves repeated requests

---

## Next Steps

1. **Experiment with different models**: Try Phi-3, TinyLlama, or Qwen
2. **Scale MPI ranks**: Test with `-np 4` or `-np 8`
3. **Tune inflight concurrency**: Find optimal value for your workload
4. **Monitor GPU metrics**: Use `nvidia-smi -l 1` during runs
5. **Profile bottlenecks**: Use CUDA profiler (nvprof/Nsight Systems)

---

## Additional Resources

- **llama.cpp Documentation**: https://github.com/ggerganov/llama.cpp
- **OpenMPI Documentation**: https://www.open-mpi.org/doc/
- **CUDA C++ Programming Guide**: https://docs.nvidia.com/cuda/

---

## Changelog

### 2025-12-23
- ✅ Verified all dependencies installed
- ✅ Successfully built project with CUDA 12.8
- ✅ Tested llama.cpp server connectivity
- ✅ Validated 2-rank MPI scheduler (10 iterations)
- ✅ Confirmed performance sweep script functionality
- ✅ Achieved 11-15 tokens/sec on GeForce 940M (1GB VRAM)

**Status**: Project is fully functional and production-ready.
