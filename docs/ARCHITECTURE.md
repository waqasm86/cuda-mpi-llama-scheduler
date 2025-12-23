# System Architecture

## Overview

This project implements a distributed LLM inference system using MPI for scheduling and CUDA for GPU acceleration. It demonstrates production-grade patterns for managing concurrent requests, measuring performance, and utilizing GPU resources efficiently.

## Architecture Components

### 1. MPI Scheduler (Rank-based Distribution)

**Rank 0: Scheduler (Ingress + Coordination)**
- Distributes job IDs to worker ranks
- Collects latency metrics via `MPI_Gatherv`
- Aggregates token counts via `MPI_Allreduce`
- Prints statistical summary (mean, p50, p95, p99)

**Worker Ranks: Execution (HTTP + CUDA)**
- Receive job assignments from Rank 0
- Build OpenAI-compatible chat completion requests
- POST to llama.cpp server via CURL
- Execute CUDA post-processing kernel (optional)
- Parse JSON response and extract metrics
- Return latency and token counts to Rank 0

### 2. llama.cpp Server Integration

**HTTP API**
- Endpoint: `POST /v1/chat/completions`
- OpenAI-compatible interface
- Supports streaming and non-streaming responses

**GPU Inference**
- CUDA-accelerated GGUF model execution
- Configurable layer offloading (`-ngl` parameter)
- KV cache management (prompt caching)
- Context checkpointing for efficient memory usage

### 3. CUDA Post-Processing Kernel

**Purpose**: Demonstrates CUDA + MPI coexistence

**Implementation**:
```cuda
__global__ void post_kernel(int iters) {
  volatile int x = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < iters; i++) {
    x = x * 1103515245 + 12345;  // LCG
  }
}
```

**Execution**:
- Grid: 16 blocks × 128 threads = 2048 GPU threads
- Configurable workload via `--cuda_work` parameter
- Measures execution time using CUDA events

### 4. Metrics Collection

**Latency Metrics**
- Per-request timing using `std::chrono`
- Statistical analysis: mean, p50, p95, p99
- Gathered across all MPI ranks

**Throughput Metrics**
- Wall clock time measurement
- Token counting from API responses
- Tokens/second calculation
- Success/error tracking

## Data Flow

```
┌─────────────────────────────────────────────────────────┐
│                    MPI_COMM_WORLD                       │
│                                                         │
│  ┌──────────┐              ┌──────────┐               │
│  │  Rank 0  │              │  Rank 1+ │               │
│  │(Scheduler)│◄────────────►│ (Worker) │               │
│  │          │  MPI_Send    │          │               │
│  │          │  MPI_Recv    │          │               │
│  └────┬─────┘              └────┬─────┘               │
│       │                         │                      │
│       ▼                         ▼                      │
│  Gather Latency            HTTP POST                  │
│  Allreduce Tokens         CUDA Post-Kernel            │
│  Print Statistics         Return Metrics              │
└─────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────┐
│                 llama.cpp Server                       │
│  ┌────────────────────┐  ┌────────────────────┐       │
│  │  CPU Layers        │  │  GPU Layers (CUDA) │       │
│  │  (13/27)           │  │  (14/27)           │       │
│  └────────────────────┘  └────────────────────┘       │
│                 GGUF Model (Q4_K_M)                    │
└─────────────────────────────────────────────────────────┘
```

## Why This is "Inference at Scale" Thinking

### 1. Tail Latency Measurement
- Not just averages - tracks p95 and p99
- Critical for user-facing applications
- Identifies performance outliers

### 2. Concurrency Management
- Configurable inflight requests
- Tests saturation points
- Finds optimal concurrency levels

### 3. Resource Utilization
- GPU VRAM monitoring
- Layer offloading strategies
- Memory-constrained optimization

### 4. Distributed Coordination
- MPI-based scheduling mimics production systems
- Worker pool pattern
- Fault tolerance considerations (error tracking)

### 5. Performance Profiling
- Throughput vs latency tradeoffs
- Prompt cache effectiveness
- GPU utilization patterns

## Key Design Patterns

**Worker Pool Pattern**: Rank 0 distributes work, workers execute independently

**Metrics Aggregation**: Centralized collection for global statistics

**Graceful Degradation**: Error tracking without stopping execution

**Configurable Backends**: Scheduler works with CPU or GPU llama-server

**Separation of Concerns**: HTTP client, JSON parsing, CUDA kernels are modular

## File Structure

```
src/
├── main.cu              # MPI orchestrator, metrics aggregation
├── http.cpp             # CURL-based HTTP client with retry
├── llama_api.cpp        # Request body builder
├── llama_parse.cpp      # JSON response parser
├── cuda_post.cu         # CUDA post-processing kernel
└── stats.cpp            # Percentile calculations

include/mls/
├── cuda_post.cuh        # CUDA kernel header
├── http.hpp             # HTTP client interface
├── llama_api.hpp        # API builder interface
├── llama_parse.hpp      # Parser interface
└── stats.hpp            # Statistics interface
```

## Performance Characteristics

**Tested Configuration**:
- GPU: NVIDIA GeForce 940M (1GB VRAM)
- Model: Gemma-3-1B-IT Q4_K_M (~762 MiB)
- Layers on GPU: 14/27 (52% offload)
- Context: 1536 tokens

**Measured Performance**:
- Throughput: 17-22 tokens/sec
- Latency (p50): 2.4-3.9 seconds
- VRAM Usage: ~610 MiB (stable)
- Success Rate: 100%

**Scalability**:
- Linear scaling with iterations (2x work = 2x time)
- Consistent performance across different inflight values
- No memory leaks or resource exhaustion

## Future Enhancements

- Multi-GPU support (CUDA peer-to-peer)
- Dynamic work stealing between ranks
- Request prioritization and queuing
- Streaming response handling
- Integration with monitoring systems (Prometheus, Grafana)
