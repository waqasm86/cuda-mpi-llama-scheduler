#!/usr/bin/env bash
set -euo pipefail

mpirun -np 2 ./build/mls \
  --llama http://127.0.0.1:8090 \
  --endpoint /v1/chat/completions \
  --iters 10 \
  --n_predict 64 \
  --timeout 60000 \
  --inflight 8 \
  --cuda_post 1 \
  --cuda_work 5000
