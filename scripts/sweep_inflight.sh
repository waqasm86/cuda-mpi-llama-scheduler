#!/usr/bin/env bash
set -euo pipefail

for infl in 1 2 4 8; do
  echo "==== inflight=$infl ===="
  mpirun -np 2 ./build/mls \
    --server http://127.0.0.1:8090 \
    --endpoint /v1/chat/completions \
    --iters 20 \
    --n_predict 64 \
    --timeout 60000 \
    --inflight $infl \
    --cuda_post 1 \
    --cuda_work 5000
  echo
done
