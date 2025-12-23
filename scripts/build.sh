#!/usr/bin/env bash
set -euo pipefail

mkdir -p build
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=50
cmake --build build -j
echo "[ok] ./build/mls"
