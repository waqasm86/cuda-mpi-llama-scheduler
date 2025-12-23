#pragma once
namespace mls {
// Tiny post-processing kernel to prove CUDA+MPI coexistence.
// Returns kernel time in milliseconds.
double cuda_post_kernel_ms(int work);
} // namespace mls
