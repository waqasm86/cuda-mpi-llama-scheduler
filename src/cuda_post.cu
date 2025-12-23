#include "mls/cuda_post.cuh"
#include <cuda_runtime.h>

namespace mls {

__global__ void post_kernel(int iters) {
  volatile int x = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = 0; i < iters; i++) {
    x = x * 1103515245 + 12345;
  }
}

double cuda_post_kernel_ms(int work) {
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);

  cudaEventRecord(a);
  post_kernel<<<16, 128>>>(work);
  cudaEventRecord(b);
  cudaEventSynchronize(b);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, a, b);

  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return (double)ms;
}

} // namespace mls
