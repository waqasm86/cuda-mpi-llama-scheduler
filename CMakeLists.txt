cmake_minimum_required(VERSION 3.24)
project(cuda_mpi_llama_scheduler LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GeForce 940M
set(CMAKE_CUDA_ARCHITECTURES 50)

find_package(MPI REQUIRED)
find_package(CURL REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(mls
  src/main.cu
  src/http.cpp
  src/stats.cpp
  src/llama_api.cpp
  src/cuda_post.cu
)

target_include_directories(mls PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(mls PRIVATE
  MPI::MPI_CXX
  CURL::libcurl
  CUDA::cudart
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(mls PRIVATE -Wall -Wextra -Wpedantic)
endif()

