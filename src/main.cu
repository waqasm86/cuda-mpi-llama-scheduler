#include <mpi.h>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include <cstdlib>

#include "mls/http.hpp"
#include "mls/stats.hpp"
#include "mls/llama_api.hpp"
#include "mls/cuda_post.cuh"

using clk = std::chrono::high_resolution_clock;

static double wall_ms() {
  return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

static std::string get_arg(int& i, int argc, char** argv) {
  if (i + 1 >= argc) return {};
  return argv[++i];
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int world = 1, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string llama_base = "http://127.0.0.1:8090";
  std::string endpoint   = "/v1/chat/completions";
  int iters = 20;
  int n_predict = 64;
  int timeout_ms = 60000;
  int inflight = 8;
  int cuda_post = 1;
  int cuda_work = 5000;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--llama")     llama_base = get_arg(i, argc, argv);
    if (a == "--endpoint")  endpoint   = get_arg(i, argc, argv);
    if (a == "--iters")     iters      = std::atoi(get_arg(i, argc, argv).c_str());
    if (a == "--n_predict") n_predict  = std::atoi(get_arg(i, argc, argv).c_str());
    if (a == "--timeout")   timeout_ms = std::atoi(get_arg(i, argc, argv).c_str());
    if (a == "--inflight")  inflight   = std::atoi(get_arg(i, argc, argv).c_str());
    if (a == "--cuda_post") cuda_post  = std::atoi(get_arg(i, argc, argv).c_str());
    if (a == "--cuda_work") cuda_work  = std::atoi(get_arg(i, argc, argv).c_str());
  }

  const int TAG_WORK = 1;
  const int TAG_DONE = 2;

  if (rank == 0) {
    std::printf("[mls] world=%d iters=%d inflight=%d llama=%s endpoint=%s n_predict=%d cpu-only-llama-ok\n",
      world, iters, inflight, llama_base.c_str(), endpoint.c_str(), n_predict);
  }

  std::vector<double> lat_ms;
  lat_ms.reserve((size_t)iters);

  // If world==1 run locally (no scheduling)
  if (world == 1) {
    for (int k = 0; k < iters; k++) {
      std::string prompt = "Say hello. Iter=" + std::to_string(k);
      std::string body = mls::build_chat_body(prompt, n_predict);

      double t0 = wall_ms();
      auto resp = mls::http_post_json(llama_base + endpoint, body, timeout_ms);
      if (cuda_post) (void)mls::cuda_post_kernel_ms(cuda_work);
      double t1 = wall_ms();

      if (resp.status < 200 || resp.status >= 300) {
        std::fprintf(stderr, "[rank0] HTTP %ld body=%s\n", resp.status, resp.body.c_str());
      }
      lat_ms.push_back(t1 - t0);
    }
  } else {
    if (rank == 0) {
      int next_worker = 1;
      int sent = 0, done = 0, infl = 0;

      while (done < iters) {
        while (sent < iters && infl < inflight) {
          int job = sent;
          MPI_Send(&job, 1, MPI_INT, next_worker, TAG_WORK, MPI_COMM_WORLD);
          sent++; infl++;
          next_worker++;
          if (next_worker >= world) next_worker = 1;
        }

        double one = 0.0;
        MPI_Status st{};
        MPI_Recv(&one, 1, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &st);
        infl--;
        done++;
        lat_ms.push_back(one);
      }

      // stop workers
      int stop = -1;
      for (int r = 1; r < world; r++) {
        MPI_Send(&stop, 1, MPI_INT, r, TAG_WORK, MPI_COMM_WORLD);
      }
    } else {
      while (true) {
        int job = 0;
        MPI_Status st{};
        MPI_Recv(&job, 1, MPI_INT, 0, TAG_WORK, MPI_COMM_WORLD, &st);
        if (job < 0) break;

        std::string prompt = "Summarize: MPI job=" + std::to_string(job) + " rank=" + std::to_string(rank);
        std::string body = mls::build_chat_body(prompt, n_predict);

        double t0 = wall_ms();
        auto resp = mls::http_post_json(llama_base + endpoint, body, timeout_ms);
        if (cuda_post) (void)mls::cuda_post_kernel_ms(cuda_work);
        double t1 = wall_ms();

        if (resp.status < 200 || resp.status >= 300) {
          // Keep workers quiet; rank0 prints summary. Optionally log.
        }

        double d = (t1 - t0);
        MPI_Send(&d, 1, MPI_DOUBLE, 0, TAG_DONE, MPI_COMM_WORLD);
      }
    }
  }

  // Collect and print stats at rank0
  int local_n = (int)lat_ms.size();
  std::vector<int> counts, displs;
  std::vector<double> all;

  if (rank == 0) counts.resize(world);
  MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    displs.resize(world, 0);
    int total = 0;
    for (int r = 0; r < world; r++) {
      displs[r] = total;
      total += counts[r];
    }
    all.resize(total);
  }

  MPI_Gatherv(lat_ms.data(), local_n, MPI_DOUBLE,
              all.data(), counts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  if (rank == 0) {
    auto s = mls::summarize_ms(all);
    std::printf("\n=== llama-server latency (ms) ===\n");
    std::printf("mean=%.3f p50=%.3f p95=%.3f p99=%.3f (n=%zu)\n",
      s.mean_ms, s.p50_ms, s.p95_ms, s.p99_ms, all.size());

    long long scheduled = 1LL * iters * n_predict;
    std::printf("\n=== throughput model ===\n");
    std::printf("scheduled_tokens_est=%lld (=iters*n_predict)\n", scheduled);
    std::printf("note: use server response token counts if you enable that parsing later\n");
  }

  MPI_Finalize();
  return 0;
}
