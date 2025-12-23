#include <mpi.h>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>
#include <cstdlib>

#include "mls/http.hpp"
#include "mls/stats.hpp"
#include "mls/llama_api.hpp"
#include "mls/llama_parse.hpp"
#include "mls/cuda_post.cuh"

using clk = std::chrono::high_resolution_clock;

static double wall_s() {
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
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

  // ---- defaults ----
  std::string server = "http://127.0.0.1:8090";
  std::string endpoint = "/v1/chat/completions";
  int iters = 20;
  int n_predict = 64;
  int inflight = 8;
  int timeout_ms = 60000;

  int cuda_post = 1;
  int cuda_work = 5000;

  // ---- CLI ----
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--server")     server   = get_arg(i, argc, argv);
    if (a == "--endpoint")   endpoint = get_arg(i, argc, argv);
    if (a == "--iters")      iters    = std::atoi(get_arg(i, argc, argv).c_str());
    if (a == "--n_predict")  n_predict= std::atoi(get_arg(i, argc, argv).c_str());
    if (a == "--inflight")   inflight = std::atoi(get_arg(i, argc, argv).c_str());
    if (a == "--timeout")    timeout_ms = std::atoi(get_arg(i, argc, argv).c_str());
    if (a == "--cuda_post")  cuda_post = std::atoi(get_arg(i, argc, argv).c_str());
    if (a == "--cuda_work")  cuda_work = std::atoi(get_arg(i, argc, argv).c_str());
  }

  // curl init (safe on all ranks)
  mls::http_global_init();

  const int TAG_WORK = 1;
  const int TAG_DONE = 2;

  if (rank == 0) {
    std::printf("[mls] world=%d iters=%d inflight=%d server=%s endpoint=%s n_predict=%d\n",
      world, iters, inflight, server.c_str(), endpoint.c_str(), n_predict);
    std::printf("[mls] cuda_post=%d cuda_work=%d | Scheduler uses CUDA for post-processing\n",
      cuda_post, cuda_work);
    std::printf("[mls] NOTE: llama-server GPU/CPU backend is determined by its own -ngl parameter\n");
  }

  std::vector<double> lat_ms;
  lat_ms.reserve((size_t)iters);

  long long local_tokens = 0; // real tokens if usage is present, else fallback
  long long local_ok = 0;
  long long local_err = 0;

  auto do_one_request = [&](int job_id) -> std::pair<double, int> {
    std::string prompt = "MPI job=" + std::to_string(job_id) + " rank=" + std::to_string(rank) +
                         ". Reply in one short sentence about CUDA+MPI scheduling.";
    std::string body = mls::build_chat_body(prompt, n_predict);

    double t0 = wall_s();
    auto resp = mls::http_post_json(server + endpoint, body, timeout_ms);
    if (cuda_post) (void)mls::cuda_post_kernel_ms(cuda_work);
    double t1 = wall_s();

    if (!resp.err.empty()) {
      local_err++;
      return {(t1 - t0) * 1000.0, 0};
    }

    if (resp.status < 200 || resp.status >= 300) {
      local_err++;
      return {(t1 - t0) * 1000.0, 0};
    }

    auto parsed = mls::parse_chat_completions_response(resp.body);
    if (!parsed.ok) {
      // still count it as an HTTP success but parse failure
      local_err++;
      return {(t1 - t0) * 1000.0, 0};
    }

    local_ok++;

    int used = 0;
    if (parsed.usage.has_usage && parsed.usage.total_tokens > 0) {
      used = parsed.usage.total_tokens;
    } else {
      // fallback: assume n_predict completion tokens
      used = n_predict;
    }

    return {(t1 - t0) * 1000.0, used};
  };

  double t_start = 0.0;
  if (rank == 0) t_start = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);

  if (world == 1) {
    for (int k = 0; k < iters; k++) {
      auto [ms, toks] = do_one_request(k);
      lat_ms.push_back(ms);
      local_tokens += toks;
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

        // receive completion: [lat_ms, tokens_used]
        double payload[2] = {0.0, 0.0};
        MPI_Status st{};
        MPI_Recv(payload, 2, MPI_DOUBLE, MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &st);

        infl--;
        done++;

        lat_ms.push_back(payload[0]);
        local_tokens += (long long)payload[1];
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

        auto [ms, toks] = do_one_request(job);

        double payload[2] = {ms, (double)toks};
        MPI_Send(payload, 2, MPI_DOUBLE, 0, TAG_DONE, MPI_COMM_WORLD);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t_end = MPI_Wtime();

  // Aggregate tokens + success/errors across ranks
  long long global_tokens = 0, global_ok = 0, global_err = 0;
  MPI_Allreduce(&local_tokens, &global_tokens, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_err, &global_err, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  // Gather latencies at rank0
  int local_n = (int)lat_ms.size();
  std::vector<int> counts, displs;
  std::vector<double> all_lat;

  if (rank == 0) counts.resize(world);
  MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    displs.resize(world, 0);
    int total = 0;
    for (int r = 0; r < world; r++) {
      displs[r] = total;
      total += counts[r];
    }
    all_lat.resize(total);
  }

  MPI_Gatherv(lat_ms.data(), local_n, MPI_DOUBLE,
              all_lat.data(), counts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  if (rank == 0) {
    auto s = mls::summarize_ms(all_lat);
    double seconds = (t_end - t_start);
    double tok_s = (seconds > 0.0) ? (double)global_tokens / seconds : 0.0;

    std::printf("\n=== llama-server latency (ms) ===\n");
    std::printf("mean=%.3f p50=%.3f p95=%.3f p99=%.3f (n=%zu)\n",
      s.mean_ms, s.p50_ms, s.p95_ms, s.p99_ms, all_lat.size());

    std::printf("\n=== throughput ===\n");
    std::printf("wall=%.3fs tokens=%lld tokens/sec=%.2f ok=%lld err=%lld\n",
      seconds, global_tokens, tok_s, global_ok, global_err);

    std::printf("\nnotes:\n");
    std::printf("- If response 'usage.total_tokens' is present, tokens/sec is real.\n");
    std::printf("- Otherwise fallback uses n_predict.\n");
  }

  mls::http_global_cleanup();
  MPI_Finalize();
  return 0;
}
