#pragma once
#include <vector>

namespace mls {
struct Summary {
  double mean_ms = 0.0;
  double p50_ms  = 0.0;
  double p95_ms  = 0.0;
  double p99_ms  = 0.0;
};
Summary summarize_ms(std::vector<double> v);
} // namespace mls
