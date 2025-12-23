#include "mls/stats.hpp"
#include <algorithm>
#include <numeric>

namespace mls {

static double percentile(std::vector<double>& v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  double idx = p * (v.size() - 1);
  size_t i0 = (size_t)idx;
  size_t i1 = std::min(i0 + 1, v.size() - 1);
  double frac = idx - (double)i0;
  return v[i0] * (1.0 - frac) + v[i1] * frac;
}

Summary summarize_ms(std::vector<double> v) {
  Summary s;
  if (v.empty()) return s;
  s.mean_ms = std::accumulate(v.begin(), v.end(), 0.0) / (double)v.size();

  auto tmp = v;
  s.p50_ms = percentile(tmp, 0.50);
  tmp = v;
  s.p95_ms = percentile(tmp, 0.95);
  tmp = v;
  s.p99_ms = percentile(tmp, 0.99);
  return s;
}

} // namespace mls
