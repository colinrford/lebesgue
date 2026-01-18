/*
 *  benchmark_scaling.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *  Rigorous scaling test for Lebesgue Gram matrix construction.
 *  Tests both Degree (N) and Sample Size (M) scaling across parallel thresholds.
 */

import std;
import lam.lebesgue;
import lam.linearalgebra;

namespace leb = lam::leb;
using lam::matrix;
using lam::vector;

int main()
{
  std::println("=== Lebesgue Scaling Benchmark ===");
  std::println("Hardware concurrency: {}", std::thread::hardware_concurrency());

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  // Scaling Dimensions
  std::vector<int> degrees = {5, 6, 10, 20, 50};
  std::vector<int> samples = {500, 900, 1000, 1100, 5000, 10000, 100000, 1000000};

  // Pre-generate maximum data to avoid alloc overhead during timed benchmarks
  int max_samples = samples.back();
  vector<double> x_all(max_samples);
  vector<double> w_all(max_samples);
  for (int i = 0; i < max_samples; ++i)
  {
    x_all[i] = dist(rng);
    w_all[i] = 1.0 / max_samples;
  }

  std::println("\n{:<8} | {:<10} | {:<15} | {:<15} | {:<10}", "Degree", "Samples", "Time (ms)", "M*N^2/s (perf)",
               "Dispatch");
  std::println("{:-^70}", "");

  for (int n : degrees)
  {
    for (int m : samples)
    {
      // Warmup
      auto x_full = x_all.as_span();
      auto w_full = w_all.as_span();
      std::span<const double> x_span = x_full.subspan(0, m);
      std::span<const double> w_span = w_full.subspan(0, m);

      // Heuristic check to predict dispatch mode
      bool expect_parallel = (m >= 1000 && n > 5);
      std::string dispatch_mode = expect_parallel ? "PARALLEL" : "SERIAL";

      // Adaptive iterations
      int iters = 1;
      if (m < 2000)
        iters = 200;
      else if (m < 20000)
        iters = 50;
      else if (m < 200000)
        iters = 10;
      else
        iters = 5;

      auto start = std::chrono::steady_clock::now();

      for (int k = 0; k < iters; ++k)
      {
        auto G = leb::gram_matrix_from_samples<double>(x_span, w_span, n);
        // Simple side-effect to prevent optimization
        if (G[0, 0] > 1e9) [[unlikely]]
          (void)G[0, 0];
      }

      auto end = std::chrono::steady_clock::now();
      double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
      double avg_ms = total_ms / iters;

      // Performance metric: Elements * Degree^2 (complexity is O(M*N^2))
      double complexity = (double)m * (double)n * (double)n;
      double perf = complexity / (avg_ms / 1000.0); // units per second

      std::println("{:<8} | {:<10} | {:<15.4f} | {:<15.2e} | {:<10}", n, m, avg_ms, perf, dispatch_mode);
    }
    std::println("{:-^70}", "-");
  }

  return 0;
}
