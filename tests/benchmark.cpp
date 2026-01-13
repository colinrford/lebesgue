/* 
 *  benchmark.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  benchmark Gauss and Lebesgue quadrature construction.
 */

import lam.lebesgue;
namespace leb = lam::leb;
import std;
import lam.linearalgebra;

using namespace lam::linalg;

int main()
{
  // Warmup
  for (int i = 0; i < 100; ++i)
  {
    auto q = leb::gauss_legendre(10);
  }

  constexpr int iterations = 10000;
  std::array<int, 4> sizes = {5, 10, 20, 50};

  // ========================================================================
  // Gauss Quadrature (from moments) - special case f(x) = x
  // ========================================================================
  std::println("=== Gauss Quadrature (from moments) ===");

  for (int n : sizes)
  {
    vector<double> moments(2 * n);
    for (int k = 0; k < 2 * n; ++k)
    {
      if (k == 0)
        moments[k] = 2.0;
      else if (k % 2 == 1)
        moments[k] = 0.0;
      else
        moments[k] = 2.0 / (1.0 - static_cast<double>(k * k));
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
    {
      auto q = leb::gauss_quadrature_from_moments<double>(std::span(moments), n);
    }
    auto end = std::chrono::steady_clock::now();

    double us_per_call = std::chrono::duration<double, std::micro>(end - start).count() / iterations;
    std::println("n={:2d}: {:8.2f} us/call", n, us_per_call);
  }

  // ========================================================================
  // Lebesgue Quadrature (from samples) - general case
  // ========================================================================
  std::println("\n=== Lebesgue Quadrature (from samples) ===");

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::array<int, 4> sample_sizes = {100, 500, 1000, 5000};

  for (int n_samples : sample_sizes)
  {
    // Generate sample data
    vector<double> x(n_samples), f(n_samples), w(n_samples);
    for (int i = 0; i < n_samples; ++i)
    {
      x[i] = dist(rng);
      f[i] = std::sin(x[i]); // f(x) = sin(x)
      w[i] = 1.0 / n_samples;
    }

    int n = 10;                  // 10-point quadrature
    int iters = iterations / 10; // Fewer iterations for sample-based

    auto start = std::chrono::steady_clock::now();
    leb::lebesgue_quadrature_data<double> last_q;
    for (int i = 0; i < iters; ++i)
    {
      last_q = leb::lebesgue_quadrature_from_samples<double>(std::span(x), std::span(f), std::span(w), n);
    }
    auto end = std::chrono::steady_clock::now();

    double us_per_call = std::chrono::duration<double, std::micro>(end - start).count() / iters;
    std::println("n=10, samples={:4d}: {:8.2f} us/call", n_samples, us_per_call);

    // Print results for accuracy comparison
    if (n_samples == 1000)
    {
      std::println("\n  Direct Gram results (samples=1000):");
      std::print("  Nodes: ");
      for (auto node : last_q.nodes)
      {
        std::print("{:.10f} ", node);
      }
      std::print("\n  Weights: ");
      for (auto wt : last_q.weights)
      {
        std::print("{:.10f} ", wt);
      }
      std::println("\n");
    }
  }

  // ========================================================================
  // Lebesgue Quadrature (from moments) - matches Java approach
  // ========================================================================
  std::println("=== Lebesgue Quadrature (from moments, like Java) ===");

  for (int n_samples : sample_sizes)
  {
    // Generate sample data
    vector<double> x(n_samples), f(n_samples), w(n_samples);
    for (int i = 0; i < n_samples; ++i)
    {
      x[i] = dist(rng);
      f[i] = std::sin(x[i]);
      w[i] = 1.0 / n_samples;
    }

    int n = 10;

    // Precompute moments (this is what Java does)
    auto moments = leb::chebyshev_moments<double>(std::span(x), std::span(w), 2 * n);
    auto f_moments = leb::chebyshev_weighted_moments<double>(std::span(x), std::span(f), std::span(w), 2 * n);

    int iters = iterations; // Can do more since moments are precomputed

    auto start = std::chrono::steady_clock::now();
    leb::lebesgue_quadrature_data<double> last_q;
    for (int i = 0; i < iters; ++i)
    {
      last_q = leb::lebesgue_quadrature_from_moments<double>(std::span(f_moments), std::span(moments), n);
    }
    auto end = std::chrono::steady_clock::now();

    double us_per_call = std::chrono::duration<double, std::micro>(end - start).count() / iters;
    std::println("n=10, samples={:4d}: {:8.2f} us/call (moments precomputed)", n_samples, us_per_call);

    if (n_samples == 1000)
    {
      std::println("\n  Moments-based results (samples=1000):");
      std::print("  Nodes: ");
      for (auto node : last_q.nodes)
      {
        std::print("{:.10f} ", node);
      }
      std::print("\n  Weights: ");
      for (auto wt : last_q.weights)
      {
        std::print("{:.10f} ", wt);
      }
      std::println("");
    }
  }

  // ========================================================================
  // Parallel Implementation Comparison
  // ========================================================================
  std::println("\n=== Parallel Implementation Comparison ===");
  std::println("Using {} hardware threads\n", std::thread::hardware_concurrency());

  std::array<int, 5> parallel_sample_sizes = {1000, 5000, 10000, 25000, 50000};

  for (int n_samples : parallel_sample_sizes)
  {
    vector<double> x(n_samples), f(n_samples), w(n_samples);
    for (int i = 0; i < n_samples; ++i)
    {
      x[i] = dist(rng);
      f[i] = std::sin(x[i]);
      w[i] = 1.0 / n_samples;
    }

    int n = 10;
    int iters = std::max(10, 1000 / (n_samples / 1000));

    std::println("samples={:5d}, iters={:4d}:", n_samples, iters);

    // Sequential
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
    {
      auto q = leb::lebesgue_quadrature_from_samples<double>(std::span(x), std::span(f), std::span(w), n);
    }
    auto end = std::chrono::steady_clock::now();
    double seq_us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
    std::println("  Sequential:  {:8.1f} us", seq_us);

    // Parallel Dispatch (Auto)
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
    {
      // Force large N heuristic or call parallel directly if I exposed it?
      // Actually, let's just call the public API which should dispatch
      auto q = leb::gram_matrix_from_samples<double>(std::span(x), std::span(w), n);
    }
    end = std::chrono::steady_clock::now();
    double parallel_us = std::chrono::duration<double, std::micro>(end - start).count() / iters;
    std::println("  Parallel Dispatch: {:8.1f} us ({:.2f}x)", parallel_us, seq_us / parallel_us);

#ifdef LEB_HAS_TBB
    std::println("  (Backend: TBB enabled)");
#else
    std::println("  (Backend: jthread fallback)");
#endif
    std::println("");
  }

  return 0;
}
