/*
 *  benchmark_singular.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *  Tests Lebesgue quadrature on singular weight functions.
 *  Target: Reconstruct Gauss-Chebyshev weights (1/sqrt(1-x^2)) from uniform samples.
 */

import std;
import lam.lebesgue;
import lam.linearalgebra;

namespace leb = lam::leb;
using lam::matrix;
using lam::vector;

int main()
{
  std::println("=== Singular Integral Benchmark (Chebyshev Weight) ===");
  std::println("Target Measure: dlambda(x) = 1/sqrt(1-x^2) dx on [-1, 1]");
  std::println("Exact Integral (Total Mass) = pi = 3.14159...");

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  // We want to simulate having data from a process, but we want to integrate
  // with respect to the singular Chebyshev weight.
  // Strategy: Sampling uniform x approx weight 0.5.
  // Measure: 1/sqrt(1-x^2) => set w_i = (1/sqrt(1-x_i^2)).
  // Normalize total weight to match analytical integral (PI).

  std::vector<int> sample_sizes = {1000, 10000, 100000};

  for (int n_samples : sample_sizes)
  {
    vector<double> x(n_samples), f(n_samples), w(n_samples);

    double sum_w = 0.0;
    int valid_samples = 0;

    while (valid_samples < n_samples)
    {
      double xi = dist(rng);
      double denom = std::sqrt(1.0 - xi * xi);
      if (denom < 1e-6)
        continue; // Skip near singularity to avoid inf

      x[valid_samples] = xi;
      f[valid_samples] = xi; // f(x)=x for spatial nodes
      double wi = (1.0 / denom);
      w[valid_samples] = wi;
      sum_w += wi;
      valid_samples++;
    }

    double scale = std::numbers::pi / sum_w;
    for (int i = 0; i < n_samples; ++i)
      w[i] *= scale;

    std::println("\n--- Samples M={} ---", n_samples);

    int n_nodes = 5;
    auto quad = leb::lebesgue_quadrature_from_samples<double>(std::span(x), std::span(f), std::span(w), n_nodes);

    std::println("Generated {} Nodes (Should match Gauss-Chebyshev):", n_nodes);
    // Exact nodes for N=5: cos( (2k-1)pi / 10 ) for k=1..5
    // 0.951, 0.588, 0.0, -0.588, -0.951

    std::vector<double> sorted_nodes(n_nodes);
    for (int i = 0; i < n_nodes; ++i)
      sorted_nodes[i] = quad.nodes[i];
    std::ranges::sort(sorted_nodes, std::greater<double>()); // Descending

    double max_node_err = 0.0;

    std::print("  Nodes: ");
    for (int k = 0; k < n_nodes; ++k)
    {
      double exact = std::cos((2.0 * (k + 1) - 1.0) * std::numbers::pi / (2.0 * n_nodes));
      double val = sorted_nodes[k];
      double nerr = std::abs(val - exact);
      max_node_err = std::max(max_node_err, nerr);
      std::print("{:.4f} ", val);
    }
    std::println("\n  Max Node Error: {:.2e}", max_node_err);

    // Test integration of a polynomial T_6(x) (should be exact? No, degree is 2N-1 = 9)
    // T_4(x). Integral against weight is 0 (orthogonal to T_0).
    double integral_T4 = quad.to_quadrature().apply([](double v) {
      // T_4(x) = 8x^4 - 8x^2 + 1
      return 8 * std::pow(v, 4) - 8 * std::pow(v, 2) + 1.0;
    });
    std::println("  Integral T_4(x) (Exact=0): {:.2e}", integral_T4);

    // Test Integral of T_0(x) = 1 (Exact=pi)
    double integral_1 = quad.to_quadrature().apply([](double v) { return 1.0; });
    std::println("  Integral 1 (Exact=pi):     {:.5f} (Err={:.2e})", integral_1,
                 std::abs(integral_1 - std::numbers::pi));

    if (max_node_err < 1e-2)
      std::println("  [PASS] Reconstructed Gauss-Chebyshev rule.");
    else
      std::println("  [WARN] Reconstruction poor.");
  }

  return 0;
}
