/*
 *  benchmark_accuracy.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *  Verifies numerical accuracy of Lebesgue quadrature rules.
 */

import std;
import lam.lebesgue;
import lam.linearalgebra;

namespace leb = lam::leb;
using lam::matrix;
using lam::vector;

// Test integration of x^k on [-1, 1] with w(x) = 0.5 (Uniform PDF)
// Exact integral of x^k * 0.5 dx from -1 to 1:
// = 0.5 * [x^{k+1}/(k+1)]_{-1}^{1}
// = 0.5/(k+1) * (1^{k+1} - (-1)^{k+1})
// If k is odd, term is 0.
// If k is even, term is 1 - (-1) = 2. -> 0.5/(k+1) * 2 = 1/(k+1).
double exact_moment_uniform(int k)
{
  if (k % 2 != 0)
    return 0.0;
  return 1.0 / (k + 1.0);
}

int main()
{
  std::println("=== Lebesgue Accuracy Benchmark ===");

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  // Generate samples from Uniform(-1, 1).
  // Discrete Accuracy: Does it integrate functions exactly over the *provided samples*?
  int n_samples = 100000;

  vector<double> x(n_samples);
  vector<double> w(n_samples);
  for (int i = 0; i < n_samples; ++i)
  {
    x[i] = dist(rng);
    w[i] = 1.0 / n_samples;
  }

  std::vector<int> degrees = {5, 10, 20};

  for (int n : degrees)
  {
    std::println("\n--- Degree N={} ---", n);
    // To generate spatial quadrature nodes, set f(x) = x.
    vector<double> f = x;

    auto quad = leb::lebesgue_quadrature_from_samples<double>(std::span(x), std::span(f), std::span(w), n);

    // Check 1: Discrete Moment Matching
    // The rule should match the sample moments up to degree 2N-1
    double max_err = 0.0;
    for (int k = 0; k < 2 * n; ++k)
    {
      // "Exact" (Sample Moment)
      double discrete_moment = 0.0;
      for (int i = 0; i < n_samples; ++i)
      {
        double val = std::pow(x[i], k);
        discrete_moment += w[i] * val;
      }

      // Quadrature Estimate
      double quad_moment = 0.0;
      for (std::size_t j = 0; j < quad.nodes.size(); ++j)
      {
        quad_moment += quad.weights[j] * std::pow(quad.nodes[j], k);
      }

      double err = std::abs(discrete_moment - quad_moment);
      max_err = std::max(max_err, err);

      if (k <= 2)
      {
        std::println("  k={:<2}: Exact={:<12.5e} Quad={:<12.5e} Err={:.2e}", k, discrete_moment, quad_moment, err);
      }

      // std::println("  k={:<2}: Err={:.2e}", k, err);
    }
    std::println("  Max Polynomial Error (Discrete, k < 2N): {:.2e}", max_err);
    if (max_err < 1e-12)
      std::println("  [PASS] Discrete moments matched.");
    else
      std::println("  [WARN] Discrete moments error high?");
  }

  return 0;
}
