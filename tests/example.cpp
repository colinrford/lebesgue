/*
 *  example.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  example usage of the Lebesgue Quadrature library.
 */

import lam.lebesgue;
namespace leb = lam::leb;
import std;
import lam.linearalgebra; // for vector

int main()
{
  // Helper to generate explicit Lebesgue rule (Spectral Position Operator)
  auto make_lebesgue_rule = [](int n) {
    // 1. Base Moments (Legendre)
    int n_moments = 2 * n;
    lam::vector<double> moments(n_moments);
    for (int k = 0; k < n_moments; ++k)
    {
      if (k == 0)
        moments[k] = 2.0;
      else if (k % 2 != 0)
        moments[k] = 0.0;
      else
        moments[k] = 2.0 / (1.0 - (double)(k * k));
    }

    // 2. Explicit Position Operator Moments (nu_k)
    // x T_k = 0.5 (T_{k+1} + T_{|k-1|})
    lam::vector<double> x_moments(n_moments);
    for (int k = 0; k < n_moments; ++k)
    {
      double m_plus = (k + 1 < n_moments) ? moments[k + 1] : 0.0;
      double m_minus = moments[std::abs(k - 1)];
      x_moments[k] = 0.5 * (m_plus + m_minus);
    }

    // 3. Solve Spectral Problem for Operator X
    auto rule_data = leb::lebesgue_quadrature_from_moments<double>(x_moments.as_span(), moments.as_span(), n);
    return rule_data.to_quadrature();
  };

  // ========================================================================
  // Example 1: Explicit Lebesgue Quadrature (Target: Position Operator)
  // ========================================================================
  std::println("=== Explicit Lebesgue Quadrature (Target: Position Operator X) ===\n");

  for (int n : {3, 5, 7, 10})
  {
    auto q = make_lebesgue_rule(n);

    std::println("{}-point Spatial Rule:", n);
    std::print("  Nodes (Eigenvalues of X): ");
    for (auto node : q.nodes)
      std::print("{} ", node);
    std::print("\n  Weights: ");
    for (auto w : q.weights)
      std::print("{} ", w);
    std::println("\n");

    // Test integrals
    double int_1 = q.apply([](double) { return 1.0; });
    double int_x2 = q.apply([](double x) { return x * x; });
    double int_exp = q.apply([](double x) { return std::exp(x); });

    std::println("  ∫1 dx = {} (exact: 2)", int_1);
    std::println("  ∫x² dx = {} (exact: 0.666...)", int_x2);
    std::println("  ∫exp(x) dx = {} (exact: {})", int_exp, std::exp(1.0) - std::exp(-1.0));
    std::println("");
  }

  // ========================================================================
  // Example 2: Lebesgue quadrature from samples
  // ========================================================================
  std::println("=== Lebesgue Quadrature from Samples ===\n");

  // Generate sample data: uniform random x in [-1,1], f(x) = sin(x)
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  const std::size_t n_samples = 100;
  lam::vector<double> x(n_samples);
  lam::vector<double> f(n_samples);
  lam::vector<double> w(n_samples);

  for (std::size_t i = 0; i < n_samples; ++i)
  {
    x[i] = dist(rng);
    f[i] = std::sin(x[i]);
    w[i] = 1.0 / n_samples; // Uniform weights
  }

  // Build 5-point Lebesgue quadrature
  auto leb_data = leb::lebesgue_quadrature_from_samples<double>(x, f, w, 5);

  std::println("5-point Lebesgue quadrature (f = sin(x)):");
  std::print("  Value-nodes (Eigenvalues of Sin(X)): ");
  for (auto node : leb_data.nodes)
    std::print("{} ", node);
  std::print("\n  Weights: ");
  for (auto wt : leb_data.weights)
    std::print("{} ", wt);
  std::println("\n");

  // Evaluate Radon-Nikodym at various points
  std::println("Radon-Nikodym derivative estimates:");
  for (double x0 : {-0.5, 0.0, 0.5})
  {
    auto rn = leb_data.evaluate_at(x0);
    std::println("  RN({}) = {}  (sin({}) = {})", x0, rn.radon_nikodym, x0, std::sin(x0));
  }

  std::println("\nDone.");
  return 0;
}
