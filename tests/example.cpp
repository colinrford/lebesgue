/* 
 *  example.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  example usage of the Lebesgue Quadrature library.
 */

import lam.lebesgue;
namespace leb = lam::leb;
import std;

int main()
{
  // ========================================================================
  // Example 1: Gauss-Legendre quadrature for interval [-1, 1]
  // ========================================================================
  std::println("=== Gauss-Legendre Quadrature ===\n");

  for (int n : {3, 5, 7, 10})
  {
    auto q = leb::gauss_legendre(n);

    std::println("{}-point Gauss-Legendre rule:", n);
    std::print("  Nodes: ");
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
  std::vector<double> x(n_samples), f(n_samples), w(n_samples);

  for (std::size_t i = 0; i < n_samples; ++i)
  {
    x[i] = dist(rng);
    f[i] = std::sin(x[i]);
    w[i] = 1.0 / n_samples; // Uniform weights
  }

  // Build 5-point Lebesgue quadrature
  auto leb_data = leb::lebesgue_quadrature_from_samples<double>(x, f, w, 5);

  std::println("5-point Lebesgue quadrature (f = sin(x)):");
  std::print("  Value-nodes (f-values): ");
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
