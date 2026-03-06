/*
 *  benchmark_policy.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  Compares chebyshev_policy (runtime recurrence) vs nttp_hierarchical_policy
 *  (compile-time coefficients with hierarchical composition).
 */

import std;
import lam.lebesgue;
import lam.linearalgebra;

namespace leb = lam::leb;

int main()
{
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "=== Lebesgue Policy Comparison ===\n\n";

  // Test points
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  constexpr std::size_t n_samples = 1000000;
  lam::vector<double> inputs(n_samples);
  for (auto& x : inputs)
    x = dist(rng);

  // ============================================================================
  // Accuracy Comparison: Chebyshev Evaluation
  // ============================================================================
  std::cout << "--- Accuracy: chebyshev_eval vs nttp_hierarchical_policy ---\n";
  std::cout << "Using runtime recurrence as reference truth.\n\n";

  auto check_accuracy = [&]<std::size_t N>(std::string name) {
    double max_err = 0.0;
    for (double x : inputs)
    {
      double val_ref = leb::chebyshev_eval(static_cast<int>(N), x);
      double val_nttp = leb::nttp_hierarchical_policy::evaluate<N>(x);
      max_err = std::max(max_err, std::abs(val_nttp - val_ref));
    }
    std::cout << std::format("  Degree {:>3} : Max Error = {:.2e}\n", N, max_err);
    return max_err;
  };

  check_accuracy.template operator()<5>("T_5");
  check_accuracy.template operator()<10>("T_10");
  check_accuracy.template operator()<20>("T_20");
  check_accuracy.template operator()<30>("T_30");
  check_accuracy.template operator()<40>("T_40");
  check_accuracy.template operator()<60>("T_60");
  check_accuracy.template operator()<100>("T_100");

  // ============================================================================
  // Performance Comparison: Evaluation Speed
  // ============================================================================
  std::cout << "\n--- Performance: 1M evaluations ---\n";

  auto benchmark = [&]<std::size_t N>() {
    // Runtime recurrence
    auto start1 = std::chrono::steady_clock::now();
    double sum1 = 0.0;
    for (double x : inputs)
      sum1 += leb::chebyshev_eval(static_cast<int>(N), x);
    auto end1 = std::chrono::steady_clock::now();
    double ms1 = std::chrono::duration<double, std::milli>(end1 - start1).count();

    // NTTP hierarchical
    auto start2 = std::chrono::steady_clock::now();
    double sum2 = 0.0;
    for (double x : inputs)
      sum2 += leb::nttp_hierarchical_policy::evaluate<N>(x);
    auto end2 = std::chrono::steady_clock::now();
    double ms2 = std::chrono::duration<double, std::milli>(end2 - start2).count();

    double speedup = ms1 / ms2;
    std::cout << std::format("  Degree {:>3} : Recurrence {:6.2f}ms, NTTP {:6.2f}ms ({:.2f}x)\n", N, ms1, ms2, speedup);
  };

  benchmark.template operator()<10>();
  benchmark.template operator()<20>();
  benchmark.template operator()<40>();
  benchmark.template operator()<60>();
  benchmark.template operator()<100>();

  std::cout << "\n--- Note ---\n";
  std::cout << "For Gram matrix construction (lebesgue_quadrature_from_samples),\n";
  std::cout << "both policies use the same evaluate_recurrence, so results are identical.\n";
  std::cout << "The nttp_hierarchical_policy::evaluate<N>() is for direct point evaluation.\n";

  std::cout << "\nDone.\n";
  return 0;
}
