/*
 *  benchmark_nttp.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  benchmark comparing runtime Chebyshev evaluation vs compile-time generated polynomial_nttp.
 */

import std;
import lam.lebesgue;
import lam.linearalgebra;
import lam.polynomial_nttp;

namespace leb = lam::leb; // for leb::chebyshev_eval

using namespace lam::polynomial;

// ============================================================================
// Compile-time Chebyshev Generation (Memoized via variable template)
// ============================================================================

// ============================================================================
// Compile-time Chebyshev Generation (Memoized via struct template)
// ============================================================================

template<typename R, std::size_t N>
struct chebyshev_t_memo
{
  static constexpr auto value = []() {
    if constexpr (N == 0) // T_0(x) = 1
      return lam::polynomial_nttp<R, 0>{{R(1)}};
    else if constexpr (N == 1) // T_1(x) = x
      return lam::polynomial_nttp<R, 1>{{R(0), R(1)}};
    else
    {
      // Recursive case
      constexpr auto T_nm1 = chebyshev_t_memo<R, N - 1>::value;
      constexpr auto T_nm2 = chebyshev_t_memo<R, N - 2>::value;

      // x as a polynomial
      constexpr auto x = lam::make_monomial<R, 1>();

      // T_{n+1} = 2x T_n - T_{n-1}
      constexpr R two = R(2);
      return two * x * T_nm1 - T_nm2;
    }
  }();
};

template<typename R, std::size_t N>
constexpr auto chebyshev_t_n()
{ return chebyshev_t_memo<R, N>::value; }

// Helper to evaluate at runtime for a dynamic N using a switch of compiled polynomials
template<typename R, std::size_t MaxN>
double eval_nttp(std::size_t n, double x)
{
  return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    double result = 0.0;
    ((n == Is ? (result = chebyshev_t_n<R, Is>()(x), true) : false) || ...);
    return result;
  }(std::make_index_sequence<MaxN + 1>{});
}

// ============================================================================
// Hierarchical Chebyshev Evaluation: T_{mn}(x) = T_m(T_n(x))
// Uses composition identity to avoid unstable high-degree coefficient expansion.
// ============================================================================

// Compile-time factorization helper
template<std::size_t N>
struct chebyshev_factors
{
  // Find best factorization: prefer balanced factors, max factor <= 15
  static constexpr std::size_t find_factor()
  {
    if constexpr (N <= 15)
      return 1; // No factorization needed

    // Try to find a factor that keeps both parts <= 15
    for (std::size_t f = 15; f >= 2; --f)
    {
      if (N % f == 0 && N / f <= 15)
        return f;
    }
    // Fallback: find any factor <= 15
    for (std::size_t f = 15; f >= 2; --f)
    {
      if (N % f == 0)
        return f;
    }
    return 1; // Prime, no factorization
  }

  static constexpr std::size_t outer = find_factor();
  static constexpr std::size_t inner = (outer > 1) ? N / outer : N;
  static constexpr bool is_composite = (outer > 1);
};

// Hierarchical evaluation via composition
template<typename R, std::size_t N>
double eval_hierarchical(double x)
{
  using factors = chebyshev_factors<N>;

  if constexpr (N <= 15)
  {
    // Small degree: use direct NTTP evaluation (stable)
    return chebyshev_t_n<R, N>()(x);
  }
  else if constexpr (factors::is_composite)
  {
    // Composite: T_{outer*inner}(x) = T_outer(T_inner(x))
    double inner_val = eval_hierarchical<R, factors::inner>(x);
    return eval_hierarchical<R, factors::outer>(inner_val);
  }
  else
  {
    // Prime or unfactorizable: fall back to recurrence
    // (Could also use NTTP here but it's unstable for N > 15)
    return leb::chebyshev_eval(static_cast<int>(N), x);
  }
}

int main()
{
  std::cout << std::fixed << std::setprecision(4);

  std::cout << "Verifying T_3(0.5): ";
  constexpr auto T3 = chebyshev_t_n<double, 3>();
  // T_3(x) = 4x^3 - 3x. T_3(0.5) = 4(0.125) - 3(0.5) = 0.5 - 1.5 = -1.0
  double val_nttp = T3(0.5);
  double val_rt = leb::chebyshev_eval(3, 0.5);
  std::cout << "NTTP=" << val_nttp << ", Runtime=" << val_rt
            << (std::abs(val_nttp - val_rt) < 1e-10 ? " [MATCH]" : " [FAIL]") << "\n\n";
  // ==========================================================================
  // Benchmark
  // ==========================================================================
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  constexpr std::size_t n_samples = 1000000;
  lam::vector<double> inputs(n_samples);
  for (auto& x : inputs)
    x = dist(rng);

  auto run_benchmark = [&](std::string name, auto func) {
    auto start = std::chrono::steady_clock::now();
    double sum = 0.0;
    for (double x : inputs)
      sum += func(x);
    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << std::format("{:<25} : {:8.2f} ms (sum={:.2f})\n", name, ms, sum);
  };

  std::cout << "=== Benchmarking Evaluation of T_10(x) (1M calls) ===\n";
  // 1. Runtime Recurrence
  run_benchmark("Runtime (chebyshev_eval)", [](double x) { return leb::chebyshev_eval(10, x); });
  // 2. Polynomial NTTP (Direct Horner)
  constexpr auto T10 = chebyshev_t_n<double, 10>();
  run_benchmark("NTTP (Horner)", [=](double x) { return T10(x); });
  // 3. Hand-unrolled Horner (to check compiler opt of NTTP)
  // T_10 coeffs roughly... 512x^10 ...
  // This is what NTTP essentially does.
  std::cout << "\n=== Benchmarking Evaluation of T_20(x) (1M calls) ===\n";
  run_benchmark("Runtime (chebyshev_eval)", [](double x) { return leb::chebyshev_eval(20, x); });

  constexpr auto T20 = chebyshev_t_n<double, 20>();
  run_benchmark("NTTP (Horner)", [=](double x) { return T20(x); });

  std::cout << "\n=== Benchmarking Evaluation of T_30(x) (1M calls) ===\n";
  run_benchmark("Runtime (chebyshev_eval)", [](double x) { return leb::chebyshev_eval(30, x); });

  constexpr auto T30 = chebyshev_t_n<double, 30>();
  constexpr auto T29 = chebyshev_t_n<double, 29>();

  run_benchmark("NTTP (Horner)", [=](double x) { return T30(x); });

  std::cout << "\n=== Benchmarking Evaluation of T_50(x) (1M calls) ===\n";

  // 1. Pure Runtime Recurrence (Baseline)
  run_benchmark("Runtime (chebyshev_eval)", [](double x) { return leb::chebyshev_eval(50, x); });
  std::cout << "\n=== Accuracy Benchmark (Max Error vs Recurrence) ===\n";
  auto check_accuracy = [&](std::string name, int n, auto func) {
    double max_err = 0.0;
    for (double x : inputs)
    {
      double val_ref = leb::chebyshev_eval(n, x);
      double val_test = func(x);
      max_err = std::max(max_err, std::abs(val_test - val_ref));
    }
    std::cout << std::format("Degree {:<2} {:<20}: Max Error = {:.2e}\n", n, name, max_err);
  };

  check_accuracy("NTTP", 10, [&](double x) { return chebyshev_t_n<double, 10>()(x); });
  check_accuracy("NTTP", 20, [&](double x) { return chebyshev_t_n<double, 20>()(x); });
  check_accuracy("NTTP", 30, [&](double x) { return chebyshev_t_n<double, 30>()(x); });

  // N=40
  constexpr auto T40 = chebyshev_t_n<double, 40>();
  check_accuracy("NTTP", 40, [&](double x) { return T40(x); });

  // N=50
  constexpr auto T50 = chebyshev_t_n<double, 50>();
  check_accuracy("NTTP", 50, [&](double x) { return T50(x); });

  // ============================================================================
  // Hierarchical Accuracy Benchmark
  // ============================================================================
  std::cout << "\n=== Hierarchical Accuracy (Max Error vs Recurrence) ===\n";
  std::cout << "Uses T_{mn}(x) = T_m(T_n(x)) composition for stability\n\n";

  // Show factorizations being used
  auto print_factors = []<std::size_t N>() {
    using F = chebyshev_factors<N>;
    if constexpr (F::is_composite)
      std::cout << std::format("  T_{} = T_{}(T_{}(x))\n", N, F::outer, F::inner);
    else
      std::cout << std::format("  T_{} = direct (N <= 15 or prime)\n", N);
  };

  std::cout << "Factorizations:\n";
  print_factors.template operator()<20>();
  print_factors.template operator()<30>();
  print_factors.template operator()<40>();
  print_factors.template operator()<50>();
  print_factors.template operator()<60>();
  print_factors.template operator()<100>();
  std::cout << "\n";

  check_accuracy("Hierarchical", 20, [](double x) { return eval_hierarchical<double, 20>(x); });
  check_accuracy("Hierarchical", 30, [](double x) { return eval_hierarchical<double, 30>(x); });
  check_accuracy("Hierarchical", 40, [](double x) { return eval_hierarchical<double, 40>(x); });
  check_accuracy("Hierarchical", 50, [](double x) { return eval_hierarchical<double, 50>(x); });
  check_accuracy("Hierarchical", 60, [](double x) { return eval_hierarchical<double, 60>(x); });
  check_accuracy("Hierarchical", 100, [](double x) { return eval_hierarchical<double, 100>(x); });

  return 0;
}
