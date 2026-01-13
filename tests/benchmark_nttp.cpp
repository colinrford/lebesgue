
/* 
 *  benchmark_nttp.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  benchmark comparing runtime Chebyshev evaluation vs compile-time generated polynomial_nttp.
 */
import std;
import lam.lebesgue;
namespace leb = lam::leb; // for leb::chebyshev_eval
import lam.polynomial_nttp;

using namespace lam::polynomial;

// ============================================================================
// Compile-time Chebyshev Generation (Memoized via variable template)
// ============================================================================

// Variable template: each chebyshev<N> is computed ONCE and cached
template<std::size_t N>
inline constexpr auto chebyshev = []() {
  if constexpr (N == 0)
    return polynomial_nttp<double, 0>{1.0};
  else if constexpr (N == 1)
    return lam::make_monomial<double, 1>();
  else
    return 2.0 * lam::make_monomial<double, 1>() * chebyshev<N - 1> - chebyshev<N - 2>;
}();

// Helper to evaluate at runtime for a dynamic N using a switch of compiled polynomials
template<typename R, std::size_t MaxN>
double eval_nttp(std::size_t n, double x)
{
  return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    double result = 0.0;
    ((n == Is ? (result = chebyshev<Is>(x), true) : false) || ...);
    return result;
  }(std::make_index_sequence<MaxN + 1>{});
}

int main()
{
  std::cout << std::fixed << std::setprecision(4);

  std::cout << "Verifying T_3(0.5): ";
  constexpr auto T3 = chebyshev<3>;
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
  std::vector<double> inputs(n_samples);
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
  constexpr auto T10 = chebyshev<10>;
  run_benchmark("NTTP (Horner)", [=](double x) { return T10(x); });
  // 3. Hand-unrolled Horner (to check compiler opt of NTTP)
  // T_10 coeffs roughly... 512x^10 ...
  // This is what NTTP essentially does.
  std::cout << "\n=== Benchmarking Evaluation of T_20(x) (1M calls) ===\n";
  run_benchmark("Runtime (chebyshev_eval)", [](double x) { return leb::chebyshev_eval(20, x); });

  constexpr auto T20 = chebyshev<20>;
  run_benchmark("NTTP (Horner)", [=](double x) { return T20(x); });

  std::cout << "\n=== Benchmarking Evaluation of T_30(x) (1M calls) ===\n";
  run_benchmark("Runtime (chebyshev_eval)", [](double x) { return leb::chebyshev_eval(30, x); });

  constexpr auto T30 = chebyshev<30>;
  constexpr auto T29 = chebyshev<29>;

  run_benchmark("NTTP (Horner)", [=](double x) { return T30(x); });

  std::cout << "\n=== Benchmarking Evaluation of T_50(x) (1M calls) ===\n";

  // 1. Pure Runtime Recurrence (Baseline)
  run_benchmark("Runtime (chebyshev_eval)", [](double x) { return leb::chebyshev_eval(50, x); });
  // 2. Hybrid: T30 (Monomial) -> Recur to T50
  run_benchmark("Hybrid (T30 Monomial + Recur)", [=](double x) {
    double t_prev2 = T29(x);
    double t_prev1 = T30(x);
    for (int k = 31; k <= 50; ++k)
    {
      double t_curr = 2 * x * t_prev1 - t_prev2;
      t_prev2 = t_prev1;
      t_prev1 = t_curr;
    }
    return t_prev1;
  });

  return 0;
}
