/*
 *  test_lebesgue.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  test suite for Lebesgue Quadrature library.
 */

import lam.lebesgue;
namespace leb = lam::leb;
import std;
import lam.linearalgebra;

void check(bool condition, const char* message)
{
  if (!condition)
  {
    std::cerr << std::format("FAIL: {}\n", message);
    std::exit(1);
  }
  std::println("PASS: {}", message);
}

void check_close(double a, double b, double tol, const char* message)
{
  if (std::abs(a - b) > tol)
  {
    std::cerr << std::format("FAIL: {} (got {}, expected {}, diff {})\n", message, a, b, std::abs(a - b));
    std::exit(1);
  }
  std::println("PASS: {} (value: {})", message, a);
}

// Helper: Explicitly construct Lebesgue Rule for Position Operator X
// This replaces implicit Gauss calls.
leb::quadrature<double> make_lebesgue_rule(int n)
{
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
}

// ============================================================================
// Tests
// ============================================================================

void test_chebyshev_eval()
{
  std::println("\n=== Chebyshev Evaluation ===");
  // Known values: T_0(x) = 1, T_1(x) = x, T_2(x) = 2x² - 1
  check_close(leb::chebyshev_eval(0, 0.5), 1.0, 1e-14, "T_0(0.5) = 1");
  check_close(leb::chebyshev_eval(1, 0.5), 0.5, 1e-14, "T_1(0.5) = 0.5");
  check_close(leb::chebyshev_eval(2, 0.5), -0.5, 1e-14, "T_2(0.5) = 2*0.25 - 1 = -0.5");
  check_close(leb::chebyshev_eval(3, 0.5), -1.0, 1e-14, "T_3(0.5) = 4*0.125 - 1.5 = -1");
  // At x = 1: T_n(1) = 1 for all n
  for (int n = 0; n <= 10; ++n)
  {
    std::string msg = std::format("T_{}(1) = 1", n);
    check_close(leb::chebyshev_eval(n, 1.0), 1.0, 1e-14, msg.c_str());
  }
  // At x = -1: T_n(-1) = (-1)^n
  for (int n = 0; n <= 10; ++n)
  {
    double expected = (n % 2 == 0) ? 1.0 : -1.0;
    std::string msg = std::format("T_{}(-1) = {}", n, expected);
    check_close(leb::chebyshev_eval(n, -1.0), expected, 1e-14, msg.c_str());
  }
}

void test_chebyshev_sum()
{
  std::println("\n=== Chebyshev Sum (Clenshaw) ===");
  // Test: 3*T_0 + 2*T_1 - T_2 at x = 0.5
  // = 3*1 + 2*0.5 - (-0.5) = 3 + 1 + 0.5 = 4.5
  lam::vector<double> coefs = {3.0, 2.0, -1.0};
  check_close(leb::chebyshev_sum<double>(coefs, 0.5), 4.5, 1e-14, "Sum test at x=0.5");
}

void test_chebyshev_moments()
{
  std::println("\n=== Chebyshev Moments ===");
  // Single point x=0.5 with weight 1
  lam::vector<double> x = {0.5};
  lam::vector<double> w = {1.0};

  auto moments = leb::chebyshev_moments<double>(x, w, 4);

  check_close(moments[0], 1.0, 1e-14, "μ_0 = T_0(0.5) = 1");
  check_close(moments[1], 0.5, 1e-14, "μ_1 = T_1(0.5) = 0.5");
  check_close(moments[2], -0.5, 1e-14, "μ_2 = T_2(0.5) = -0.5");
}

void test_lebesgue_spatial() // was test_gauss_legendre
{
  std::println("\n=== Lebesgue Spatial Quadrature (Operator X) ===");

  for (int n = 3; n <= 7; ++n)
  {
    auto q = make_lebesgue_rule(n);
    // Total weight should be 2 (length of [-1,1])
    std::string msg_weight = std::format("n={}: total weight = 2", n);
    check_close(q.total_weight(), 2.0, 1e-10, msg_weight.c_str());
    // Integrate x^0 = constant 1 → integral = 2
    double int_1 = q.apply([](double) { return 1.0; });
    std::string msg_1 = std::format("n={}: ∫1 dx = 2", n);
    check_close(int_1, 2.0, 1e-10, msg_1.c_str());
    // Integrate x → integral = 0 (odd function)
    double int_x = q.apply([](double x) { return x; });
    std::string msg_x = std::format("n={}: ∫x dx = 0", n);
    check_close(int_x, 0.0, 1e-10, msg_x.c_str());
    // Integrate x² → integral = 2/3
    double int_x2 = q.apply([](double x) { return x * x; });
    std::string msg_x2 = std::format("n={}: ∫x² dx = 2/3", n);
    check_close(int_x2, 2.0 / 3.0, 1e-10, msg_x2.c_str());
  }
}

void test_transcendental()
{
  std::println("\n=== Transcendental Function Integration ===");

  auto q = make_lebesgue_rule(10); // 10-point rule via Lebesgue X-operator
  // ∫ sin(x) dx from -1 to 1 = 0 (odd function)
  double int_sin = q.apply([](double x) { return std::sin(x); });
  check_close(int_sin, 0.0, 1e-10, "∫sin(x) dx = 0");
  // ∫ cos(x) dx from -1 to 1 = 2*sin(1)
  double int_cos = q.apply([](double x) { return std::cos(x); });
  check_close(int_cos, 2.0 * std::sin(1.0), 1e-10, "∫cos(x) dx = 2*sin(1)");
  // ∫ exp(x) dx from -1 to 1 = e - 1/e
  double int_exp = q.apply([](double x) { return std::exp(x); });
  check_close(int_exp, std::exp(1.0) - std::exp(-1.0), 1e-10, "∫exp(x) dx = e - 1/e");
  // ∫ 1/(1+x²) dx from -1 to 1 = 2*atan(1) = π/2
  double int_rational = q.apply([](double x) { return 1.0 / (1.0 + x * x); });
  check_close(int_rational, 2.0 * std::atan(1.0), 1e-7, "∫1/(1+x²) dx = π/2");
}

void test_polynomial_exactness()
{
  std::println("\n=== Polynomial Exactness ===");
  // n-point rule should integrate x^k exactly for k = 0..2n-1 (Gauss exactness)
  for (int n = 3; n <= 6; ++n)
  {
    auto q = make_lebesgue_rule(n);

    for (int k = 0; k < 2 * n; ++k)
    {
      double exact = (k % 2 == 1) ? 0.0 : 2.0 / (k + 1);
      double approx = q.apply([k](double x) { return std::pow(x, k); });

      std::string msg = std::format("n={}: ∫x^{} dx", n, k);
      check_close(approx, exact, 1e-10, msg.c_str());
    }
  }
}

// ============================================================================
// Main
// ============================================================================

int main()
{
  std::println("===========================================");
  std::println("Lebesgue Quadrature Test Suite");
  std::println("===========================================");

  try
  {
    test_chebyshev_eval();
    test_chebyshev_sum();
    test_chebyshev_moments();
    test_lebesgue_spatial();
    test_transcendental();
    test_polynomial_exactness();

    std::println("\n===========================================");
    std::println("ALL TESTS PASSED");
    std::println("===========================================");
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << std::format("Exception: {}\n", e.what());
    return 1;
  }
}
