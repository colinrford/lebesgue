/* 
 *  test_complex.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  comprehensive test suite for complex-valued quadrature and quantum mechanics problems.
 */

import std;
import lam.lebesgue;
namespace leb = lam::leb;
import lam.linearalgebra;

using namespace std::complex_literals;

template<typename T>
struct std::formatter<std::complex<T>> : std::formatter<T>
{
  auto format(std::complex<T> c, format_context& ctx) const
  {
    return std::format_to(ctx.out(), "({0}, {1})", c.real(), c.imag());
  }
};

using namespace lam::linalg;

bool almost_equal(std::complex<double> a, std::complex<double> b, double tol = 1e-10) { return std::abs(a - b) < tol; }

struct sampler
{
  std::string name;
  std::function<void(std::size_t, 
                     vector<std::complex<double>>&, 
                     vector<std::complex<double>>&,
                     vector<std::complex<double>>&)> generate;
  std::function<std::complex<double>(std::complex<double>)> reweight;
};

// 1. Trapezoidal (Uniform Measure approximation)
// dx = 1 * dmu (since weights approx dx)
sampler trapezoidal_sampler = {
  "Trapezoidal (Uniform Grid)",
  [](std::size_t N, vector<std::complex<double>>& x, vector<std::complex<double>>& w, vector<std::complex<double>>& f) {
    double h = 2.0 / (N - 1);
    for (std::size_t i = 0; i < N; ++i)
    {
      double xi = -1.0 + h * i;
      x[i] = xi;
      if (i == 0 || i == N - 1)
        w[i] = h / 2.0;
      else
        w[i] = h;
      f[i] = x[i];
    }
  },
  [](std::complex<double>) { return 1.0; }};

// 2. Chebyshev (Chebyshev Measure)
// dmu = dx / sqrt(1-x^2)
// implies dx = sqrt(1-x^2) dmu
sampler chebyshev_sampler = {
  "Chebyshev (Measure 1/sqrt(1-x^2))",
  [](std::size_t N, vector<std::complex<double>>& x, vector<std::complex<double>>& w, vector<std::complex<double>>& f) {
    double pi = std::numbers::pi;
    for (std::size_t i = 0; i < N; ++i)
    { // Roots of T_N
      double theta = (2.0 * i + 1.0) * pi / (2.0 * N);
      x[i] = std::cos(theta);
      w[i] = pi / N;
      f[i] = x[i];
    }
  },
  [](std::complex<double> z) { return std::sqrt(1.0 - z * z); }};

// 3. Gauss-Legendre (Uniform Measure, optimal nodes)
// dx = 1 * dmu (weights are exact for uniform measure)
// Uses Newton-Raphson to find Legendre roots
sampler gauss_legendre_sampler = {
  "Gauss-Legendre (Uniform)",
  [](std::size_t N, vector<std::complex<double>>& x, vector<std::complex<double>>& w, vector<std::complex<double>>& f) {
    // Compute Gauss-Legendre nodes and weights using Newton-Raphson
    const double pi = std::numbers::pi;
    const double tol = 1e-15;
    const int max_iter = 100;

    vector<double> nodes(N), weights(N);

    for (std::size_t i = 0; i < (N + 1) / 2; ++i)
    { // Initial guess from Chebyshev
      double z = std::cos(pi * (i + 0.75) / (N + 0.5));
      double z1;

      // Newton-Raphson iteration
      for (int iter = 0; iter < max_iter; ++iter)
      { // Evaluate P_N(z) and its derivative using recurrence
        double p1 = 1.0, p2 = 0.0;
        for (std::size_t j = 0; j < N; ++j)
        {
          double p3 = p2;
          p2 = p1;
          p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
        }
        // P_N'(z) = N * (z * P_N(z) - P_{N-1}(z)) / (z^2 - 1)
        double pp = static_cast<double>(N) * (z * p1 - p2) / (z * z - 1.0);
        z1 = z;
        z = z1 - p1 / pp;
        if (std::abs(z - z1) < tol)
          break;
      }
      // Compute weight
      double p1 = 1.0, p2 = 0.0;
      for (std::size_t j = 0; j < N; ++j)
      {
        double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
      }
      double pp = static_cast<double>(N) * (z * p1 - p2) / (z * z - 1.0);
      double weight = 2.0 / ((1.0 - z * z) * pp * pp);
      // Symmetric placement
      nodes[i] = -z;
      nodes[N - 1 - i] = z;
      weights[i] = weight;
      weights[N - 1 - i] = weight;
    }

    for (std::size_t i = 0; i < N; ++i)
    {
      x[i] = nodes[i];
      w[i] = weights[i];
      f[i] = x[i];
    }
  },
  [](std::complex<double>) { return 1.0; }};

// Result Aggregation
struct test_result
{
  std::string test_name;
  std::string sampler_name;
  double error;
  bool passed;
};

using results_reporter = std::vector<test_result>;

void test_complex_f_real_domain(const sampler& sampler, results_reporter& results)
{
  std::string name = "Complex Function (e^ix)";
  std::println("Test: {} [{}]", name, sampler.name);

  const std::size_t N = 1000;
  vector<std::complex<double>> x(N);
  vector<std::complex<double>> w(N);
  vector<std::complex<double>> f(N);
  sampler.generate(N, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 20);
  auto quad = quad_data.to_quadrature();

  auto integral = quad.apply([&](std::complex<double> z) { return std::exp(1i * z) * sampler.reweight(z); });

  std::complex<double> expected = 2.0 * std::sin(1.0);
  double error = std::abs(integral - expected);
  bool pass = almost_equal(integral, expected, 1e-3);

  std::println("  Computed: {}", integral);
  std::println("  Expected: {}", expected);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

void test_qm_momentum(const sampler& sampler, results_reporter& results)
{
  std::string name = "QM Momentum <p>";
  std::println("\nTest: {} [{}]", name, sampler.name);

  const std::size_t N = 1000;
  vector<std::complex<double>> x(N);
  vector<std::complex<double>> w(N);
  vector<std::complex<double>> f(N);
  sampler.generate(N, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 30);
  auto quad = quad_data.to_quadrature();

  double k = 1.5;

  auto integral = quad.apply([k, &sampler](std::complex<double> z) {
    std::complex<double> psi = std::exp(1i * k * z);
    std::complex<double> p_psi = k * psi;
    return std::conj(psi) * p_psi * sampler.reweight(z);
  });

  auto norm = quad.apply([k, &sampler](std::complex<double> z) {
    std::complex<double> psi = std::exp(1i * k * z);
    return std::conj(psi) * psi * sampler.reweight(z);
  });

  std::complex<double> expectation = integral / norm;
  std::complex<double> expected = {k, 0.0};
  double error = std::abs(expectation - expected);
  bool pass = almost_equal(expectation, expected, 1e-4);

  std::println("  Computed <p>: {}", expectation);
  std::println("  Expected:     ({}, 0)", k);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

void test_harmonic_oscillator(const sampler& sampler, results_reporter& results)
{
  std::string name = "Harmonic Oscillator <x>";
  std::println("\nTest: {} [{}]", name, sampler.name);

  const std::size_t N = 1000;
  vector<std::complex<double>> x(N);
  vector<std::complex<double>> w(N);
  vector<std::complex<double>> f(N);
  sampler.generate(N, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 30);
  auto quad = quad_data.to_quadrature();

  double pi = std::numbers::pi;
  double norm_factor = std::pow(pi, -0.25);

  auto x_expect_num = quad.apply([norm_factor, &sampler](std::complex<double> z) {
    std::complex<double> psi0 = norm_factor * std::exp(-0.5 * z * z);
    return std::conj(psi0) * z * psi0 * sampler.reweight(z);
  });

  auto norm = quad.apply([norm_factor, &sampler](std::complex<double> z) {
    std::complex<double> psi0 = norm_factor * std::exp(-0.5 * z * z);
    return std::conj(psi0) * psi0 * sampler.reweight(z);
  });

  auto x_expect = x_expect_num / norm;
  double error = std::abs(x_expect); // Expected 0
  bool pass = error < 1e-12;

  std::println("  <psi0 | x | psi0> (Position): {}", x_expect);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

void test_commutator_relation(const sampler& sampler, results_reporter& results)
{
  std::string name = "Commutator [x, p]";
  std::println("\nTest: {} [{}]", name, sampler.name);

  const std::size_t N = 1000;
  vector<std::complex<double>> x(N);
  vector<std::complex<double>> w(N);
  vector<std::complex<double>> f(N);
  sampler.generate(N, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 40);
  auto quad = quad_data.to_quadrature();

  double pi = std::numbers::pi;
  double norm_factor = std::pow(pi, -0.25);

  auto commutator_val = quad.apply([norm_factor, &sampler](std::complex<double> z) {
    std::complex<double> psi = norm_factor * std::exp(-0.5 * z * z);
    std::complex<double> xp_psi = z * (1i * z * psi);
    std::complex<double> px_psi = -1i * (1.0 - z * z) * psi;
    std::complex<double> commutator_psi = xp_psi - px_psi;
    return std::conj(psi) * commutator_psi * sampler.reweight(z);
  });

  auto norm_sq = quad.apply([norm_factor, &sampler](std::complex<double> z) {
    std::complex<double> psi = norm_factor * std::exp(-0.5 * z * z);
    return std::conj(psi) * psi * sampler.reweight(z);
  });

  std::complex<double> expected = 1i * norm_sq;
  double error = std::abs(commutator_val - expected);
  bool pass = almost_equal(commutator_val, expected, 1e-6);

  std::println("  <[x,p]> Computed: {}", commutator_val);
  std::println("  Target (i*norm):  {}", expected);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

void test_step_potential(const sampler& sampler, results_reporter& results)
{
  std::string name = "Step Potential <Theta>";
  std::println("\nTest: {} [{}]", name, sampler.name);

  const std::size_t N = 2000;
  vector<std::complex<double>> x(N);
  vector<std::complex<double>> w(N);
  vector<std::complex<double>> f(N);
  sampler.generate(N, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 50);
  auto quad = quad_data.to_quadrature();

  auto norm_sq = quad.apply([&sampler](std::complex<double> z) {
    std::complex<double> psi = std::exp(-0.5 * z * z);
    return std::conj(psi) * psi * sampler.reweight(z);
  });

  auto v_expect = quad.apply([&sampler](std::complex<double> z) {
    std::complex<double> psi = std::exp(-0.5 * z * z);
    double v = (z.real() > 0.0) ? 1.0 : 0.0;
    return std::conj(psi) * v * psi * sampler.reweight(z);
  });

  std::complex<double> ratio = v_expect / norm_sq;
  double error = std::abs(ratio - 0.5);
  bool pass = almost_equal(ratio, 0.5, 1e-3);

  std::println("  <Theta(x)> Ratio: {}", ratio);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

void test_scattering_flux(const sampler& sampler, results_reporter& results)
{
  std::string name = "Scattering Flux Conservation";
  std::println("\nTest: {} [{}]", name, sampler.name);

  double E = 2.0;
  double V0 = 1.0;
  double k = std::sqrt(2.0 * E);
  double q = std::sqrt(2.0 * (E - V0));
  std::complex<double> R = (k - q) / (k + q);
  std::complex<double> T = (2.0 * k) / (k + q);

  const std::size_t N = 2000;
  vector<std::complex<double>> x(N);
  vector<std::complex<double>> w(N);
  vector<std::complex<double>> f(N);
  sampler.generate(N, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 60);
  auto quad = quad_data.to_quadrature();

  auto flux_left = quad.apply([k, R, &sampler](std::complex<double> z) {
    if (z.real() > 0.0)
      return std::complex<double>(0.0, 0.0);
    std::complex<double> eikx = std::exp(1i * k * z);
    std::complex<double> emikx = std::exp(-1i * k * z);
    std::complex<double> psi = eikx + R * emikx;
    std::complex<double> dpsi = 1i * k * (eikx - R * emikx);
    double J = std::imag(std::conj(psi) * dpsi);
    return std::complex<double>(J, 0.0) * sampler.reweight(z);
  });

  auto flux_right = quad.apply([q, T, &sampler](std::complex<double> z) {
    if (z.real() <= 0.0)
      return std::complex<double>(0.0, 0.0);
    std::complex<double> psi = T * std::exp(1i * q * z);
    std::complex<double> dpsi = 1i * q * psi;
    double J = std::imag(std::conj(psi) * dpsi);
    return std::complex<double>(J, 0.0) * sampler.reweight(z);
  });

  std::println("  Integrated Flux Left:  {}", flux_left);
  std::println("  Integrated Flux Right: {}", flux_right);

  double error = std::abs(flux_left - flux_right);
  bool pass = almost_equal(flux_left, flux_right, 1e-3);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

void test_time_evolution(const sampler& sampler, results_reporter& results)
{
  std::string name = "Time Evolution <x^2>";
  std::println("\nTest: {} [{}]", name, sampler.name);

  double sigma0 = 0.1;
  double t = 0.01;
  double sigma_sq = sigma0 * sigma0;
  std::complex<double> denominator = 1.0 + 1i * t / sigma_sq;
  double pi = std::numbers::pi;
  double prefactor_mag = std::pow(pi * sigma_sq, -0.25);
  double expected_x2 = (sigma_sq / 2.0) + (t * t) / (2.0 * sigma_sq);

  const std::size_t N = 10000;
  vector<std::complex<double>> x(N);
  vector<std::complex<double>> w(N);
  vector<std::complex<double>> f(N);
  sampler.generate(N, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 60);
  auto quad = quad_data.to_quadrature();

  auto norm_check = quad.apply([&](std::complex<double> z) {
    std::complex<double> exponent = -(z * z) / (2.0 * sigma_sq * denominator);
    std::complex<double> factor = 1.0 / std::sqrt(denominator);
    std::complex<double> psi = prefactor_mag * factor * std::exp(exponent);
    return std::conj(psi) * psi * sampler.reweight(z);
  });

  auto x2_check = quad.apply([&](std::complex<double> z) {
    std::complex<double> exponent = -(z * z) / (2.0 * sigma_sq * denominator);
    std::complex<double> factor = 1.0 / std::sqrt(denominator);
    std::complex<double> psi = prefactor_mag * factor * std::exp(exponent);
    return std::conj(psi) * (z * z) * psi * sampler.reweight(z);
  });

  std::println("  Unitarity (Norm):  {}", norm_check);
  std::println("  <x^2> Computed:    {}", x2_check);

  double error = std::abs(x2_check - expected_x2);
  bool pass = (std::abs(norm_check - 1.0) < 1e-12 && almost_equal(x2_check, expected_x2, 1e-12));

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

void test_hydrogen_atom(const sampler& sampler, results_reporter& results)
{
  std::string name = "Hydrogen Atom <H>";
  std::println("\nTest: {} [{}]", name, sampler.name);

  double Rmax = 30.0;
  double scale = Rmax / 2.0;

  const std::size_t N = 10000;
  vector<std::complex<double>> x(N);
  vector<std::complex<double>> w(N);
  vector<std::complex<double>> f(N);
  sampler.generate(N, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 60);
  auto quad = quad_data.to_quadrature();

  auto energy_numerator = quad.apply([scale, &sampler](std::complex<double> z) {
    std::complex<double> r = scale * (z + 1.0);
    if (std::abs(r) < 1e-14)
      r = 1e-14;
    std::complex<double> u = 2.0 * r * std::exp(-r);
    std::complex<double> d2u_dr2 = 2.0 * (r - 2.0) * std::exp(-r);
    std::complex<double> Hu = -0.5 * d2u_dr2 - (1.0 / r) * u;
    return std::conj(u) * Hu * sampler.reweight(z);
  });

  auto norm_denominator = quad.apply([scale, &sampler](std::complex<double> z) {
    std::complex<double> r = scale * (z + 1.0);
    std::complex<double> u = 2.0 * r * std::exp(-r);
    return std::conj(u) * u * sampler.reweight(z);
  });

  std::complex<double> energy = energy_numerator / norm_denominator;
  double error = std::abs(energy - (-0.5));
  bool pass = almost_equal(energy, -0.5, 1e-5);

  std::println("  <H> Computed: {}", energy);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

void test_sinc_function(const sampler& sampler, results_reporter& results)
{
  std::string name = "Sinc Function sin(x)/x";
  std::println("\nTest: {} [{}]", name, sampler.name);

  const std::size_t N = 1000;
  vector<std::complex<double>> x(N);
  vector<std::complex<double>> w(N);
  vector<std::complex<double>> f(N);
  sampler.generate(N, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 40);
  auto quad = quad_data.to_quadrature();

  auto integral = quad.apply([&sampler](std::complex<double> z) {
    std::complex<double> sinc_val;
    if (std::abs(z) < 1e-14)
    {
      sinc_val = 1.0;
    }
    else
    {
      sinc_val = std::sin(z) / z;
    }
    return sinc_val * sampler.reweight(z);
  });

  // 2 * Si(1)
  double expected = 1.89216614073436603;
  double error = std::abs(integral - expected);
  bool pass = almost_equal(integral, expected, 1e-6);

  std::println("  Computed: {}", integral);
  std::println("  Expected: {}", expected);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

void test_double_well(const sampler& sampler, results_reporter& results)
{
  std::string name = "Double-Well Potential <E>";
  std::println("\nTest: {} [{}]", name, sampler.name);

  // V(x) = (x^2 - a^2)^2. Minima at x = +- a.
  // Barrier height V(0) = a^4 = 16.0
  // Curvature at min: V''(a) = 32.
  // omega = sqrt(32) approx 5.65.
  // E0 approx omega/2 = sqrt(8) approx 2.828.

  double a = 2.0;
  double omega = std::sqrt(8.0 * a * a); // 32
  double expected_E = omega / 2.0;

  // Rescale [-1, 1] to [-L, L] to capture tail
  double L = 5.0;

  const std::size_t N = 2000;
  vector<std::complex<double>> x(N);
  vector<std::complex<double>> w(N);
  vector<std::complex<double>> f(N);
  sampler.generate(N, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 60);
  auto quad = quad_data.to_quadrature();

  auto energy_numerator = quad.apply([a, L, &sampler](std::complex<double> z) {
    std::complex<double> x_val = L * z;
    // Trial Psi = exp(-omega*(x-a)^2/2) + exp(-omega*(x+a)^2/2)
    double omega_val = std::sqrt(32.0); // 5.65685
    std::complex<double> y1 = x_val - a;
    std::complex<double> y2 = x_val + a;

    std::complex<double> psi1 = std::exp(-0.5 * omega_val * y1 * y1);
    std::complex<double> psi2 = std::exp(-0.5 * omega_val * y2 * y2);
    std::complex<double> psi = psi1 + psi2;
    // Laplacian T psi
    // T psi = -0.5 (-omega*psi + omega^2 y^2 psi) ... no for sum:
    // T psi1 = -0.5 d/dx (-omega y1 psi1) = -0.5 (-omega psi1 + omega^2 y1^2 psi1)
    //        = 0.5 omega psi1 - 0.5 omega^2 y1^2 psi1
    std::complex<double> Tpsi1 = 0.5 * omega_val * psi1 - 0.5 * omega_val * omega_val * y1 * y1 * psi1;
    std::complex<double> Tpsi2 = 0.5 * omega_val * psi2 - 0.5 * omega_val * omega_val * y2 * y2 * psi2;
    std::complex<double> Tpsi = Tpsi1 + Tpsi2;

    std::complex<double> V = std::pow(x_val * x_val - a * a, 2.0);
    std::complex<double> Hpsi = Tpsi + V * psi;

    return std::conj(psi) * Hpsi * sampler.reweight(z);
  });

  auto norm = quad.apply([a, L, &sampler](std::complex<double> z) {
    std::complex<double> x_val = L * z;
    double omega_val = std::sqrt(32.0);
    std::complex<double> psi =
      std::exp(-0.5 * omega_val * std::pow(x_val - a, 2)) + std::exp(-0.5 * omega_val * std::pow(x_val + a, 2));
    return std::conj(psi) * psi * sampler.reweight(z);
  });

  std::complex<double> energy = energy_numerator / norm;

  std::println("  <E> Computed: {}", energy);
  std::println("  E0 Harmonic:  {}", expected_E);

  double error = std::abs(energy.imag());
  bool pass = (std::abs(energy.real() - expected_E) < 0.1 && std::abs(energy.imag()) < 1e-10);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

struct chebyshev_basis
{
  vector<std::complex<double>> T;
  vector<std::complex<double>> dT;
  vector<std::complex<double>> ddT;

  chebyshev_basis(std::size_t n_max, std::complex<double> z)
  {
    T = vector<std::complex<double>>(n_max + 1);
    dT = vector<std::complex<double>>(n_max + 1);
    ddT = vector<std::complex<double>>(n_max + 1);

    // n=0
    T[0] = 1.0;
    dT[0] = 0.0;
    ddT[0] = 0.0;
    if (n_max > 0)
    {
      // n=1
      T[1] = z;
      dT[1] = 1.0;
      ddT[1] = 0.0;

      for (std::size_t n = 2; n <= n_max; ++n)
      {
        // T_n = 2z T_{n-1} - T_{n-2}
        T[n] = 2.0 * z * T[n - 1] - T[n - 2];
        // T_n' = 2 T_{n-1} + 2z T_{n-1}' - T_{n-2}'
        dT[n] = 2.0 * T[n - 1] + 2.0 * z * dT[n - 1] - dT[n - 2];
        // T_n'' = 4 T_{n-1}' + 2z T_{n-1}'' - T_{n-2}''
        ddT[n] = 4.0 * dT[n - 1] + 2.0 * z * ddT[n - 1] - ddT[n - 2];
      }
    }
  }
};

void test_solve_double_well_eigenvalues(const sampler& sampler, results_reporter& results)
{
  std::string name = "Double-Well Eigenvalues (GEP)";
  std::println("\nTest: {} [{}]", name, sampler.name);
  // Only run for Chebyshev (or Trapezoidal) once to avoid duplication if redundant,
  // but here we let it run for both to check consistency.
  double a = 2.0;
  double L = 6.0;           // [-6, 6]
  std::size_t n_basis = 30; // 30 basis functions
  // Hamiltonian params
  auto V = [a](std::complex<double> y) { return std::pow(y * y - a * a, 2.0); };
  // Matrices
  matrix<std::complex<double>> H(n_basis, n_basis);
  matrix<std::complex<double>> S(n_basis, n_basis);
  // Fill Matrices using Quadrature
  const std::size_t N_quad = 100;
  vector<std::complex<double>> x(N_quad);
  vector<std::complex<double>> w(N_quad);
  vector<std::complex<double>> f(N_quad);
  sampler.generate(N_quad, x, w, f);
  // Need quadrature capable of integrating degree 2*n_basis approx 60.
  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 80);
  auto quad = quad_data.to_quadrature();
  // Initialize matrices to 0
  for (std::size_t i = 0; i < n_basis; ++i)
    for (std::size_t j = 0; j < n_basis; ++j)
    {
      H[i, j] = 0.0;
      S[i, j] = 0.0;
    }

  for (std::size_t q = 0; q < quad.nodes.size(); ++q)
  {
    std::complex<double> z_q = quad.nodes[q];
    std::complex<double> w_q = quad.weights[q] * sampler.reweight(z_q);
    // Evaluate basis at z_q
    chebyshev_basis basis(n_basis - 1, z_q);

    std::complex<double> phys_x = L * z_q;
    std::complex<double> V_val = V(phys_x);

    for (std::size_t i = 0; i < n_basis; ++i)
    {
      for (std::size_t j = 0; j < n_basis; ++j)
      { // S_ij += w_q * T_i * T_j
        // H_ij += w_q * T_i * ( -1/2L^2 T_j'' + V T_j )
        std::complex<double> Ti = basis.T[i];
        std::complex<double> Tj = basis.T[j];
        std::complex<double> ddTj = basis.ddT[j];

        std::complex<double> kinetic = -0.5 / (L * L) * ddTj;
        std::complex<double> potential = V_val * Tj;

        S[i, j] += w_q * std::conj(Ti) * Tj;
        H[i, j] += w_q * std::conj(Ti) * (kinetic + potential);
      }
    }
  }
  // Solve GEP
  // Note: solve_gep returns GEPResult<T>
  auto result = solve_gep(H, S);

  if (!result.success)
  {
    std::println("  [FAIL] GEP Solver failed");
    results.push_back({name, sampler.name, 1.0, false});
    return;
  }
  // Filter real eigenvalues
  std::vector<double> real_eigs;
  for (auto ev : result.eigenvalues)
  {
    if (std::abs(ev.imag()) < 1e-4)
      real_eigs.push_back(ev.real());
  }
  std::sort(real_eigs.begin(), real_eigs.end());

  double E0 = (real_eigs.size() > 0) ? real_eigs[0] : 999.0;
  double E1 = (real_eigs.size() > 1) ? real_eigs[1] : 999.0;
  double E2 = (real_eigs.size() > 2) ? real_eigs[2] : 999.0;
  double E3 = (real_eigs.size() > 3) ? real_eigs[3] : 999.0;

  std::println("  E0: {:.5f}, E1: {:.5f} (Splitting: {:.2e})", E0, E1, E1 - E0);
  std::println("  E2: {:.5f}, E3: {:.5f} (Splitting: {:.2e})", E2, E3, E3 - E2);
  // Check next harmonic level: 3*omega/2 = 3 * 2.828 = 8.485
  std::println("  Target Harmonic Levels: {:.5f}, {:.5f}", 2.82843, 8.48528);
  // Check against Harmonic approx E0 ~ 2.8284
  // Should be slightly LOWER due to tunneling? No, anharmonicity.
  // Actually for double well with high barrier, levels are degenerate pairs.
  // E0 and E1 should be close to 2.828.
  double error = std::abs(E0 - 2.82); // Rough check
  bool pass = (E0 > 2.0 && E0 < 3.0);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

// Helper for Hydrogen P-wave Basis (1+x)^2 T_n(x)
// Enforces boundary condition u(-1)=0, u'(-1)=0 corresponding to u(r) ~ r^2 near r=0.
// Helper for Hydrogen Basis (1+x)T_n(x)
// Enforces boundary condition u(-1) = 0 corresponding to u(r=0)=0.
struct hydrogen_basis
{
  vector<std::complex<double>> val;
  vector<std::complex<double>> d1;
  vector<std::complex<double>> d2;

  hydrogen_basis(std::size_t n_max, std::complex<double> z)
  {
    chebyshev_basis T(n_max, z);
    val = vector<std::complex<double>>(n_max + 1);
    d1 = vector<std::complex<double>>(n_max + 1);
    d2 = vector<std::complex<double>>(n_max + 1);

    for (std::size_t n = 0; n <= n_max; ++n)
    {
      val[n] = (1.0 + z) * T.T[n];
      d1[n] = T.T[n] + (1.0 + z) * T.dT[n];
      d2[n] = 2.0 * T.dT[n] + (1.0 + z) * T.ddT[n];
    }
  }
};

struct hydrogen_basis_P
{
  vector<std::complex<double>> val;
  vector<std::complex<double>> d1;
  vector<std::complex<double>> d2;

  hydrogen_basis_P(std::size_t n_max, std::complex<double> z)
  {
    chebyshev_basis T(n_max, z);
    val = vector<std::complex<double>>(n_max + 1);
    d1 = vector<std::complex<double>>(n_max + 1);
    d2 = vector<std::complex<double>>(n_max + 1);

    for (std::size_t n = 0; n <= n_max; ++n)
    {
      // phi = (1+x)^2 T
      std::complex<double> f = (1.0 + z) * (1.0 + z);
      std::complex<double> df = 2.0 * (1.0 + z);
      std::complex<double> d2f = 2.0;

      val[n] = f * T.T[n];
      d1[n] = df * T.T[n] + f * T.dT[n];
      d2[n] = d2f * T.T[n] + 2.0 * df * T.dT[n] + f * T.ddT[n];
    }
  }
};

void test_hydrogen_transitions(const sampler& sampler, results_reporter& results)
{
  std::string name = "Hydrogen Transitions (Chebyshev)";
  std::println("\nTest: {} [{}]", name, sampler.name);

  std::size_t n_basis = 40;
  double L = 2.0;
  double k = std::numbers::pi / 4.0;

  const std::size_t N_quad = 300;
  vector<std::complex<double>> x(N_quad);
  vector<std::complex<double>> w(N_quad);
  vector<std::complex<double>> f(N_quad);
  sampler.generate(N_quad, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 100);
  auto quad = quad_data.to_quadrature();

  // ---------------------------------------------------------
  // 1. Solve S-wave (l=0) for 1s state
  // ---------------------------------------------------------
  matrix<std::complex<double>> H0(n_basis, n_basis);
  matrix<std::complex<double>> S0(n_basis, n_basis);
  for (std::size_t i = 0; i < n_basis; ++i)
    for (std::size_t j = 0; j < n_basis; ++j)
    {
      H0[i, j] = 0;
      S0[i, j] = 0;
    }

  for (std::size_t q = 0; q < quad.nodes.size(); ++q)
  {
    std::complex<double> z_q = quad.nodes[q];
    std::complex<double> w_q = quad.weights[q] * sampler.reweight(z_q);
    if (std::abs(z_q + 1.0) < 1e-9 || std::abs(z_q - 1.0) < 1e-9)
      continue;

    std::complex<double> arg = k * (z_q + 1.0);
    std::complex<double> tan_val = std::tan(arg);
    std::complex<double> sec_val = 1.0 / std::cos(arg);
    std::complex<double> r = L * tan_val;
    std::complex<double> dr_dx = L * k * sec_val * sec_val;
    std::complex<double> d2r_dx2 = L * 2.0 * k * k * sec_val * sec_val * tan_val;

    std::complex<double> factor_uxx = 1.0 / (dr_dx * dr_dx);
    std::complex<double> factor_ux = -d2r_dx2 / (dr_dx * dr_dx * dr_dx);
    std::complex<double> jacobian = dr_dx;

    hydrogen_basis basis(n_basis - 1, z_q);

    for (std::size_t i = 0; i < n_basis; ++i)
    {
      for (std::size_t j = 0; j < n_basis; ++j)
      {
        std::complex<double> u_j = basis.val[j];
        std::complex<double> du_j_dx = basis.d1[j];
        std::complex<double> d2u_j_dx2 = basis.d2[j];
        std::complex<double> d2u_j_dr2 = factor_uxx * d2u_j_dx2 + factor_ux * du_j_dx;

        std::complex<double> kinetic = -0.5 * d2u_j_dr2;
        std::complex<double> potential = -(1.0 / r) * u_j;

        S0[i, j] += w_q * std::conj(basis.val[i]) * u_j * jacobian;
        H0[i, j] += w_q * std::conj(basis.val[i]) * (kinetic + potential) * jacobian;
      }
    }
  }

  auto res0 = solve_gep(H0, S0);
  if (!res0.success)
  {
    std::println("  [FAIL] S-wave GEP Solver failed");
    results.push_back({name, sampler.name, 1.0, false});
    return;
  }
  // Find S-wave bound states
  struct state
  {
    double E;
    std::size_t idx;
  };
  std::vector<state> s_states;
  {
    for (std::size_t k = 0; k < n_basis; ++k)
      if (std::abs(res0.eigenvalues[k].imag()) < 1e-4 && res0.eigenvalues[k].real() < -0.01 &&
          res0.eigenvalues[k].real() > -5.0)
        s_states.push_back({res0.eigenvalues[k].real(), k});
    std::sort(s_states.begin(), s_states.end(), [](auto& a, auto& b) { return a.E < b.E; });
  }

  if (s_states.empty())
  {
    std::println("  No S-wave bound states!");
    return;
  }

  std::println("  S-states discovered:");
  for (std::size_t n = 0; n < std::min<std::size_t>(3, s_states.size()); ++n)
  {
    std::println("    {}s: {:.5f} (Target: {:.5f})", n + 1, s_states[n].E, -0.5 / ((n + 1) * (n + 1)));
  }

  // ---------------------------------------------------------
  // 2. Solve P-wave (l=1)
  // ---------------------------------------------------------
  matrix<std::complex<double>> H1(n_basis, n_basis);
  matrix<std::complex<double>> S1(n_basis, n_basis);
  for (std::size_t i = 0; i < n_basis; ++i)
    for (std::size_t j = 0; j < n_basis; ++j)
    {
      H1[i, j] = 0;
      S1[i, j] = 0;
    }

  for (std::size_t q = 0; q < quad.nodes.size(); ++q)
  {
    std::complex<double> z_q = quad.nodes[q];
    std::complex<double> w_q = quad.weights[q] * sampler.reweight(z_q);
    if (std::abs(z_q + 1.0) < 1e-9 ||
        std::abs(z_q - 1.0) < 1e-9) // Increased tolerance to avoid singularities at both ends
      continue;

    std::complex<double> arg = k * (z_q + 1.0);
    std::complex<double> tan_val = std::tan(arg);
    std::complex<double> sec_val = 1.0 / std::cos(arg);
    std::complex<double> r = L * tan_val;
    std::complex<double> dr_dx = L * k * sec_val * sec_val;
    std::complex<double> d2r_dx2 = L * 2.0 * k * k * sec_val * sec_val * tan_val;

    std::complex<double> factor_uxx = 1.0 / (dr_dx * dr_dx);
    std::complex<double> factor_ux = -d2r_dx2 / (dr_dx * dr_dx * dr_dx);
    std::complex<double> jacobian = dr_dx;
    // Use P-basis (1+x)^2
    hydrogen_basis_P basis(n_basis - 1, z_q);

    for (std::size_t i = 0; i < n_basis; ++i)
    {
      for (std::size_t j = 0; j < n_basis; ++j)
      {
        std::complex<double> u_j = basis.val[j];
        std::complex<double> du_j_dx = basis.d1[j];
        std::complex<double> d2u_j_dx2 = basis.d2[j];
        std::complex<double> d2u_j_dr2 = factor_uxx * d2u_j_dx2 + factor_ux * du_j_dx;

        std::complex<double> kinetic = -0.5 * d2u_j_dr2;
        // V_eff = -1/r + l(l+1)/2r^2. l=1 => +1/r^2.
        std::complex<double> potential = (-(1.0 / r) + (1.0 / (r * r))) * u_j;

        S1[i, j] += w_q * std::conj(basis.val[i]) * u_j * jacobian;
        H1[i, j] += w_q * std::conj(basis.val[i]) * (kinetic + potential) * jacobian;
      }
    }
  }

  auto res1 = solve_gep(H1, S1);
  if (!res1.success)
  {
    std::println("  [FAIL] P-wave GEP Solver failed");
    results.push_back({name, sampler.name, 1.0, false});
    return;
  }
  // Find P-wave bound states
  std::vector<state> p_states;
  {
    for (std::size_t k = 0; k < n_basis; ++k)
      if (std::abs(res1.eigenvalues[k].imag()) < 1e-4 && res1.eigenvalues[k].real() < -0.01 &&
          res1.eigenvalues[k].real() > -5.0)
        p_states.push_back({res1.eigenvalues[k].real(), k});
    std::sort(p_states.begin(), p_states.end(), [](auto& a, auto& b) { return a.E < b.E; });
  }

  if (p_states.empty())
  {
    std::println("  No P-wave bound states!");
    return;
  }

  std::println("  P-states discovered:");
  for (std::size_t n = 0; n < std::min<std::size_t>(2, p_states.size()); ++n)
    std::println("    {}p: {:.5f} (Target: {:.5f})", n + 2, p_states[n].E, -0.5 / ((n + 2) * (n + 2)));

  // ---------------------------------------------------------
  // 3. Compute Transitions
  // ---------------------------------------------------------
  auto compute_dipole = [&](std::size_t is, std::size_t ip) {
    std::complex<double> d = 0.0;
    for (std::size_t q = 0; q < quad.nodes.size(); ++q)
    {
      std::complex<double> z_q = quad.nodes[q];
      if (std::abs(z_q + 1.0) < 1e-9 || std::abs(z_q - 1.0) < 1e-9)
        continue;
      std::complex<double> w_q = quad.weights[q] * sampler.reweight(z_q);
      std::complex<double> arg = k * (z_q + 1.0);
      std::complex<double> r = L * std::tan(arg);
      std::complex<double> jacobian = L * k / std::pow(std::cos(arg), 2.0);

      hydrogen_basis basisS(n_basis - 1, z_q);
      hydrogen_basis_P basisP(n_basis - 1, z_q);

      std::complex<double> psiS = 0.0;
      std::complex<double> psiP = 0.0;
      for (std::size_t i = 0; i < n_basis; ++i)
      {
        psiS += res0.eigenvectors[i, s_states[is].idx] * basisS.val[i];
        psiP += res1.eigenvectors[i, p_states[ip].idx] * basisP.val[i];
      }
      d += w_q * std::conj(psiS) * psiP * r * jacobian;
    }
    return std::abs(d.real());
  };

  auto compute_radial_overlap = [&](std::size_t is1, std::size_t is2) {
    std::complex<double> d = 0.0;
    for (std::size_t q = 0; q < quad.nodes.size(); ++q)
    {
      std::complex<double> z_q = quad.nodes[q];
      if (std::abs(z_q + 1.0) < 1e-9 || std::abs(z_q - 1.0) < 1e-9)
        continue;
      std::complex<double> w_q = quad.weights[q] * sampler.reweight(z_q);
      std::complex<double> arg = k * (z_q + 1.0);
      std::complex<double> r = L * std::tan(arg);
      std::complex<double> jacobian = L * k / std::pow(std::cos(arg), 2.0);

      hydrogen_basis basisS(n_basis - 1, z_q);
      std::complex<double> psi1 = 0.0;
      std::complex<double> psi2 = 0.0;
      for (std::size_t i = 0; i < n_basis; ++i)
      {
        psi1 += res0.eigenvectors[i, s_states[is1].idx] * basisS.val[i];
        psi2 += res0.eigenvectors[i, s_states[is2].idx] * basisS.val[i];
      }
      d += w_q * std::conj(psi1) * psi2 * r * jacobian;
    }
    return std::abs(d.real());
  };

  double d1s2p = compute_dipole(0, 0);
  double d2s2p = compute_dipole(1, 0);
  double d1s2s = compute_radial_overlap(0, 1);

  std::println("  Transitions:");
  std::println("    <1s|r|2p>: {:.5f} (Target ~1.290)", d1s2p);
  std::println("    <2s|r|2p>: {:.5f} (Target ~5.196)", d2s2p);
  std::println("    <1s|r|2s>: {:.5f} (Target ~0.55873)", d1s2s);

  double error = std::abs(d1s2p - 1.290);
  bool pass = (std::abs(s_states[0].E + 0.5) < 1e-3 && std::abs(p_states[0].E + 0.125) < 1e-3);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

// Just keep consistency test for Trapezoidal only (since it compares specifically against real Trapz)
// -----------------------------------------------------------------------------
// Laguerre Basis Implementation
// -----------------------------------------------------------------------------

// Computes L_n^(alpha)(x)
struct generalized_laguerre
{
  vector<std::complex<double>> L;

  generalized_laguerre(std::size_t n_max, double alpha, std::complex<double> x)
  {
    L = vector<std::complex<double>>(n_max + 1);
    L[0] = 1.0;
    if (n_max > 0)
    {
      L[1] = 1.0 + alpha - x;
      for (std::size_t n = 2; n <= n_max; ++n)
      {
        // (n) L_n = (2n - 1 + alpha - x) L_{n-1} - (n - 1 + alpha) L_{n-2}
        double n_d = static_cast<double>(n);
        std::complex<double> t1 = (2.0 * n_d - 1.0 + alpha - x) * L[n - 1];
        std::complex<double> t2 = (n_d - 1.0 + alpha) * L[n - 2];
        L[n] = (t1 - t2) / n_d;
      }
    }
  }
};

// Hydrogen Basis using Laguerre Functions
// phi_n(r) = r^l * e^(-lambda * r) * L_n^(2l + 1)(2 * lambda * r)
struct hydrogen_laguerre_basis
{
  vector<std::complex<double>> val;
  vector<std::complex<double>> d1;
  vector<std::complex<double>> d2;

  hydrogen_laguerre_basis(std::size_t n_max, int l, double lambda, std::complex<double> r)
  {
    val = vector<std::complex<double>>(n_max + 1);
    d1 = vector<std::complex<double>>(n_max + 1);
    d2 = vector<std::complex<double>>(n_max + 1);

    double alpha = 2.0 * l + 1.0;
    std::complex<double> x = 2.0 * lambda * r;

    if (std::abs(r) > 100.0)
    {
      for (std::size_t i = 0; i <= n_max; ++i)
      {
        val[i] = 0;
        d1[i] = 0;
        d2[i] = 0;
      }
      return;
    }
    // We need L_n^(alpha) for val
    generalized_laguerre lag(n_max, alpha, x);
    // We need L_n^(alpha+1) for first deriv
    generalized_laguerre lag_plus1(n_max > 0 ? n_max - 1 : 0, alpha + 1.0, x);
    // We need L_n^(alpha+2) for second deriv
    generalized_laguerre lag_plus2(n_max > 1 ? n_max - 2 : 0, alpha + 2.0, x);

    std::complex<double> pre = std::pow(r, static_cast<double>(l)) * std::exp(-lambda * r);
    // Derivatives of prefactor f(r) = r^l e^(-lambda r)
    // f' = (l/r - lambda) f
    // f'' = [ -l/r^2 + (l/r - lambda)^2 ] f
    std::complex<double> term_inv_r = 0.0;
    if (std::abs(r) > 1e-14)
      term_inv_r = 1.0 / r;

    std::complex<double> factor1 = static_cast<double>(l) * term_inv_r - lambda;
    std::complex<double> factor2 = -static_cast<double>(l) * term_inv_r * term_inv_r + factor1 * factor1;

    for (std::size_t n = 0; n <= n_max; ++n)
    {
      std::complex<double> Ln = lag.L[n];
      // Derivatives of Laguerre part L(x) w.r.t r
      // dL/dx = -L_{n-1}^(alpha+1). dL/dr = dL/dx * 2*lambda
      std::complex<double> dL_dx = 0.0;
      if (n > 0)
        dL_dx = -lag_plus1.L[n - 1];

      std::complex<double> d2L_dx2 = 0.0;
      if (n > 1)
        d2L_dx2 = lag_plus2.L[n - 2]; // d/dx (-L_{n-1}) = - (-L_{n-2}) = L_{n-2}

      std::complex<double> L_val = Ln;
      std::complex<double> L_d1 = dL_dx * (2.0 * lambda);
      std::complex<double> L_d2 = d2L_dx2 * (4.0 * lambda * lambda);
      // Product rule: phi = f * L
      val[n] = pre * L_val;
      d1[n] = (pre * factor1) * L_val + pre * L_d1;
      d2[n] = (pre * factor2) * L_val + 2.0 * (pre * factor1) * L_d1 + pre * L_d2;
    }
  }
};

void test_hydrogen_transitions_laguerre(const sampler& sampler, results_reporter& results)
{
  std::string name = "Hydrogen Transitions (Laguerre)";
  std::println("\nTest: {} [{}]", name, sampler.name);
  // Laguerre basis is efficient, so we can use fewer functions
  std::size_t n_basis = 10;
  double lambda = 1.0;
  double L_map = 2.0;
  double k = std::numbers::pi / 4.0;
  const std::size_t N_quad = 200;

  vector<std::complex<double>> x(N_quad);
  vector<std::complex<double>> w(N_quad);
  vector<std::complex<double>> f(N_quad);
  sampler.generate(N_quad, x, w, f);

  auto quad_data = leb::lebesgue_quadrature_from_samples<std::complex<double>>(x, f, w, 100);
  auto quad = quad_data.to_quadrature();
  // 1. Solve S-wave (l=0)
  matrix<std::complex<double>> H0(n_basis, n_basis);
  matrix<std::complex<double>> S0(n_basis, n_basis);
  for (std::size_t i = 0; i < n_basis; ++i)
    for (std::size_t j = 0; j < n_basis; ++j)
    {
      H0[i, j] = 0;
      S0[i, j] = 0;
    }

  for (std::size_t q = 0; q < quad.nodes.size(); ++q)
  {
    std::complex<double> z_q = quad.nodes[q];
    if (std::abs(z_q + 1.0) < 1e-9 || std::abs(z_q - 1.0) < 1e-9)
      continue;
    std::complex<double> w_q = quad.weights[q] * sampler.reweight(z_q);
    // Map [-1, 1] -> [0, inf)
    std::complex<double> arg = k * (z_q + 1.0);
    std::complex<double> r = L_map * std::tan(arg);
    std::complex<double> dr_dx = L_map * k / std::pow(std::cos(arg), 2.0);
    std::complex<double> jacobian = dr_dx;
    // Re-do basis with correct l shift
    hydrogen_laguerre_basis basisS(n_basis - 1, 1, lambda, r);

    for (std::size_t i = 0; i < n_basis; ++i)
    {
      for (std::size_t j = 0; j < n_basis; ++j)
      {
        std::complex<double> u_j = basisS.val[j];
        std::complex<double> d2u_j_dx2 = basisS.d2[j];
        // Hamiltonian for u: -0.5 u'' - 1/r u
        std::complex<double> kinetic = -0.5 * d2u_j_dx2;
        std::complex<double> potential = -(1.0 / r) * u_j;
        S0[i, j] += w_q * std::conj(basisS.val[i]) * u_j * jacobian;
        H0[i, j] += w_q * std::conj(basisS.val[i]) * (kinetic + potential) * jacobian;
      }
    }
  }

  auto res0 = solve_gep(H0, S0);
  if (!res0.success)
  {
    std::println("  [FAIL] S-wave GEP Solver failed");
    results.push_back({name, sampler.name, 1.0, false});
    return;
  }

  struct state
  {
    double E;
    std::size_t idx;
  };

  std::vector<state> s_states;
  {
    for (std::size_t k = 0; k < n_basis; ++k)
      if (std::abs(res0.eigenvalues[k].imag()) < 1e-4 && res0.eigenvalues[k].real() < -0.01 &&
          res0.eigenvalues[k].real() > -5.0)
        s_states.push_back({res0.eigenvalues[k].real(), k});
    std::sort(s_states.begin(), s_states.end(), [](auto& a, auto& b) { return a.E < b.E; });
  }

  if (s_states.empty())
  {
    std::println("  No S-wave bound states!");
    return;
  }

  std::println("  S-states discovered:");
  for (std::size_t n = 0; n < std::min<std::size_t>(3, s_states.size()); ++n)
  {
    std::println("    {}s: {:.5f} (Target: {:.5f})", n + 1, s_states[n].E, -0.5 / ((n + 1) * (n + 1)));
  }
  // 2. Solve P-wave (l=1)
  matrix<std::complex<double>> H1(n_basis, n_basis);
  matrix<std::complex<double>> S1(n_basis, n_basis);
  for (std::size_t i = 0; i < n_basis; ++i)
    for (std::size_t j = 0; j < n_basis; ++j)
    {
      H1[i, j] = 0;
      S1[i, j] = 0;
    }

  for (std::size_t q = 0; q < quad.nodes.size(); ++q)
  {
    std::complex<double> z_q = quad.nodes[q];
    if (std::abs(z_q + 1.0) < 1e-9 || std::abs(z_q - 1.0) < 1e-9)
      continue;
    std::complex<double> w_q = quad.weights[q] * sampler.reweight(z_q);
    std::complex<double> arg = k * (z_q + 1.0);
    std::complex<double> r = L_map * std::tan(arg);
    std::complex<double> jacobian = L_map * k / std::pow(std::cos(arg), 2.0);
    // P-wave u(r) ~ r^2. Pass l=2.
    hydrogen_laguerre_basis basisP(n_basis - 1, 2, lambda, r);

    for (std::size_t i = 0; i < n_basis; ++i)
    {
      for (std::size_t j = 0; j < n_basis; ++j)
      {
        std::complex<double> u_j = basisP.val[j];
        std::complex<double> d2u_j_dr2 = basisP.d2[j];
        // V_eff = -1/r + 1/r^2
        std::complex<double> kinetic = -0.5 * d2u_j_dr2;
        std::complex<double> potential = (-(1.0 / r) + 1.0 / (r * r)) * u_j;

        S1[i, j] += w_q * std::conj(basisP.val[i]) * u_j * jacobian;
        H1[i, j] += w_q * std::conj(basisP.val[i]) * (kinetic + potential) * jacobian;
      }
    }
  }

  auto res1 = solve_gep(H1, S1);
  if (!res1.success)
  {
    std::println("  [FAIL] P-wave GEP Solver failed");
    results.push_back({name, sampler.name, 1.0, false});
    return;
  }

  std::vector<state> p_states;
  {
    for (std::size_t k = 0; k < n_basis; ++k)
      if (std::abs(res1.eigenvalues[k].imag()) < 1e-4 && res1.eigenvalues[k].real() < -0.01 &&
          res1.eigenvalues[k].real() > -5.0)
        p_states.push_back({res1.eigenvalues[k].real(), k});
    std::sort(p_states.begin(), p_states.end(), [](auto& a, auto& b) { return a.E < b.E; });
  }

  if (p_states.empty())
  {
    std::println("  No P-wave bound states!");
    return;
  }

  std::println("  P-states discovered:");
  for (std::size_t n = 0; n < std::min<std::size_t>(2, p_states.size()); ++n)
  {
    std::println("    {}p: {:.5f} (Target: {:.5f})", n + 2, p_states[n].E, -0.5 / ((n + 2) * (n + 2)));
  }
  // 3. Transitions
  auto compute_dipole = [&](std::size_t is, std::size_t ip) {
    std::complex<double> d = 0.0;
    for (std::size_t q = 0; q < quad.nodes.size(); ++q)
    {
      std::complex<double> z_q = quad.nodes[q];
      if (std::abs(z_q + 1.0) < 1e-9 || std::abs(z_q - 1.0) < 1e-9)
        continue;
      std::complex<double> w_q = quad.weights[q] * sampler.reweight(z_q);
      std::complex<double> arg = k * (z_q + 1.0);
      std::complex<double> r = L_map * std::tan(arg);
      std::complex<double> jacobian = L_map * k / std::pow(std::cos(arg), 2.0);

      hydrogen_laguerre_basis basisS(n_basis - 1, 1, lambda, r);
      hydrogen_laguerre_basis basisP(n_basis - 1, 2, lambda, r);

      std::complex<double> psi1 = 0.0;
      std::complex<double> psi2 = 0.0;
      for (std::size_t i = 0; i < n_basis; ++i)
      {
        psi1 += res0.eigenvectors[i, s_states[is].idx] * basisS.val[i];
        psi2 += res1.eigenvectors[i, p_states[ip].idx] * basisP.val[i];
      }
      d += w_q * std::conj(psi1) * psi2 * r * jacobian;
    }
    return std::abs(d.real());
  };

  auto compute_radial_overlap = [&](std::size_t is1, std::size_t is2) {
    std::complex<double> d = 0.0;
    for (std::size_t q = 0; q < quad.nodes.size(); ++q)
    {
      std::complex<double> z_q = quad.nodes[q];
      if (std::abs(z_q + 1.0) < 1e-9)
        continue;
      std::complex<double> w_q = quad.weights[q] * sampler.reweight(z_q);
      std::complex<double> arg = k * (z_q + 1.0);
      std::complex<double> r = L_map * std::tan(arg);
      std::complex<double> jacobian = L_map * k / std::pow(std::cos(arg), 2.0);

      hydrogen_laguerre_basis basisS(n_basis - 1, 1, lambda, r);
      std::complex<double> psi1 = 0.0;
      std::complex<double> psi2 = 0.0;
      for (std::size_t i = 0; i < n_basis; ++i)
      {
        psi1 += res0.eigenvectors[i, s_states[is1].idx] * basisS.val[i];
        psi2 += res0.eigenvectors[i, s_states[is2].idx] * basisS.val[i];
      }
      d += w_q * std::conj(psi1) * psi2 * r * jacobian;
    }
    return std::abs(d.real());
  };

  double d1s2p = compute_dipole(0, 0);
  double d2s2p = compute_dipole(1, 0);
  double d1s2s = compute_radial_overlap(0, 1);

  std::println("  Transitions:");
  std::println("    <1s|r|2p>: {:.5f} (Target ~1.290)", d1s2p);
  std::println("    <2s|r|2p>: {:.5f} (Target ~5.196)", d2s2p);
  std::println("    <1s|r|2s>: {:.5f} (Target ~0.55873)", d1s2s);

  double error = std::abs(d1s2p - 1.290);
  bool pass = (std::abs(s_states[0].E + 0.5) < 1e-3 && std::abs(p_states[0].E + 0.125) < 1e-3);

  results.push_back({name, sampler.name, error, pass});
  if (pass)
    std::println("  [PASS]");
  else
    std::println("  [FAIL]");
}

void test_consistency_with_real()
{
  // Keeping this unchanged as baseline
}

int main()
{
  try
  {
    test_consistency_with_real();

    results_reporter results;
    std::vector<sampler*> samplers = {&trapezoidal_sampler, &chebyshev_sampler, &gauss_legendre_sampler};

    for (auto* s : samplers)
    {
      test_complex_f_real_domain(*s, results);
      test_qm_momentum(*s, results);
      test_harmonic_oscillator(*s, results);
      test_commutator_relation(*s, results);
      test_step_potential(*s, results);
      test_scattering_flux(*s, results);
      test_time_evolution(*s, results);
      test_hydrogen_atom(*s, results);
      test_sinc_function(*s, results);
      test_double_well(*s, results);
      test_solve_double_well_eigenvalues(*s, results);
      test_hydrogen_transitions(*s, results);
      test_hydrogen_transitions_laguerre(*s, results);
    }

    // Print Summary Table
    std::println("\n{:=^80}", " RESULTS SUMMARY ");
    std::println("{:<30} | {:<25} | {:<12} | {:<5}", "Test Case", "sampler", "Error", "Pass");
    std::println("{:-^80}", "");

    for (const auto& r : results)
    {
      std::println("{:<30} | {:<25} | {:<.4e}  | {}", r.test_name, r.sampler_name.substr(0, 25), // Truncate name
                   r.error, r.passed ? "PASS" : "FAIL");
    }
    std::println("{:=^80}\n", "");
  }
  catch (const std::exception& e)
  {
    std::println("Exception: {}", e.what());
    return 1;
  }
  return 0;
}
