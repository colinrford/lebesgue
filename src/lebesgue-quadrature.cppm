/*
 *  lebesgue-quadrature.cppm – based on Java implementation of Vladislav Malyshkin
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  Core Lebesgue quadrature logic, including node/weight definitions and construction algorithms.
 */

export module lam.lebesgue:quadrature;

import std;
import lam.linearalgebra;
import :detail;
import :basis;
import :gram;

export namespace lam::leb
{

using lam::matrix;
using lam::vector;
namespace stdr = std::ranges;


/**
 * Basic quadrature rule: nodes and weights.
 */
template<typename scalar = double>
struct quadrature
{
  vector<scalar> nodes;
  vector<scalar> weights;

  /**
   * Apply quadrature to integrate function f.
   * Returns Σ weights[k] * f(nodes[k])
   */
  template<std::invocable<scalar> F>
  scalar apply(F&& f) const
  {
    scalar sum = ::leb::detail::constants<scalar>::zero;
    for (auto [node, weight] : std::views::zip(nodes, weights))
      sum += weight * std::forward<F>(f)(node);
    return sum;
  }

  /**
   * Sum of all weights (should equal integral of 1 over the measure).
   */
  [[nodiscard]] scalar total_weight() const
  { return std::accumulate(stdr::begin(weights), stdr::end(weights), ::leb::detail::constants<scalar>::zero); }
};

/**
 * Radon-Nikodym evaluation result at a point.
 */
template<typename scalar>
struct radon_nikodym_eval
{
  scalar radon_nikodym;
  scalar least_squares;
  scalar christoffel_inv;
};

/**
 * Extended Lebesgue quadrature with eigendecomposition data.
 * Allows evaluation of Radon-Nikodym derivative at arbitrary points.
 */
template<typename scalar>
struct lebesgue_quadrature_data
{
  vector<scalar> nodes;        // Eigenvalues (f-values)
  vector<scalar> weights;      // quadrature weights
  vector<scalar> psi_averages; // <ψ_k> = G[0,:] · v_k
  matrix<scalar> eigenvectors; // Normalized eigenvectors (columns)
  matrix<scalar> gram_matrix;  // For RN evaluation

  /**
   * Evaluate Radon-Nikodym, least squares, and Christoffel at point x0.
   */
  [[nodiscard]] radon_nikodym_eval<scalar> evaluate_at(scalar x0) const
  {
    const int n = static_cast<int>(nodes.size());

    // Compute basis functions at x0
    vector<scalar> T(n);
    T[0] = ::leb::detail::constants<scalar>::one;
    if (n > 1)
      T[1] = x0;

    scalar two_x0 = ::leb::detail::constants<scalar>::two * x0;
    for (int k = 2; k < n; ++k)
      T[k] = two_x0 * T[k - 1] - T[k - 2];

    scalar sum_psi_sq = ::leb::detail::constants<scalar>::zero;
    scalar sum_lambda_psi_sq = ::leb::detail::constants<scalar>::zero;
    scalar sum_lambda_psi_avg_psi = ::leb::detail::constants<scalar>::zero;

    for (int k = 0; k < n; ++k)
    {
      // Compute ψ_k(x0) = v_k · T
      scalar psi_k = ::leb::detail::constants<scalar>::zero;
      for (int i = 0; i < n; ++i)
        psi_k += eigenvectors[i, k] * T[i];

      // K(x, x) = sum |psi_k(x)|^2
      // For complex case, we want magnitude squared: psi * conj(psi)
      scalar psi_mag_sq = psi_k * ::leb::detail::conj(psi_k);

      sum_psi_sq += psi_mag_sq;
      sum_lambda_psi_sq += nodes[k] * psi_mag_sq;
      sum_lambda_psi_avg_psi += nodes[k] * psi_averages[k] * psi_k;
    }

    scalar rn =
      (std::abs(sum_psi_sq) > 1e-14) ? sum_lambda_psi_sq / sum_psi_sq : ::leb::detail::constants<scalar>::zero;

    return {rn, sum_lambda_psi_avg_psi, sum_psi_sq};
  }

  /**
   * Get basic quadrature (nodes and weights only).
   */
  [[nodiscard]] quadrature<scalar> to_quadrature() const { return {nodes, weights}; }
};


/**
 * Build Lebesgue quadrature from samples.
 *
 * @param x Sample x-values
 * @param f Function values f(x[i])
 * @param w Sample weights
 * @param n Number of quadrature points to generate
 * @return Extended quadrature data with eigendecomposition
 */
template<typename scalar>
inline lebesgue_quadrature_data<scalar> lebesgue_quadrature_from_samples(std::span<const scalar> x,
                                                                         std::span<const scalar> f,
                                                                         std::span<const scalar> w, std::size_t n)
{
  // 1. Build Gram matrix G = <T_j, T_k>
  auto G = gram_matrix_from_samples<scalar>(x, w, n);
  // 2. Build weighted Gram matrix F = <f·T_j, T_k>
  auto F = weighted_gram_matrix_from_samples<scalar>(x, f, w, n);
  // 3. Solve GEP: F*v = λ*G*v
  auto gep = lam::linalg::solve_gep(F, G);
  if (!gep.success)
  {
    throw std::runtime_error("lebesgue_quadrature: solve_gep failed. Measure likely singular or n too high.");
  }
  // 4. Extract results
  lebesgue_quadrature_data<scalar> result;
  result.gram_matrix = G;
  result.eigenvectors = gep.eigenvectors;

  result.nodes = vector<scalar>(n);
  result.weights = vector<scalar>(n);
  result.psi_averages = vector<scalar>(n);

  for (std::size_t k = 0; k < n; ++k)
  {
    result.nodes[k] = gep.eigenvalues[k];
    // ψ_average = <1, ψ_k> = <T_0, ψ_k>
    // sum_j v_{jk} G[0, j]
    scalar psi_avg = ::leb::detail::constants<scalar>::zero;
    for (std::size_t i = 0; i < n; ++i)
      psi_avg += G[0, i] * gep.eigenvectors[i, k];
    result.psi_averages[k] = psi_avg;
    result.weights[k] = psi_avg * ::leb::detail::conj(psi_avg);
  }

  return result;
}

/**
 * Build Lebesgue quadrature from precomputed moments.
 */
template<typename scalar>
inline lebesgue_quadrature_data<scalar> lebesgue_quadrature_from_moments(std::span<const scalar> f_moments,
                                                                         std::span<const scalar> moments, std::size_t n)
{
  if (moments.size() < 2 * n || f_moments.size() < 2 * n)
  {
    throw std::invalid_argument("Need at least 2n moments for n-point quadrature");
  }

  // 1. Build Gram matrix from standard moments
  auto G = gram_matrix_from_moments<scalar>(moments, n);
  // 2. Build weighted Gram matrix from f-weighted moments
  auto F = gram_matrix_from_moments<scalar>(f_moments, n);
  // 3. Solve GEP
  auto gep = lam::linalg::solve_gep(F, G);
  if (!gep.success)
  {
    throw std::runtime_error("lebesgue_quadrature: solve_gep failed for moments.");
  }
  // 4. Extract results
  lebesgue_quadrature_data<scalar> result;
  result.gram_matrix = G;
  result.eigenvectors = gep.eigenvectors;
  result.nodes = vector<scalar>(n);
  result.weights = vector<scalar>(n);
  result.psi_averages = vector<scalar>(n);

  for (std::size_t k = 0; k < n; ++k)
  {
    result.nodes[k] = gep.eigenvalues[k];
    scalar psi_avg = ::leb::detail::constants<scalar>::zero;
    for (std::size_t i = 0; i < n; ++i)
      psi_avg += G[0, i] * gep.eigenvectors[i, k];
    result.psi_averages[k] = psi_avg;
    result.weights[k] = psi_avg * ::leb::detail::conj(psi_avg);
  }

  return result;
}

/**
 * Build Gauss quadrature from Chebyshev moments.
 */
template<typename scalar>
inline lebesgue_quadrature_data<scalar> gauss_quadrature_from_moments(std::span<const scalar> moments, std::size_t n)
{
  if (moments.size() < 2 * n)
  {
    throw std::invalid_argument("Need at least 2n moments for n-point quadrature");
  }

  // 1. Build Gram matrix from moments
  auto G = gram_matrix_from_moments<scalar>(moments, n);
  // 2. Compute <x·T_k> moments from <T_k> moments
  auto x_mom = x_moments_from_moments<scalar>(moments);
  // 3. Build weighted Gram matrix F = <x·T_j, T_k>
  auto F = gram_matrix_from_moments<scalar>(std::span(x_mom), n);
  // 4. Solve GEP
  auto gep = lam::linalg::solve_gep(F, G);
  if (!gep.success)
  {
    throw std::runtime_error("lebesgue_quadrature: solve_gep failed for Gauss-Chebyshev moments.");
  }
  // 5. Extract results
  lebesgue_quadrature_data<scalar> result;
  result.gram_matrix = G;
  result.eigenvectors = gep.eigenvectors;
  result.nodes = vector<scalar>(n);
  result.weights = vector<scalar>(n);
  result.psi_averages = vector<scalar>(n);

  for (std::size_t k = 0; k < n; ++k)
  {
    result.nodes[k] = gep.eigenvalues[k];
    scalar psi_avg = ::leb::detail::constants<scalar>::zero;
    for (std::size_t i = 0; i < n; ++i)
      psi_avg += G[0, i] * gep.eigenvectors[i, k];
    result.psi_averages[k] = psi_avg;
    result.weights[k] = psi_avg * ::leb::detail::conj(psi_avg);
  }

  return result;
}

/**
 * Build Gauss-Legendre quadrature for interval [-1, 1].
 */
inline quadrature<double> gauss_legendre(std::size_t n)
{
  vector<double> moments(2 * n);
  for (std::size_t k = 0; k < 2 * n; ++k)
  {
    if (k == 0)
      moments[k] = 2.0;
    else if (k % 2 == 1)
      moments[k] = 0.0;
    else
    {
      auto k2 = static_cast<double>(k * k);
      moments[k] = 2.0 / (1.0 - k2);
    }
  }

  auto data = gauss_quadrature_from_moments<double>(moments, n);
  return data.to_quadrature();
}

} // namespace lam::leb
