/* 
 *  lebesgue-basis.cppm – based on Java implementation of Vladislav Malyshkin
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  Implementation of Chebyshev polynomial basis evaluation, summation, and moment computation.
 */

export module lam.lebesgue:basis;

import std;
import lam.linearalgebra;
import :detail;

export namespace lam::leb
{

using lam::vector;

// ============================================================================
// Chebyshev Polynomial Basis
// ============================================================================

/**
 * Evaluate Chebyshev polynomial T_n(x) using three-term recurrence:
 *   T_0(x) = 1
 *   T_1(x) = x
 *   T_n(x) = 2x * T_{n-1}(x) - T_{n-2}(x)  for n >= 2
 */
template<typename scalar>
constexpr scalar chebyshev_eval(int n, scalar x) noexcept
{
  if (n == 0)
    return ::leb::detail::constants<scalar>::one;
  if (n == 1)
    return x;

  scalar T_prev2 = ::leb::detail::constants<scalar>::one; // T_{n-2}
  scalar T_prev1 = x;                                     // T_{n-1}
  scalar T_curr = ::leb::detail::constants<scalar>::zero;

  scalar two_x = ::leb::detail::constants<scalar>::two * x;

  for (int k = 2; k <= n; ++k)
  {
    T_curr = two_x * T_prev1 - T_prev2;
    T_prev2 = T_prev1;
    T_prev1 = T_curr;
  }
  return T_curr;
}

/**
 * Evaluate Chebyshev series sum: Σ coefs[k] * T_k(x)
 * Uses stable Clenshaw recurrence algorithm.
 */
template<typename scalar>
inline scalar chebyshev_sum(std::span<const scalar> coefs, scalar x) noexcept
{
  const int n = static_cast<int>(coefs.size()) - 1;
  if (n < 0)
    return ::leb::detail::constants<scalar>::zero;

  scalar b1 = ::leb::detail::constants<scalar>::zero;
  scalar b2 = ::leb::detail::constants<scalar>::zero;
  scalar two_x = ::leb::detail::constants<scalar>::two * x;

  for (int k = n; k >= 0; --k)
  {
    scalar alpha_x = (k == 0) ? x : two_x;
    scalar result = alpha_x * b1 - b2 + coefs[k];

    b2 = b1;
    b1 = result;
  }
  return b1;
}

/**
 * Compute Chebyshev moments from samples.
 * μ_k = Σᵢ w[i] * T_k(x[i])
 */
template<typename scalar>
inline vector<scalar> chebyshev_moments(std::span<const scalar> x, std::span<const scalar> w, std::size_t n_moments)
{
  vector<scalar> moments(n_moments);
  // Initialize to zero handled by vector constructor

  for (std::size_t i = 0; i < x.size(); ++i)
  {
    scalar xi = x[i];
    scalar wi = w[i];

    scalar T_prev2 = ::leb::detail::constants<scalar>::one; // T_0
    scalar T_prev1 = xi;                                    // T_1
    scalar two_xi = ::leb::detail::constants<scalar>::two * xi;

    if (n_moments > 0)
      moments[0] += wi * T_prev2;
    if (n_moments > 1)
      moments[1] += wi * T_prev1;

    for (std::size_t k = 2; k < n_moments; ++k)
    {
      scalar T_curr = two_xi * T_prev1 - T_prev2;
      moments[k] += wi * T_curr;
      T_prev2 = T_prev1;
      T_prev1 = T_curr;
    }
  }
  return moments;
}

/**
 * Compute weighted Chebyshev moments from samples.
 * μ_k = Σᵢ w[i] * f[i] * T_k(x[i])
 */
template<typename scalar>
inline vector<scalar> chebyshev_weighted_moments(std::span<const scalar> x, std::span<const scalar> f,
                                                 std::span<const scalar> w, std::size_t n_moments)
{
  vector<scalar> moments(n_moments);

  for (std::size_t i = 0; i < x.size(); ++i)
  {
    scalar xi = x[i];
    scalar wfi = w[i] * f[i];

    scalar T_prev2 = ::leb::detail::constants<scalar>::one;
    scalar T_prev1 = xi;
    scalar two_xi = ::leb::detail::constants<scalar>::two * xi;

    if (n_moments > 0)
      moments[0] += wfi * T_prev2;
    if (n_moments > 1)
      moments[1] += wfi * T_prev1;

    for (std::size_t k = 2; k < n_moments; ++k)
    {
      scalar T_curr = two_xi * T_prev1 - T_prev2;
      moments[k] += wfi * T_curr;
      T_prev2 = T_prev1;
      T_prev1 = T_curr;
    }
  }
  return moments;
}

/**
 * Compute <x * T_k> moments from <T_k> moments.
 *
 * Uses identity: x * T_k(x) = (1/α_k) * (T_{k+1} + T_{k-1})
 * where α_0 = 1, α_k = 2 for k >= 1.
 */
template<typename scalar>
inline vector<scalar> x_moments_from_moments(std::span<const scalar> moments)
{
  if (moments.empty())
    return vector<scalar>(0);
  const std::size_t n = moments.size() - 1;
  vector<scalar> x_moments(n);

  for (std::size_t k = 0; k < n; ++k)
  {
    scalar alpha_k = (k == 0) ? ::leb::detail::constants<scalar>::one : ::leb::detail::constants<scalar>::two;

    scalar term_prev = (k > 0) ? moments[k - 1] : ::leb::detail::constants<scalar>::zero;

    x_moments[k] = (moments[k + 1] + term_prev) / alpha_k;
  }
  return x_moments;
}

} // namespace lam::leb
