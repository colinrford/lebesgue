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


template<typename P, typename T>
concept basis_policy_c = requires(int k, T x, T t_prev, T t_prev2) {
  // Must provide recurrence: P_k(x) = f(x, P_{k-1}, P_{k-2})
  { P::evaluate_recurrence(k, x, t_prev, t_prev2) } -> std::convertible_to<T>;

  // Must provide Shift Parameters (a_k, b_k) for Position Operator X
  // x P_k = b_k P_{k+1} + a_k P_k + b_{k-1} P_{k-1}
  // Returns pair {a_k, b_k}
  { P::jacobi_parameters(k) } -> std::convertible_to<std::pair<double, double>>;
};

struct chebyshev_policy
{
  template<typename T>
  static constexpr T evaluate_recurrence(int, T x, T t_prev, T t_prev2)
  { return 2.0 * x * t_prev - t_prev2; }

  static constexpr std::pair<double, double> jacobi_parameters(int)
  {
    // x T_k = 0.5 T_{k+1} + 0.5 T_{k-1}
    // a_k = 0 (symmetric), b_k = 0.5 (except k=0? no, handled by special case usually)
    return {0.0, 0.5};
  }
};

// ============================================================================
// NTTP Hierarchical Policy: Uses polynomial_nttp with composition identity
// T_{mn}(x) = T_m(T_n(x)) for numerical stability at high degrees.
// Compile-time coefficients, runtime evaluation via Horner.
// ============================================================================

namespace nttp_detail
{
// Compile-time Chebyshev polynomial generation (memoized)
template<std::size_t N>
struct chebyshev_coeffs
{
  static constexpr auto compute()
  {
    std::array<double, N + 1> c{};
    if constexpr (N == 0)
    {
      c[0] = 1.0;
    }
    else if constexpr (N == 1)
    {
      c[0] = 0.0;
      c[1] = 1.0;
    }
    else
    {
      // T_n = 2x * T_{n-1} - T_{n-2}
      auto c1 = chebyshev_coeffs<N - 1>::value;
      auto c2 = chebyshev_coeffs<N - 2>::value;

      // Shift c1 by x (multiply by x) and multiply by 2
      for (std::size_t i = N; i >= 1; --i)
        c[i] = 2.0 * c1[i - 1];
      c[0] = 0.0;

      // Subtract c2
      for (std::size_t i = 0; i <= N - 2; ++i)
        c[i] -= c2[i];
    }
    return c;
  }

  static constexpr auto value = compute();
};

// Horner evaluation of compile-time coefficient array using FMA
// std::fma(a, b, c) = a*b + c with single rounding for better accuracy
template<std::size_t N>
constexpr double horner_eval(const std::array<double, N + 1>& c, double x)
{
  double result = c[N];
  for (int i = static_cast<int>(N) - 1; i >= 0; --i)
    result = std::fma(result, x, c[i]);
  return result;
}

// Factorization for hierarchical composition
// Uses max base of 8 for better numerical stability (lower coefficient magnitude)
// Recursively factors the inner term for multi-level composition
template<std::size_t N>
struct factors
{
  static constexpr std::size_t MAX_BASE = 6;

  static constexpr std::size_t find_factor()
  {
    if constexpr (N <= MAX_BASE)
      return 1;
    // Prefer factors that keep both parts <= MAX_BASE
    for (std::size_t f = MAX_BASE; f >= 2; --f)
      if (N % f == 0 && N / f <= MAX_BASE)
        return f;
    // Otherwise find any factor <= MAX_BASE
    for (std::size_t f = MAX_BASE; f >= 2; --f)
      if (N % f == 0)
        return f;
    return 1;
  }

  static constexpr std::size_t outer = find_factor();
  static constexpr std::size_t inner = (outer > 1) ? N / outer : N;
  static constexpr bool is_composite = (outer > 1);
};

// Multi-level hierarchical evaluation
// Recursively applies composition: T_{a*b}(x) = T_a(T_b(x))
// With inner term also factored if needed: T_100 = T_4(T_5(T_5(x)))
template<std::size_t N>
double eval_hierarchical(double x)
{
  using F = factors<N>;
  if constexpr (N <= F::MAX_BASE)
    return horner_eval<N>(chebyshev_coeffs<N>::value, x);
  else if constexpr (F::is_composite)
    // Recursive: both outer and inner are factored if needed
    return eval_hierarchical<F::outer>(eval_hierarchical<F::inner>(x));
  else
  {
    // Fallback to recurrence for primes > MAX_BASE
    double t0 = 1.0, t1 = x;
    for (std::size_t k = 2; k <= N; ++k)
    {
      double t2 = 2.0 * x * t1 - t0;
      t0 = t1;
      t1 = t2;
    }
    return t1;
  }
}
} // namespace nttp_detail

struct nttp_hierarchical_policy
{
  template<typename T>
  static constexpr T evaluate_recurrence(int k, T x, T t_prev, T t_prev2)
  {
    // Standard Chebyshev recurrence (used by gram matrix construction)
    return 2.0 * x * t_prev - t_prev2;
  }

  static constexpr std::pair<double, double> jacobi_parameters(int) { return {0.0, 0.5}; }

  // Additional method: Direct evaluation using hierarchical NTTP
  // Can be called explicitly for single-point evaluation
  template<std::size_t N>
  static double evaluate(double x)
  { return nttp_detail::eval_hierarchical<N>(x); }
};


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

  scalar T_prev2 = ::leb::detail::constants<scalar>::one;
  scalar T_prev1 = x;
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

  for (std::size_t i = 0; i < x.size(); ++i)
  {
    scalar xi = x[i];
    scalar wi = w[i];

    scalar T_prev2 = ::leb::detail::constants<scalar>::one;
    scalar T_prev1 = xi;
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
