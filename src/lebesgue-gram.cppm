/*
 *  lebesgue-gram.cppm – based on Java implementation of Vladislav Malyshkin
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  Construction of the Gram matrix from moments or samples, including parallel implementations.
 */

module;
#ifdef LEB_HAS_TBB
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#endif

export module lam.lebesgue:gram;

import std;
import lam.linearalgebra;
import :detail;
import :parallel;
import :basis;

export namespace lam::leb
{

using lam::matrix;
using lam::vector;
namespace stdr = std::ranges;


/**
 * Build n×n Gram matrix G[j,k] = <T_j, T_k> from moments using standard relation.
 *
 * WARNING: This implementation assumes standard real-based recurrence identity holds.
 * For complex measures, this is approximate or assumes symmetry.
 */
template<typename scalar>
inline matrix<scalar> gram_matrix_from_moments(std::span<const scalar> moments, std::size_t n)
{
  matrix<scalar> G(n, n);
  scalar half = ::leb::detail::constants<scalar>::one / ::leb::detail::constants<scalar>::two;

  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t k = 0; k <= j; ++k)
    {
      std::size_t idx_diff = j - k;
      std::size_t idx_sum = j + k;

      scalar value = half * moments[idx_diff] + half * moments[idx_sum];

      G[j, k] = value;
      // Force symmetry/hermiticity based on type
      if constexpr (std::is_same_v<scalar, std::complex<double>>)
        G[k, j] = ::leb::detail::conj(value);
      else
        G[k, j] = value;
    }
  return G;
}

/**
 * Build Gram matrix directly from samples.
 * G[j,k] = Σᵢ w[i] * T_j(x[i]) * conj(T_k(x[i]))
 * Ensures Hermitian result.
 */
template<typename scalar, basis_policy_c<scalar> Policy = chebyshev_policy>
inline matrix<scalar> gram_matrix_from_samples_sequential(std::span<const scalar> x, std::span<const scalar> w,
                                                          std::size_t n)
{
  matrix<scalar> G(n, n);
  stdr::fill(std::span(G.begin(), G.end()), ::leb::detail::constants<scalar>::zero);

  for (std::size_t i = 0; i < x.size(); ++i)
  {
    vector<scalar> T(n);
    T[0] = ::leb::detail::constants<scalar>::one;
    if (n > 1)
      T[1] = x[i];

    scalar two_x = ::leb::detail::constants<scalar>::two * x[i];
    for (std::size_t k = 2; k < n; ++k)
      T[k] = Policy::evaluate_recurrence(k, x[i], T[k - 1], T[k - 2]);

    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k <= j; ++k)
      { // <T_j, T_k> = w[i] * T_j * conj(T_k)
        scalar contrib = w[i] * T[j] * ::leb::detail::conj(T[k]);
        G[j, k] += contrib;
        if (j != k)
          G[k, j] += ::leb::detail::conj(contrib);
      }
  }
  return G;
}

/**
 * Build weighted Gram matrix from samples.
 * F[j,k] = Σᵢ w[i] * f[i] * T_j(x[i]) * conj(T_k(x[i]))
 */
template<typename scalar>
inline matrix<scalar> weighted_gram_matrix_from_samples(std::span<const scalar> x, std::span<const scalar> f,
                                                        std::span<const scalar> w, std::size_t n)
{
  matrix<scalar> F(n, n);
  stdr::fill(std::span(F.begin(), F.end()), ::leb::detail::constants<scalar>::zero);

  for (std::size_t i = 0; i < x.size(); ++i)
  {
    vector<scalar> T(n);
    T[0] = ::leb::detail::constants<scalar>::one;
    if (n > 1)
      T[1] = x[i];

    scalar two_x = ::leb::detail::constants<scalar>::two * x[i];
    for (std::size_t k = 2; k < n; ++k)
      T[k] = two_x * T[k - 1] - T[k - 2];

    scalar wf = w[i] * f[i];
    for (std::size_t j = 0; j < n; ++j)
    {
      for (std::size_t k = 0; k <= j; ++k)
      {
        scalar contrib = wf * T[j] * ::leb::detail::conj(T[k]);
        F[j, k] += contrib;

        if (j != k)
        {
          scalar contrib_trans = wf * T[k] * ::leb::detail::conj(T[j]);
          F[k, j] += contrib_trans;
        }
      }
    }
  }
  return F;
}


/**
 * Parallel Gram matrix construction using TBB or jthread pool.
 */
template<typename scalar, basis_policy_c<scalar> Policy = chebyshev_policy>
inline matrix<scalar> gram_matrix_from_samples_parallel(std::span<const scalar> x, std::span<const scalar> w,
                                                        std::size_t n)
{
#ifdef LEB_HAS_TBB
  return tbb::parallel_reduce(
    tbb::blocked_range<std::size_t>(0, x.size()), matrix<scalar>(n, n),
    [&](const tbb::blocked_range<std::size_t>& r, matrix<scalar> init) -> matrix<scalar> {
      if (init.rows() != n)
      {
        init = matrix<scalar>(n, n);
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j)
            init[i, j] = ::leb::detail::constants<scalar>::zero;
      }

      vector<scalar> T(n);
      for (std::size_t i = r.begin(); i != r.end(); ++i)
      {
        T[0] = ::leb::detail::constants<scalar>::one;
        if (n > 1)
          T[1] = x[i];

        scalar two_x = ::leb::detail::constants<scalar>::two * x[i];
        for (std::size_t k = 2; k < n; ++k)
          T[k] = Policy::evaluate_recurrence(k, x[i], T[k - 1], T[k - 2]);

        for (std::size_t j = 0; j < n; ++j)
          for (std::size_t k = 0; k <= j; ++k)
          {
            scalar contrib = w[i] * T[j] * ::leb::detail::conj(T[k]);
            init[j, k] += contrib;
            if (j != k)
              init[k, j] += ::leb::detail::conj(contrib);
          }
      }
      return init;
    },
    [](matrix<scalar> a, const matrix<scalar>& b) -> matrix<scalar> {
      for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j)
          a[i, j] += b[i, j];
      return a;
    });
#else // Fallback: jthread pool
  auto& pool = parallel::thread_pool::instance();
  const unsigned n_threads = parallel::thread_count();
  const std::size_t samples = x.size();
  const std::size_t chunk_size = (samples + n_threads - 1) / n_threads;

  lam::vector<matrix<scalar>> partials;
  for (unsigned t = 0; t < n_threads; ++t)
  {
    partials.emplace_back(n, n);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        partials[t][i, j] = ::leb::detail::constants<scalar>::zero;
  }
  std::latch done(n_threads);

  for (unsigned t = 0; t < n_threads; ++t)
  {
    pool.enqueue([&, t, n, &latch = done] {
      std::size_t start = t * chunk_size;
      std::size_t end = std::min(start + chunk_size, samples);

      vector<scalar> T(n);

      for (std::size_t i = start; i < end; ++i)
      {
        T[0] = ::leb::detail::constants<scalar>::one;
        if (n > 1)
          T[1] = x[i];

        scalar two_x = ::leb::detail::constants<scalar>::two * x[i];
        for (std::size_t k = 2; k < n; ++k)
          T[k] = Policy::evaluate_recurrence(k, x[i], T[k - 1], T[k - 2]);

        for (std::size_t j = 0; j < n; ++j)
          for (std::size_t k = 0; k <= j; ++k)
          {
            scalar contrib = w[i] * T[j] * ::leb::detail::conj(T[k]);
            partials[t][j, k] += contrib;
            if (j != k)
              partials[t][k, j] += ::leb::detail::conj(contrib);
          }
      }
      latch.count_down();
    });
  }

  done.wait();

  matrix<scalar> G(n, n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      G[i, j] = ::leb::detail::constants<scalar>::zero;
  for (const auto& p : partials)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        G[i, j] += p[i, j];
  return G;
#endif
}

/**
 * Main entry point: Build Gram matrix directly from samples.
 * Dispatches to Sequential or Parallel based on size.
 */
template<typename scalar, basis_policy_c<scalar> Policy = chebyshev_policy>
inline matrix<scalar> gram_matrix_from_samples(std::span<const scalar> x, std::span<const scalar> w, std::size_t n)
{
  if (x.size() >= 1000 && n > 5)
    return gram_matrix_from_samples_parallel<scalar, Policy>(x, w, n);
  else
    return gram_matrix_from_samples_sequential<scalar, Policy>(x, w, n);
}

} // namespace lam::leb
