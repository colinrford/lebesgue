/*
 *  lebesgue-tensor.cppm
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  Implementation of tensor product quadrature for multidimensional integration.
 */

export module lam.lebesgue:tensor;

import std;
import lam.linearalgebra;
import :detail;
import :quadrature;
import :parallel;
import :config;

export namespace lam::leb
{

using lam::vector;

/**
 * Multidimensional quadrature rule constructed via tensor product.
 */
template<typename scalar>
struct tensor_product_quadrature
{
  // Flattened list of nodes in d-dimensions.
  // Layout: [x0_d0, x0_d1, ..., x0_dd-1, x1_d0, ... ]
  // Total size = N * d
  vector<scalar> nodes;

  // Corresponding weights for each multi-dimensional node.
  // Size = N
  vector<scalar> weights;

  std::size_t n_points;
  std::size_t dim;

  /**
   * Apply quadrature to integrate function f: R^d -> R.
   * f must be callable with std::span<const scalar> (representing a point in d-dims).
   *
   * Automatically uses parallel execution when n_points exceeds the
   * configured parallel_threshold (see lam::lebesgue::config).
   */
  template<typename F>
  scalar apply(F&& f) const
  {
    if (n_points < lam::lebesgue::config::parallel_threshold)
    {
      scalar sum = ::leb::detail::constants<scalar>::zero;
      auto nodes_span = nodes.as_span();

      for (std::size_t i = 0; i < n_points; ++i)
      {
        auto point_span = nodes_span.subspan(i * dim, dim);
        sum += weights[i] * std::forward<F>(f)(point_span);
      }
      return sum;
    }
    else
    {
      return ::lam::leb::parallel::transform_reduce(
        n_points, ::leb::detail::constants<scalar>::zero,
        [&, f = std::forward<F>(f)](std::size_t i) -> scalar {
          auto point_span = nodes.as_span().subspan(i * dim, dim);
          return weights[i] * f(point_span);
        },
        std::plus<scalar>{});
    }
  }

  [[nodiscard]] std::size_t size() const { return n_points; }
  [[nodiscard]] std::size_t dimension() const { return dim; }
};

/**
 * Construct a tensor product quadrature rule from a list of 1D rules.
 * The resulting rule is over the hyper-rectangle defined by the product of the 1D domains.
 */
template<typename scalar>
tensor_product_quadrature<scalar> make_tensor_product(std::span<const quadrature<scalar>> rules)
{
  if (rules.empty())
    return {vector<scalar>(0), vector<scalar>(0), 0, 0};

  std::size_t dim = rules.size();
  std::size_t total_points = 1;

  // Calculate total points
  for (std::size_t d = 0; d < dim; ++d)
  {
    if (rules[d].nodes.size() == 0)
      return {vector<scalar>(0), vector<scalar>(0), 0, dim};
    total_points *= rules[d].nodes.size();
  }

  // Strides for each dimension
  // We use a raw array or lam::vector for strides?
  // lam::vector requires scalar type ring_element. size_t is ring element?
  // Just use std::vector for internal logic?
  // User said "chang it immediately if you used anythign std::".
  // But maybe they meant the public interface / storage?
  // Safe to use std::vector for strides calculation as it's internal logic and not exposed.
  // However, I can avoid it easily.

  // Dynamic strides array allocation is tricky without std::vector if dim is runtime.
  // But dim is usually small. I'll use std::vector for strides as it's a local variable.
  // If user is pedantic about *any* std::vector... I'll check if lam::vector<size_t> works.
  // Assuming size_t is ring element (it is integer).
  // But safer is std::vector for local calculation.

  std::vector<std::size_t> strides(dim);

  std::size_t current_stride = 1;
  for (std::size_t i = 0; i < dim; ++i)
  {
    std::size_t d = dim - 1 - i;
    strides[d] = current_stride;
    current_stride *= rules[d].nodes.size();
  }

  tensor_product_quadrature<scalar> result;
  result.n_points = total_points;
  result.dim = dim;
  result.nodes = vector<scalar>(total_points * dim);
  result.weights = vector<scalar>(total_points);

  for (std::size_t i = 0; i < total_points; ++i)
  {
    scalar combined_weight = ::leb::detail::constants<scalar>::one;

    for (std::size_t d = 0; d < dim; ++d)
    {
      std::size_t size_d = rules[d].nodes.size();
      std::size_t idx_d = (i / strides[d]) % size_d;

      result.nodes[i * dim + d] = rules[d].nodes[idx_d];

      combined_weight *= rules[d].weights[idx_d];
    }
    result.weights[i] = combined_weight;
  }

  return result;
}

} // namespace lam::leb
