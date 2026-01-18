/*
 *  lebesgue-detail.cppm – based on Java implementation of Vladislav Malyshkin
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  Basic constants and utility functions (e.g., conjugation) for scalar and complex types.
 */

module;

export module lam.lebesgue:detail;

import std;

namespace leb::detail
{
template<typename T>
struct constants;

template<>
struct constants<double>
{
  static constexpr double zero = 0.0;
  static constexpr double one = 1.0;
  static constexpr double two = 2.0;
};

template<>
struct constants<std::complex<double>>
{
  static constexpr std::complex<double> zero = {0.0, 0.0};
  static constexpr std::complex<double> one = {1.0, 0.0};
  static constexpr std::complex<double> two = {2.0, 0.0};
};

template<typename T>
constexpr T conj(const T& x)
{
  if constexpr (std::is_same_v<T, std::complex<double>>)
    return std::conj(x);
  else
    return x;
}

} // namespace leb::detail
