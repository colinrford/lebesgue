/* 
 *  lebesgue.cppm – based on Java implementation of Vladislav Malyshkin
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 *
 *  Primary module interface exporting all sub-modules (detail, basis, gram, quadrature).
 */

export module lam.lebesgue;

export import :detail;
export import :basis;
export import :gram;
export import :quadrature;

import std;

export namespace lam
{
using leb::gauss_legendre;
using leb::gauss_quadrature_from_moments;
using leb::lebesgue_quadrature_data;
using leb::lebesgue_quadrature_from_moments;
using leb::lebesgue_quadrature_from_samples;
using leb::quadrature;

using leb::chebyshev_eval;
using leb::chebyshev_moments;
using leb::chebyshev_sum;
} // namespace lam
