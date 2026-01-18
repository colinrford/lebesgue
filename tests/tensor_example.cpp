/*
 *  tensor_example.cpp
 *  see github.com/colinrford/lebesgue for GPL 3.0 license and for more info
 */

import lam.lebesgue;
namespace leb = lam::leb;
import std;
import lam.linearalgebra; // for vector

int main()
{
  std::cout << "=== Tensor Product Quadrature Example ===\n" << std::endl;

  // Helper to generate explicit Lebesgue rule (Spectral Position Operator)
  auto make_lebesgue_rule = [](int n) {
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
  };

  // 1. Create a 1D rule (Explicit Lebesgue 5-point)
  auto gl5 = make_lebesgue_rule(5);

  // 2. Create a 2D rule by taking the tensor product of two 1D rules
  std::vector<leb::quadrature<double>> rules = {gl5, gl5};

  auto rule_2d = leb::make_tensor_product<double>(rules);

  std::cout << "1D points: " << gl5.nodes.size() << std::endl;
  std::cout << "2D points: " << rule_2d.size() << " (should be 25)" << std::endl;
  std::cout << "Dimensions: " << rule_2d.dimension() << std::endl;

  // 3. Define functions to integrate over [-1, 1]^2
  auto integral_1 = rule_2d.apply([](std::span<const double>) { return 1.0; });

  auto integral_x2_plus_y2 = rule_2d.apply([](std::span<const double> p) {
    double x = p[0];
    double y = p[1];
    return x * x + y * y;
  });

  // Exact values
  // Area of [-1, 1]^2 is 2*2 = 4
  // Integral of x^2 + y^2 is 8/3 = 2.666...

  std::cout << "\nResults:" << std::endl;
  std::cout << "  Int(1) = " << integral_1 << " (Exact: 4.0)" << std::endl;
  std::cout << "  Int(x^2 + y^2) = " << integral_x2_plus_y2 << " (Exact: 2.66666...)" << std::endl;

  // 4. Test 3D case
  std::vector<leb::quadrature<double>> rules_3d = {gl5, gl5, gl5};
  auto rule_3d = leb::make_tensor_product<double>(rules_3d);

  std::cout << "\n3D points: " << rule_3d.size() << " (should be 125)" << std::endl;

  // Volume of [-1, 1]^3 is 8
  auto integral_1_3d = rule_3d.apply([](std::span<const double>) { return 1.0; });
  std::cout << "  Int(1) [3D] = " << integral_1_3d << " (Exact: 8.0)" << std::endl;

  return 0;
}
