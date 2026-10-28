#include "matrix.hpp"
#include "tests.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

int main() {
  Matrix2x<mfloat> a({{mfloat{2.0f}, mfloat{3.0f}, mfloat{4.0f}},
                      {mfloat{1.0f}, mfloat{4.0f}, mfloat{5.0f}},
                      {mfloat{1.0f}, mfloat{2.0f}, mfloat{3.0f}}});

  auto a_view = a.submatrix_view(0, 0, a.rows(), a.cols());

  auto b = inverse(a_view, enumMatMult::Binet);
  std::cout << b << "HEJKA" << std::endl;

  return 0;
}