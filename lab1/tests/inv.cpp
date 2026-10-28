#include "matrix.hpp"
#include <cassert>

int main() {
  Matrix2x<mfloat> I(10, 10);
  Matrix2x<mfloat> M(10, 10);
  for (int i = 0; i < 10; ++i) {
    I(i, i) = mfloat{1.0f};
    for (int j = 0; j < 10; ++j)
      M(i, j) = static_cast<float>(rand() % 10) / 100;
  }
  auto matrix = matrix_mul(M.transpose(), M, enumMatMult::Binet) + I;

  auto view = matrix.submatrix_view(0, 0, 10, 10);
  auto inversed = inverse(view, enumMatMult::Strassen);

  auto output = matrix_mul(inversed, matrix, enumMatMult::Strassen) - I;

  // std::cout << output << std::endl;

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      assert(std::abs(output(i, j)) < 1e-5);
    }
  }

  auto M_view = M.submatrix_view(0, 0, 10, 10);
  auto [L, U] = LU(M_view, enumMatMult::Binet);

  std::cout << L << std::endl;
  std::cout << U << std::endl;

  auto lu_mul = matrix_mul(L, U, enumMatMult::Binet) - M;

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      assert(std::abs(lu_mul(i, j)) < 1e-5);
    }
  }

  return 0;
}