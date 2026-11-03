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
  auto inversed = inverse(view, enumMatMult::Laderman);

  auto output = matrix_mul(inversed, matrix, enumMatMult::Strassen) - I;

  std::cout << output << std::endl;

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      assert(std::abs(output(i, j)) < 1e-5);
    }
  }

  // Test Laderman inverse across multiple sizes
  for (int n_test : {3, 4, 5, 6, 7, 9, 10, 12}) {
    Matrix2x<mfloat> I_t(n_test, n_test);
    Matrix2x<mfloat> M_t(n_test, n_test);
    for (int i = 0; i < n_test; ++i) {
      I_t(i, i) = mfloat{1.0f};
      for (int j = 0; j < n_test; ++j)
        M_t(i, j) = static_cast<float>((i * 3 + j * 7 + 1) % 10) / 100.0f;
    }
    auto mat = matrix_mul(M_t.transpose(), M_t, enumMatMult::Laderman) + I_t;
    auto mat_view = mat.submatrix_view(0, 0, n_test, n_test);
    auto inv_lad = inverse_laderman(mat_view);
    auto diff = matrix_mul(inv_lad, mat, enumMatMult::Laderman) - I_t;
    for (int i = 0; i < n_test; ++i) {
      for (int j = 0; j < n_test; ++j) {
        assert(std::abs(diff(i, j)) < 1e-4);
      }
    }
  }
  std::cout << "[TESTS] Laderman inverse tests for sizes 3, 4, 5, 6, 7, 9, 10, "
               "12 passed!"
            << std::endl;

  auto M_view = M.submatrix_view(0, 0, 10, 10);
  auto [L, U] = LU(M_view, enumMatMult::Binet);

  // std::cout << L << std::endl;
  // std::cout << U << std::endl;

  auto lu_mul = matrix_mul(L, U, enumMatMult::Binet) - M;

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      assert(std::abs(lu_mul(i, j)) < 1e-5);
    }
  }

  return 0;
}