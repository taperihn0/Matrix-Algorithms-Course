#include "lu.hpp"
#include "lab1/src/matrix_utils.hpp"

#include <cassert>
#include <utility>

namespace {

template <typename T>
void mul_views(Matrix2xView<T> &a, Matrix2xView<T> &b, Matrix2xView<T> &acc,
               enumMatMult type) {
  switch (type) {
  case enumMatMult::Default:
    mutils::matrix_mul_alg<T, enumMatMult::Default>(a, b, acc);
    break;
  case enumMatMult::Binet:
    mutils::matrix_mul_alg<T, enumMatMult::Binet>(a, b, acc);
    break;
  case enumMatMult::Strassen:
    mutils::matrix_mul_alg<T, enumMatMult::Strassen>(a, b, acc);
    break;
  case enumMatMult::Laderman:
    mutils::matrix_mul_alg<T, enumMatMult::Laderman>(a, b, acc);
    break;
  default:
    assert(false);
    break;
  }
}

// Matrix2x<T>::inverse works on whole matrices, so invert a copy of the view.
template <typename T> Matrix2x<T> inverse_of(Matrix2xView<T> &a, enumMatMult type) {
  Matrix2x<T> src(a.rows(), a.cols());

  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t j = 0; j < a.cols(); ++j) {
      src(i, j) = a(i, j);
    }
  }

  Matrix2x<T> acc(a.rows(), a.cols());
  return src.inverse(acc, type);
}

} // namespace

template <typename T>
std::pair<Matrix2x<T>, Matrix2x<T>> LU(Matrix2xView<T> &a, enumMatMult type) {
  const std::size_t n = a.rows();

  assert(n == a.cols());

  // Case n = 1
  if (n == 1) {
    Matrix2x<T> L(1, 1);
    Matrix2x<T> U(1, 1);

    L(0, 0) = static_cast<T>(1);
    U(0, 0) = a(0, 0);

    return {std::move(L), std::move(U)};
  }

  const std::size_t n1 = n / 2;
  const std::size_t n2 = n - n1;

  // Split to blocks
  auto a11 = Matrix2xView<T>(a, 0, 0, n1, n1);
  auto a12 = Matrix2xView<T>(a, 0, n1, n1, n2);
  auto a21 = Matrix2xView<T>(a, n1, 0, n2, n1);
  auto a22 = Matrix2xView<T>(a, n1, n1, n2, n2);

  // [L11, U11] = LU(A11)
  auto [L11, U11] = LU(a11, type);

  // inv(U11)
  auto U11_view = U11.submatrix_view(0, 0, n1, n1);
  auto inv_U11 = inverse_of(U11_view, type);
  auto inv_U11_view = inv_U11.submatrix_view(0, 0, n1, n1);

  // L21 = A21 * inv(U11)
  Matrix2x<T> L21(n2, n1);
  auto L21_view = L21.submatrix_view(0, 0, n2, n1);

  mul_views(a21, inv_U11_view, L21_view, type);

  // inv(L11)
  auto L11_view = L11.submatrix_view(0, 0, n1, n1);
  auto inv_L11 = inverse_of(L11_view, type);
  auto inv_L11_view = inv_L11.submatrix_view(0, 0, n1, n1);

  // U12 = inv(L11) * A12
  Matrix2x<T> U12(n1, n2);
  auto U12_view = U12.submatrix_view(0, 0, n1, n2);

  mul_views(inv_L11_view, a12, U12_view, type);

  // S = A22 - L21 * U12
  Matrix2x<T> product(n2, n2);
  auto product_view = product.submatrix_view(0, 0, n2, n2);

  mul_views(L21_view, U12_view, product_view, type);

  Matrix2x<T> S(n2, n2);

  for (std::size_t i = 0; i < n2; ++i) {
    for (std::size_t j = 0; j < n2; ++j) {
      S(i, j) = a22(i, j) - product(i, j);
    }
  }

  // [L22, U22] = LU(S)
  auto S_view = S.submatrix_view(0, 0, n2, n2);
  auto [L22, U22] = LU(S_view, type);

  // Build L and U
  Matrix2x<T> L(n, n);
  Matrix2x<T> U(n, n);

  auto L_view = L.submatrix_view(0, 0, n, n);
  auto U_view = U.submatrix_view(0, 0, n, n);

  L_view.fill(static_cast<T>(0));
  U_view.fill(static_cast<T>(0));

  // L11 U11 blocks
  for (std::size_t i = 0; i < n1; ++i) {
    for (std::size_t j = 0; j < n1; ++j) {
      L(i, j) = L11(i, j);
      U(i, j) = U11(i, j);
    }
  }

  // U12 block
  for (std::size_t i = 0; i < n1; ++i) {
    for (std::size_t j = 0; j < n2; ++j) {
      U(i, n1 + j) = U12(i, j);
    }
  }

  // L21 block
  for (std::size_t i = 0; i < n2; ++i) {
    for (std::size_t j = 0; j < n1; ++j) {
      L(n1 + i, j) = L21(i, j);
    }
  }

  // L22 U22 blocks
  for (std::size_t i = 0; i < n2; ++i) {
    for (std::size_t j = 0; j < n2; ++j) {
      L(n1 + i, n1 + j) = L22(i, j);
      U(n1 + i, n1 + j) = U22(i, j);
    }
  }

  return {std::move(L), std::move(U)};
}

template std::pair<Matrix2x<mfloat>, Matrix2x<mfloat>>
LU<mfloat>(Matrix2xView<mfloat> &, enumMatMult);

template std::pair<Matrix2x<mdouble>, Matrix2x<mdouble>>
LU<mdouble>(Matrix2xView<mdouble> &, enumMatMult);
