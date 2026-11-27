#include "matrix_utils.hpp"
#include "matrix.hpp"
#include "intermediate_buffer.hpp"
#include "types.hpp"
#include "../../lab3/src/vector.hpp"
#include <array>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

template <typename T>
Vector<T> Matrix2x<T>::operator*(const Vector<T> &rhs) const {
  if (cols() != rhs.len()) {
    throw std::invalid_argument(
        "Matrix2x: size mismatch for matrix-vector multiplication");
  }

  Vector<T> res(rows());

  for (size_type i = 0; i < rows(); ++i) {
    T sum = static_cast<T>(0);

    for (size_type j = 0; j < cols(); ++j) {
      sum += (*this)(i, j) * rhs[j];
    }

    res[i] = sum;
  }

  return res;
}

template Vector<float> Matrix2x<float>::operator*(const Vector<float> &) const;
template Vector<double>
    Matrix2x<double>::operator*(const Vector<double> &) const;
template Vector<mfloat>
    Matrix2x<mfloat>::operator*(const Vector<mfloat> &) const;
template Vector<mdouble>
    Matrix2x<mdouble>::operator*(const Vector<mdouble> &) const;

template <typename T>
Matrix2x<T> matrix_mul(const Matrix2x<T> &a, const Matrix2x<T> &b,
                       enumMatMult alg) {
  switch (alg) {
  case enumMatMult::Default:
    return mutils::matrix_mul_alg<T, enumMatMult::Default>(a, b);
  case enumMatMult::Binet:
    return mutils::matrix_mul_alg<T, enumMatMult::Binet>(a, b);
  case enumMatMult::Strassen:
    return mutils::matrix_mul_alg<T, enumMatMult::Strassen>(a, b);
  case enumMatMult::Laderman:
    return mutils::matrix_mul_alg<T, enumMatMult::Laderman>(a, b);
  default:
    assert(false);
    return {};
  }
}

template Matrix2x<mfloat> matrix_mul<mfloat>(const Matrix2x<mfloat> &,
                                             const Matrix2x<mfloat> &,
                                             enumMatMult alg);
template Matrix2x<mdouble> matrix_mul<mdouble>(const Matrix2x<mdouble> &,
                                               const Matrix2x<mdouble> &,
                                               enumMatMult alg);

