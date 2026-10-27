#include "matrix_utils.hpp"
#include "matrix.hpp"
#include "intermediate_buffer.hpp"
#include "types.hpp"
#include <array>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <vector>

template <typename T>
Matrix2x<T> matrix_mul(const Matrix2x<T> &a, const Matrix2x<T> &b,
                       enumMatMult alg) {
  switch (alg) {
  case enumMatMult::Default:
    return utils::matrix_mul_alg<T, enumMatMult::Default>(a, b);
  case enumMatMult::Binet:
    return utils::matrix_mul_alg<T, enumMatMult::Binet>(a, b);
  case enumMatMult::Strassen:
    return utils::matrix_mul_alg<T, enumMatMult::Strassen>(a, b);
  case enumMatMult::Laderman:
    return utils::matrix_mul_alg<T, enumMatMult::Laderman>(a, b);
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

