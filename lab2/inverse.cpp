#include "../lab1/src/matrix.hpp"
#include "../lab1/intermediate_buffer.hpp"
#include "../lab1/src/types.hpp"
#include <cassert>

template <typename T>
Matrix2x<T> inverse(const Matrix2x<T> &a, enumMatMult alg) {
  assert(a.rows() == a.cols());
  // TODO: implement inverse using the selected multiplication algorithm
  return {};
}

template Matrix2x<mfloat> inverse<mfloat>(const Matrix2x<mfloat> &, enumMatMult);
template Matrix2x<mdouble> inverse<mdouble>(const Matrix2x<mdouble> &, enumMatMult);
