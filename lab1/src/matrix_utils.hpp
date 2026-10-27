#pragma once

#include "../src/types.hpp"
#include "intermediate_buffer.hpp"

template <typename T> class Matrix2xView;

namespace mutils {

template <typename T>
void default_matrix_mul_accum(const Matrix2xView<T> &a,
                              const Matrix2xView<T> &b, Matrix2xView<T> &acc);

namespace binet {
template <typename T>
void binet_matrix_mul_accum(const Matrix2xView<T> &a, const Matrix2xView<T> &b,
                            Matrix2xView<T> &acc);
} // namespace binet

namespace strassen {
template <typename T>
void strassen_matrix_mul_accum(const Matrix2xView<T> &a,
                               const Matrix2xView<T> &b, Matrix2xView<T> &acc,
                               IntermediateBuffer<T> &pbuf,
                               IntermediateBuffer<T> &uvbuf);
} // namespace strassen

namespace laderman {
template <typename T>
void laderman_matrix_mul_accum(const Matrix2xView<T> &a,
                               const Matrix2xView<T> &b, Matrix2xView<T> &acc,
                               IntermediateBuffer<T> &mbuf,
                               IntermediateBuffer<T> &uvbuf);
} // namespace laderman

template <typename T, enumMatMult Alg>
Matrix2x<T> matrix_mul_alg(const Matrix2x<T> &a, const Matrix2x<T> &b);

} // namespace mutils
