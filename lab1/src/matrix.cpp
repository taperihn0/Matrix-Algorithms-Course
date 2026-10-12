#include "matrix.hpp"

template <typename T, enumMatMult Alg>
Matrix2x<T> matrix_mul(const Matrix2x<T>& a, const Matrix2x<T>& b) {
    (void)a;
    (void)b;
    return Matrix2x<T>();
}

template Matrix2x<mfloat> matrix_mul<mfloat, enumMatMult::Default>(const Matrix2x<mfloat>&, const Matrix2x<mfloat>&);
template Matrix2x<mdouble> matrix_mul<mdouble, enumMatMult::Default>(const Matrix2x<mdouble>&, const Matrix2x<mdouble>&);
template Matrix2x<mfloat> matrix_mul<mfloat, enumMatMult::Binet>(const Matrix2x<mfloat>&, const Matrix2x<mfloat>&);
template Matrix2x<mdouble> matrix_mul<mdouble, enumMatMult::Binet>(const Matrix2x<mdouble>&, const Matrix2x<mdouble>&);
template Matrix2x<mfloat> matrix_mul<mfloat, enumMatMult::Strassen>(const Matrix2x<mfloat>&, const Matrix2x<mfloat>&);
template Matrix2x<mdouble> matrix_mul<mdouble, enumMatMult::Strassen>(const Matrix2x<mdouble>&, const Matrix2x<mdouble>&);
template Matrix2x<mfloat> matrix_mul<mfloat, enumMatMult::Laderman>(const Matrix2x<mfloat>&, const Matrix2x<mfloat>&);
template Matrix2x<mdouble> matrix_mul<mdouble, enumMatMult::Laderman>(const Matrix2x<mdouble>&, const Matrix2x<mdouble>&);
