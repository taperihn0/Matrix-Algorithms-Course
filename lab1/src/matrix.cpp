#include "matrix.hpp"

template <typename T>
void default_matrix_mul(const Matrix2xView<T>& a, const Matrix2xView<T>& b, Matrix2xView<T>& c) {
    assert(a.rows() == b.cols());

    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            T sum = 0; 

            for (std::size_t k = 0; k < a.cols(); ++k) {
                sum += a(i, k) * b(k, j);
            }

            c.at(i, j) += sum;
        }
    }

    return;
}

template <typename T>
void binet_matrix_mul(const Matrix2xView<T>& a, const Matrix2xView<T>& b, Matrix2xView<T>& c) {
    assert(a.cols() == b.rows());
    assert(a.rows() == c.rows() && b.cols() == c.cols());

    static constexpr std::size_t BaseBlockSize = 4;
    const auto n = a.rows();

    if (n <= BaseBlockSize) {
        default_matrix_mul(a, b, c);
        return;
    }

    const auto n_div_l = n / 2;
    const auto n_div_r = n - n_div_l;

    const auto a11 = Matrix2xView<T>(a, 0, 0, n_div_l, n_div_l);
    const auto a12 = Matrix2xView<T>(a, 0, n_div_l, n_div_l, n_div_r);
    const auto a21 = Matrix2xView<T>(a, n_div_l, 0, n_div_r, n_div_l);
    const auto a22 = Matrix2xView<T>(a, n_div_l, n_div_l, n_div_r, n_div_r);

    const auto b11 = Matrix2xView<T>(b, 0, 0, n_div_l, n_div_l);
    const auto b12 = Matrix2xView<T>(b, 0, n_div_l, n_div_l, n_div_r);
    const auto b21 = Matrix2xView<T>(b, n_div_l, 0, n_div_r, n_div_l);
    const auto b22 = Matrix2xView<T>(b, n_div_l, n_div_l, n_div_r, n_div_r);

    auto c11 = Matrix2xView<T>(c, 0, 0, n_div_l, n_div_l);
    auto c12 = Matrix2xView<T>(c, 0, n_div_l, n_div_l, n_div_r);
    auto c21 = Matrix2xView<T>(c, n_div_l, 0, n_div_r, n_div_l);
    auto c22 = Matrix2xView<T>(c, n_div_l, n_div_l, n_div_r, n_div_r);
    
    binet_matrix_mul(a11, b11, c11);
    binet_matrix_mul(a12, b21, c11);

    binet_matrix_mul(a11, b12, c12);
    binet_matrix_mul(a12, b22, c12);

    binet_matrix_mul(a21, b11, c21);
    binet_matrix_mul(a22, b21, c21);

    binet_matrix_mul(a21, b12, c22);
    binet_matrix_mul(a22, b22, c22);
}

template <typename T>
Matrix2x<T> matrix_mul(const Matrix2x<T>& a, const Matrix2x<T>& b, enumMatMult alg) {
    assert(a.cols() == b.rows());

    const auto n = a.cols();
    const auto m = b.rows();

    Matrix2x<T> res(n, m);
    Matrix2xView view(res.data(), m, 0, 0, n, m);

    switch (alg) {
    case enumMatMult::Default:
        default_matrix_mul(a, b, view);
    case enumMatMult::Binet:
        binet_matrix_mul(a, b, view);
    default: 
        assert(false);
    }

    return res;
}

template Matrix2x<mfloat> matrix_mul<mfloat>(const Matrix2x<mfloat>&, const Matrix2x<mfloat>&, enumMatMult alg);
template Matrix2x<mdouble> matrix_mul<mdouble>(const Matrix2x<mdouble>&, const Matrix2x<mdouble>&, enumMatMult alg);
