#include "matrix.hpp"
#include <cassert>
#include <memory>
#include <vector>
#include <stdexcept>

template <typename T>
class StrassenMulBuffer {
public:
    StrassenMulBuffer() = delete;
    explicit StrassenMulBuffer(std::size_t buf_size) 
        : m_size(buf_size)
        , m_data(::new T[buf_size]) 
    {}

    Matrix2xView<T> query_push_matrix_view(std::size_t view_rows, std::size_t view_cols) { 
        const std::size_t req = view_rows * view_cols;

        if (m_pointer + req > m_size) {
            throw std::runtime_error("Overflow");
        }

        const auto mv = Matrix2xView<T>(m_data.get() + m_pointer, view_cols, 0, 0, view_rows, view_cols);
        m_pointer += req;
        return mv;
    }

private:
    std::unique_ptr<T[]> m_data = nullptr;
    std::size_t m_size = 0;
    std::size_t m_pointer = 0;
};

// @TODO: SIMD version?
template <typename T>
void default_matrix_mul(const Matrix2xView<T>& a, const Matrix2xView<T>& b, Matrix2xView<T>& c) {
    assert(a.cols() == b.rows());
    assert(a.rows() == c.rows() && b.cols() == c.cols());

    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            T sum = static_cast<T>(0); 
            
            for (std::size_t k = 0; k < a.cols(); ++k) {
                sum += a(i, k) * b(k, j);
            }

            c(i, j) += sum;
        }
    }
}

template <typename T>
void strassen_matrix_mul(const Matrix2xView<T>& a, 
                         const Matrix2xView<T>& b, 
                         Matrix2xView<T>& c, 
                         StrassenMulBuffer<T>& pbuf,
                         StrassenMulBuffer<T>& uvbuf) {
    assert(a.cols() == b.rows());
    assert(a.rows() == a.cols() && b.rows() == b.cols()); 
    assert(a.rows() == c.rows() && b.cols() == c.cols());

    const auto n = a.rows();
    static constexpr std::size_t StrassenBaseBlockSize = 32;

    if (n <= StrassenBaseBlockSize) {
        default_matrix_mul(a, b, c);
        return;
    }

    const bool even = n & 1;
    const auto n_div = n / 2;

    auto c11 = Matrix2xView<T>(c, 0, 0, n_div, n_div);
    auto c12 = Matrix2xView<T>(c, 0, n_div, n_div, n_div);
    auto c21 = Matrix2xView<T>(c, n_div, 0, n_div, n_div);
    auto c22 = Matrix2xView<T>(c, n_div, n_div, n_div, n_div);

    const auto a11 = Matrix2xView<T>(a, 0, 0, n_div, n_div);
    const auto a12 = Matrix2xView<T>(a, 0, n_div, n_div, n_div);
    const auto a21 = Matrix2xView<T>(a, n_div, 0, n_div, n_div);
    const auto a22 = Matrix2xView<T>(a, n_div, n_div, n_div, n_div);

    const auto b11 = Matrix2xView<T>(b, 0, 0, n_div, n_div);
    const auto b12 = Matrix2xView<T>(b, 0, n_div, n_div, n_div);
    const auto b21 = Matrix2xView<T>(b, n_div, 0, n_div, n_div);
    const auto b22 = Matrix2xView<T>(b, n_div, n_div, n_div, n_div);

    auto p1 = pbuf.query_push_matrix_view(n_div, n_div);
    auto p2 = pbuf.query_push_matrix_view(n_div, n_div);
    auto p3 = pbuf.query_push_matrix_view(n_div, n_div);
    auto p4 = pbuf.query_push_matrix_view(n_div, n_div);
    auto p5 = pbuf.query_push_matrix_view(n_div, n_div);
    auto p6 = pbuf.query_push_matrix_view(n_div, n_div);
    auto p7 = pbuf.query_push_matrix_view(n_div, n_div);

    p1.fill(0);
    p2.fill(0);
    p3.fill(0);
    p4.fill(0);
    p5.fill(0);
    p6.fill(0);
    p7.fill(0);

    auto u = uvbuf.query_push_matrix_view(n_div, n_div);
    auto v = uvbuf.query_push_matrix_view(n_div, n_div);

    // P1 = (A11 + A22) * (B11 + B22)
    Matrix2xView<T>::add_accum(a11, a22, u);
    Matrix2xView<T>::add_accum(b11, b22, v);
    strassen_matrix_mul(u, v, p1, pbuf, uvbuf);

    // P2 = (A21 + A22) * B11
    Matrix2xView<T>::add_accum(a21, a22, u);
    strassen_matrix_mul(u, b11, p2, pbuf, uvbuf);

    // P3 = A11 * (B12 - B22)
    Matrix2xView<T>::sub_accum(b12, b22, v);
    strassen_matrix_mul(a11, v, p3, pbuf, uvbuf);

    // P4 = A22 * (B21 - B11)
    Matrix2xView<T>::sub_accum(b21, b11, v);
    strassen_matrix_mul(a22, v, p4, pbuf, uvbuf);

    // P5 = (A11 + A12) * B22
    Matrix2xView<T>::add_accum(a11, a12, u);
    strassen_matrix_mul(u, b22, p5, pbuf, uvbuf);

    // P6 = (A21 - A11) * (B11 + B12)
    Matrix2xView<T>::sub_accum(a21, a11, u);
    Matrix2xView<T>::add_accum(b11, b12, v);
    strassen_matrix_mul(u, v, p6, pbuf, uvbuf);

    // P7 = (A12 - A22) * (B21 + B22)
    Matrix2xView<T>::sub_accum(a12, a22, u);
    Matrix2xView<T>::add_accum(b21, b22, v);
    strassen_matrix_mul(u, v, p7, pbuf, uvbuf);

    // Move that directly onto c submatrices
    for (std::size_t i = 0; i < n_div; ++i) {
        for (std::size_t j = 0; j < n_div; ++j) {
            c11(i, j) += p1(i, j) + p4(i, j) - p5(i, j) + p7(i, j);
            c12(i, j) += p3(i, j) + p5(i, j);
            c21(i, j) += p2(i, j) + p4(i, j);
            c22(i, j) += p1(i, j) - p2(i, j) + p3(i, j) + p6(i, j);
        }
    }

    if (even) { // means we're done, because entire matrix is already multiplied
        return;
    }

    // Multiply cropped row and column

    const std::size_t last = n - 1;

    for (std::size_t i = 0; i < last; ++i) {
        for (std::size_t j = 0; j < last; ++j) {
            c(i, j) += a(i, last) * b(last, j);
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        T sum = static_cast<T>(0);
        
        for (std::size_t k = 0; k < n; ++k) {
            sum += a(i, k) * b(k, last);
        }

        c(i, last) += sum;
    }

    for (std::size_t j = 0; j < last; ++j) {
        T sum = static_cast<T>(0);

        for (std::size_t k = 0; k < n; ++k) {
            sum += a(last, k) * b(k, j);
        }

        c(last, j) += sum;
    }
}

template <typename T>
void binet_matrix_mul(const Matrix2xView<T>& a, const Matrix2xView<T>& b, Matrix2xView<T>& c) {
    assert(a.cols() == b.rows());
    assert(a.rows() == a.cols() && b.rows() == b.cols()); 
    assert(a.rows() == c.rows() && b.cols() == c.cols());

    static constexpr std::size_t BinetBaseBlockSize = 4;

    const auto n = a.rows();

    if (n <= BinetBaseBlockSize) {
        default_matrix_mul(a, b, c);
        return;
    }

    const auto n_div_l = n / 2;
    const auto n_div_r = n - n_div_l;

    // Submatrices of C
    auto c11 = Matrix2xView<T>(c, 0, 0, n_div_l, n_div_l);
    auto c12 = Matrix2xView<T>(c, 0, n_div_l, n_div_l, n_div_r);
    auto c21 = Matrix2xView<T>(c, n_div_l, 0, n_div_r, n_div_l);
    auto c22 = Matrix2xView<T>(c, n_div_l, n_div_l, n_div_r, n_div_r);

    // Submatrices of A
    const auto a11 = Matrix2xView<T>(a, 0, 0, n_div_l, n_div_l);
    const auto a12 = Matrix2xView<T>(a, 0, n_div_l, n_div_l, n_div_r);
    const auto a21 = Matrix2xView<T>(a, n_div_l, 0, n_div_r, n_div_l);
    const auto a22 = Matrix2xView<T>(a, n_div_l, n_div_l, n_div_r, n_div_r);

    // Submatrices of B
    const auto b11 = Matrix2xView<T>(b, 0, 0, n_div_l, n_div_l);
    const auto b12 = Matrix2xView<T>(b, 0, n_div_l, n_div_l, n_div_r);
    const auto b21 = Matrix2xView<T>(b, n_div_l, 0, n_div_r, n_div_l);
    const auto b22 = Matrix2xView<T>(b, n_div_l, n_div_l, n_div_r, n_div_r);
    
    // Final stage: perform operations on submatrices

    // C11 = A11 * B11 + A12 * B21
    binet_matrix_mul(a11, b11, c11);
    binet_matrix_mul(a12, b21, c11);

    // C12 = A11 * B12 + A12 * B22
    binet_matrix_mul(a11, b12, c12);
    binet_matrix_mul(a12, b22, c12);

    // C21 = A21 * B11 + A22 * B21
    binet_matrix_mul(a21, b11, c21);
    binet_matrix_mul(a22, b21, c21);

    // C22 = A21 * B12 + A22 * B22
    binet_matrix_mul(a21, b12, c22);
    binet_matrix_mul(a22, b22, c22);
}

template <typename T>
Matrix2x<T> matrix_mul(const Matrix2x<T>& a, const Matrix2x<T>& b, enumMatMult alg) {
    assert(a.cols() == b.rows());

    const auto a_rows = a.rows();
    const auto a_cols = a.cols();
    const auto b_rows = b.rows();
    const auto b_cols = b.cols();
    const auto res_rows = a_rows;
    const auto res_cols = b_cols;

    Matrix2xView<T> a_view(a.data(), a_cols, 0, 0, a_rows, a_cols);
    Matrix2xView<T> b_view(b.data(), b_cols, 0, 0, b_rows, b_cols);

    Matrix2x<T> res(res_rows, res_cols);
    Matrix2xView<T> res_view(res.data(), res_cols, 0, 0, res_rows, res_cols);

    switch (alg) {
    case enumMatMult::Default:
        default_matrix_mul(a_view, b_view, res_view);
        break;
    case enumMatMult::Binet:
        binet_matrix_mul(a_view, b_view, res_view);
        break;
    default: 
        assert(false);
    }

    return res;
}

template Matrix2x<mfloat> matrix_mul<mfloat>(const Matrix2x<mfloat>&, const Matrix2x<mfloat>&, enumMatMult alg);
template Matrix2x<mdouble> matrix_mul<mdouble>(const Matrix2x<mdouble>&, const Matrix2x<mdouble>&, enumMatMult alg);