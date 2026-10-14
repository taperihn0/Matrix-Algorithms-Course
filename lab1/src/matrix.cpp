#include "matrix.hpp"
#include <array>
#include <cassert>
#include <memory>
#include <vector>
#include <stdexcept>

namespace utils {

template <typename T>
class MulIntermediateBuffer {
public:
    MulIntermediateBuffer() = delete;
    explicit MulIntermediateBuffer(std::size_t buf_size) 
        : m_size(buf_size)
        , m_data(::new T[buf_size]) 
    {}

    Matrix2xView<T> query_push_matrix_view(std::size_t view_rows, std::size_t view_cols) { 
        const std::size_t req = view_rows * view_cols;

#ifdef DEBUG
        check_pointer_overflow(req);
#endif

        const auto mv = Matrix2xView<T>(m_data.get() + m_pointer, view_cols, 0, 0, view_rows, view_cols);
        
        m_pointer += req;
        return mv;
    }

private:
#ifdef DEBUG
void check_pointer_overflow(std::size_t req) {
        if (m_pointer + req > m_size) {
            throw std::runtime_error("Overflow");
        }
    }
#endif

    std::unique_ptr<T[]> m_data = nullptr;
    std::size_t m_size = 0uz;
    std::size_t m_pointer = 0uz;
};

} // namespace utils

// @TODO: SIMD version?
template <typename T>
void default_matrix_mul_accum(const Matrix2xView<T>& a, 
                              const Matrix2xView<T>& b, 
                              Matrix2xView<T>& acc) 
{
    assert(a.cols() == b.rows());
    assert(a.rows() == acc.rows() && b.cols() == acc.cols());

    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            T sum = static_cast<T>(0); 
            
            for (std::size_t k = 0; k < a.cols(); ++k) {
                sum += a(i, k) * b(k, j);
            }

            acc(i, j) += sum;
        }
    }
}

namespace strassen {

template <typename T>
void strassen_matrix_mul_accum(const Matrix2xView<T>& a, 
                               const Matrix2xView<T>& b, 
                               Matrix2xView<T>& acc, 
                               utils::MulIntermediateBuffer<T>& pbuf,
                               utils::MulIntermediateBuffer<T>& uvbuf) {
    assert(a.cols() == b.rows());
    assert(a.rows() == a.cols() && b.rows() == b.cols()); 
    assert(a.rows() == acc.rows() && b.cols() == acc.cols());

    const auto n = a.rows();
    static constexpr auto StrassenBaseBlockSize = 4uz;

    if (n <= StrassenBaseBlockSize) {
        default_matrix_mul_accum(a, b, acc);
        return;
    }

    const bool even = !(n & 1);
    const auto n_div = n / 2;

    auto c11 = Matrix2xView<T>(acc, 0, 0, n_div, n_div);
    auto c12 = Matrix2xView<T>(acc, 0, n_div, n_div, n_div);
    auto c21 = Matrix2xView<T>(acc, n_div, 0, n_div, n_div);
    auto c22 = Matrix2xView<T>(acc, n_div, n_div, n_div, n_div);

    const auto a11 = Matrix2xView<T>(a, 0, 0, n_div, n_div);
    const auto a12 = Matrix2xView<T>(a, 0, n_div, n_div, n_div);
    const auto a21 = Matrix2xView<T>(a, n_div, 0, n_div, n_div);
    const auto a22 = Matrix2xView<T>(a, n_div, n_div, n_div, n_div);

    const auto b11 = Matrix2xView<T>(b, 0, 0, n_div, n_div);
    const auto b12 = Matrix2xView<T>(b, 0, n_div, n_div, n_div);
    const auto b21 = Matrix2xView<T>(b, n_div, 0, n_div, n_div);
    const auto b22 = Matrix2xView<T>(b, n_div, n_div, n_div, n_div);

    std::array<Matrix2xView<T>, 7> p;

    for (auto& pi : p) {
        pi = pbuf.query_push_matrix_view(n_div, n_div);
        pi.fill(static_cast<T>(0));
    }

    // Create intermediate buffers for operation results
    auto u = uvbuf.query_push_matrix_view(n_div, n_div);
    auto v = uvbuf.query_push_matrix_view(n_div, n_div);

    // P1 = (A11 + A22) * (B11 + B22)
    u.matrix_add2_accum_into(a11, a22);
    v.matrix_add2_accum_into(b11, b22);
    strassen_matrix_mul_accum(u, v, p[0], pbuf, uvbuf);

    // P2 = (A21 + A22) * B11
    u.matrix_add2_accum_into(a21, a22);
    strassen_matrix_mul_accum(u, b11, p[1], pbuf, uvbuf);

    // P3 = A11 * (B12 - B22)
    v.matrix_sub2_accum_into(b12, b22);
    strassen_matrix_mul_accum(a11, v, p[2], pbuf, uvbuf);

    // P4 = A22 * (B21 - B11)
    v.matrix_sub2_accum_into(b21, b11);
    strassen_matrix_mul_accum(a22, v, p[3], pbuf, uvbuf);

    // P5 = (A11 + A12) * B22
    u.matrix_add2_accum_into(a11, a12);
    strassen_matrix_mul_accum(u, b22, p[4], pbuf, uvbuf);

    // P6 = (A21 - A11) * (B11 + B12)
    u.matrix_sub2_accum_into(a21, a11);
    v.matrix_add2_accum_into(b11, b12);
    strassen_matrix_mul_accum(u, v, p[5], pbuf, uvbuf);

    // P7 = (A12 - A22) * (B21 + B22)
    u.matrix_sub2_accum_into(a12, a22);
    v.matrix_add2_accum_into(b21, b22);
    strassen_matrix_mul_accum(u, v, p[6], pbuf, uvbuf);

    // Move that directly onto C submatrices
    c11.matrix_addsub_accum_into({&p[0], &p[3], &p[6]}, {&p[4]});
    c12.matrix_add2_accum_into(p[2], p[4]);
    c21.matrix_add2_accum_into(p[1], p[3]);
    c22.matrix_addsub_accum_into({&p[0], &p[2], &p[5]}, {&p[1]});

    if (even) { // means we're done, because entire matrix is already multiplied
        return;
    }

    // Multiply cropped row and column

    const auto last = n - 1;

    for (std::size_t i = 0; i < last; ++i) {
        for (std::size_t j = 0; j < last; ++j) {
            acc(i, j) += a(i, last) * b(last, j);
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        T sum = static_cast<T>(0);
        
        for (std::size_t k = 0; k < n; ++k) {
            sum += a(i, k) * b(k, last);
        }

        acc(i, last) += sum;
    }

    for (std::size_t j = 0; j < last; ++j) {
        T sum = static_cast<T>(0);

        for (std::size_t k = 0; k < n; ++k) {
            sum += a(last, k) * b(k, j);
        }

        acc(last, j) += sum;
    }
}

} // namespace strassen

namespace binet {

template <typename T>
void binet_matrix_mul_accum(const Matrix2xView<T>& a, 
                            const Matrix2xView<T>& b, 
                            Matrix2xView<T>& acc) {
    assert(a.cols() == b.rows());
    assert(a.rows() == a.cols() && b.rows() == b.cols()); 
    assert(a.rows() == acc.rows() && b.cols() == acc.cols());

    static constexpr auto BinetBaseBlockSize = 4uz;

    const auto n = a.rows();

    if (n <= BinetBaseBlockSize) {
        default_matrix_mul_accum(a, b, acc);
        return;
    }

    const auto n_div_l = n / 2;
    const auto n_div_r = n - n_div_l;

    // Submatrices of C
    auto c11 = Matrix2xView<T>(acc, 0, 0, n_div_l, n_div_l);
    auto c12 = Matrix2xView<T>(acc, 0, n_div_l, n_div_l, n_div_r);
    auto c21 = Matrix2xView<T>(acc, n_div_l, 0, n_div_r, n_div_l);
    auto c22 = Matrix2xView<T>(acc, n_div_l, n_div_l, n_div_r, n_div_r);

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
    binet_matrix_mul_accum(a11, b11, c11);
    binet_matrix_mul_accum(a12, b21, c11);

    // C12 = A11 * B12 + A12 * B22
    binet_matrix_mul_accum(a11, b12, c12);
    binet_matrix_mul_accum(a12, b22, c12);

    // C21 = A21 * B11 + A22 * B21
    binet_matrix_mul_accum(a21, b11, c21);
    binet_matrix_mul_accum(a22, b21, c21);

    // C22 = A21 * B12 + A22 * B22
    binet_matrix_mul_accum(a21, b12, c22);
    binet_matrix_mul_accum(a22, b22, c22);
}

} // namespace binet

namespace laderman {

template <typename T>
void laderman_matrix_mul_accum(const Matrix2xView<T>& a, 
                               const Matrix2xView<T>& b, 
                               Matrix2xView<T>& acc, 
                               utils::MulIntermediateBuffer<T>& mbuf,
                               utils::MulIntermediateBuffer<T>& uvbuf) {
    assert(a.cols() == b.rows());
    assert(a.rows() == a.cols() && b.rows() == b.cols()); 
    assert(a.rows() == acc.rows() && b.cols() == acc.cols());

    static constexpr auto LadermanBaseBlockSize = 4uz;

    const auto n = a.rows();

    if (n <= LadermanBaseBlockSize) {
        default_matrix_mul_accum(a, b, acc);
        return;
    }

    const bool n_is3div = !(n % 3);
    const auto n_div = n / 3;

    auto c11 = Matrix2xView<T>(acc, 0, 0, n_div, n_div);
    auto c12 = Matrix2xView<T>(acc, 0, n_div, n_div, n_div);
    auto c13 = Matrix2xView<T>(acc, 0, 2 * n_div, n_div, n_div);
    auto c21 = Matrix2xView<T>(acc, n_div, 0, n_div, n_div);
    auto c22 = Matrix2xView<T>(acc, n_div, n_div, n_div, n_div);
    auto c23 = Matrix2xView<T>(acc, n_div, 2 * n_div, n_div, n_div);
    auto c31 = Matrix2xView<T>(acc, 2 * n_div, 0, n_div, n_div);
    auto c32 = Matrix2xView<T>(acc, 2 * n_div, n_div, n_div, n_div);
    auto c33 = Matrix2xView<T>(acc, 2 * n_div, 2 * n_div, n_div, n_div);

    const auto a11 = Matrix2xView<T>(a, 0, 0, n_div, n_div);
    const auto a12 = Matrix2xView<T>(a, 0, n_div, n_div, n_div);
    const auto a13 = Matrix2xView<T>(a, 0, 2 * n_div, n_div, n_div);
    const auto a21 = Matrix2xView<T>(a, n_div, 0, n_div, n_div);
    const auto a22 = Matrix2xView<T>(a, n_div, n_div, n_div, n_div);
    const auto a23 = Matrix2xView<T>(a, n_div, 2 * n_div, n_div, n_div);
    const auto a31 = Matrix2xView<T>(a, 2 * n_div, 0, n_div, n_div);
    const auto a32 = Matrix2xView<T>(a, 2 * n_div, n_div, n_div, n_div);
    const auto a33 = Matrix2xView<T>(a, 2 * n_div, 2 * n_div, n_div, n_div);

    const auto b11 = Matrix2xView<T>(b, 0, 0, n_div, n_div);
    const auto b12 = Matrix2xView<T>(b, 0, n_div, n_div, n_div);
    const auto b13 = Matrix2xView<T>(b, 0, 2 * n_div, n_div, n_div);
    const auto b21 = Matrix2xView<T>(b, n_div, 0, n_div, n_div);
    const auto b22 = Matrix2xView<T>(b, n_div, n_div, n_div, n_div);
    const auto b23 = Matrix2xView<T>(b, n_div, 2 * n_div, n_div, n_div);
    const auto b31 = Matrix2xView<T>(b, 2 * n_div, 0, n_div, n_div);
    const auto b32 = Matrix2xView<T>(b, 2 * n_div, n_div, n_div, n_div);
    const auto b33 = Matrix2xView<T>(b, 2 * n_div, 2 * n_div, n_div, n_div);

    std::array<Matrix2xView<T>, 23> m;

    for (auto& mi : m) {
      mi = mbuf.query_push_matrix_view(n_div, n_div);
      mi.fill(static_cast<T>(0));
    }

    // Create intermediate buffers for operation results
    auto u = uvbuf.query_push_matrix_view(n_div, n_div);
    auto v = uvbuf.query_push_matrix_view(n_div, n_div);

    {
    // m1 = (a11 + a12 + a13 - a21 - a22 - a32 - a33) * b22 
    u.matrix_addsub_accum_into({&a11, &a12, &a13}, {&a21, &a22, &a32, &a33});
    laderman_matrix_mul_accum(u, b22, m[0], mbuf, uvbuf);

    // m2 = (a11 - a21) * (-b12 + b22) 
    u.matrix_addsub_accum_into({&a11}, {&a21});
    v.matrix_addsub_accum_into({&b22}, {&b12});
    laderman_matrix_mul_accum(u, v, m[1], mbuf, uvbuf);

    // m3 = a22 * (-b11 + b12 + b21 - b22 - b23 - b31 + b33) 
    v.matrix_addsub_accum_into({&b12, &b21, &b33}, {&b11, &b22, &b23, &b31});
    laderman_matrix_mul_accum(a22, v, m[2], mbuf, uvbuf);

    // m4 = (-a11 + a21 + a22) * (b11 - b12 + b22) 
    u.matrix_addsub_accum_into({&a21, &a22}, {&a11});
    v.matrix_addsub_accum_into({&b11, &b22}, {&b12});
    laderman_matrix_mul_accum(u, v, m[3], mbuf, uvbuf);

    // m5 = (a21 + a22) * (-b11 + b12) 
    u.matrix_add2_accum_into(a21, a22);
    v.matrix_addsub_accum_into({&b12}, {&b11});
    laderman_matrix_mul_accum(u, v, m[4], mbuf, uvbuf);

    // m6 = a11 * b11 
    laderman_matrix_mul_accum(a11, b11, m[5], mbuf, uvbuf);

    // m7 = (-a11 + a31 + a32) * (b11 - b13 + b23)
    u.matrix_addsub_accum_into({&a31, &a32}, {&a11});
    v.matrix_addsub_accum_into({&b11, &b23}, {&b13});
    laderman_matrix_mul_accum(u, v, m[6], mbuf, uvbuf);

    // m8 = (-a11 + a31) * (b13 - b23) 
    u.matrix_addsub_accum_into({&a31}, {&a11});
    v.matrix_addsub_accum_into({&b13}, {&b23});
    laderman_matrix_mul_accum(u, v, m[7], mbuf, uvbuf);

    // m9 = (a31 + a32) * (-b11 + b13) 
    u.matrix_add2_accum_into(a31, a32);
    v.matrix_addsub_accum_into({&b13}, {&b11});
    laderman_matrix_mul_accum(u, v, m[8], mbuf, uvbuf);

    // m10 = (a11 + a12 + a13 - a22 - a23 - a31 - a32) * b23 
    u.matrix_addsub_accum_into({&a11, &a12, &a13}, {&a22, &a23, &a31, &a32});
    laderman_matrix_mul_accum(u, b23, m[9], mbuf, uvbuf);

    // m11 = a32 * (-b11 + b13 + b21 - b22 - b23 - b31 + b32) 
    v.matrix_addsub_accum_into({&b13, &b21, &b32}, {&b11, &b22, &b23, &b31});
    laderman_matrix_mul_accum(a32, v, m[10], mbuf, uvbuf);

    // m12 = (-a13 + a32 + a33) * (b22 + b31 - b32) 
    u.matrix_addsub_accum_into({&a32, &a33}, {&a13});
    v.matrix_addsub_accum_into({&b22, &b31}, {&b32});
    laderman_matrix_mul_accum(u, v, m[11], mbuf, uvbuf);

    // m13 = (a13 - a33) * (b22 - b32) 
    u.matrix_addsub_accum_into({&a13}, {&a33});
    v.matrix_addsub_accum_into({&b22}, {&b32});
    laderman_matrix_mul_accum(u, v, m[12], mbuf, uvbuf);

    // m14 = a13 * b31 
    laderman_matrix_mul_accum(a13, b31, m[13], mbuf, uvbuf);

    // m15 = (a32 + a33) * (-b31 + b32) 
    u.matrix_add2_accum_into(a32, a33);
    v.matrix_addsub_accum_into({&b32}, {&b31});
    laderman_matrix_mul_accum(u, v, m[14], mbuf, uvbuf);

    // m16 = (-a13 + a22 + a23) * (b23 + b31 - b33) 
    u.matrix_addsub_accum_into({&a22, &a23}, {&a13});
    v.matrix_addsub_accum_into({&b23, &b31}, {&b33});
    laderman_matrix_mul_accum(u, v, m[15], mbuf, uvbuf);

    // m17 = (a13 - a23) * (b23 - b33) 
    u.matrix_addsub_accum_into({&a13}, {&a23});
    v.matrix_addsub_accum_into({&b23}, {&b33});
    laderman_matrix_mul_accum(u, v, m[16], mbuf, uvbuf);

    // m18 = (a22 + a23) * (-b31 + b33) 
    u.matrix_add2_accum_into(a22, a23);
    v.matrix_addsub_accum_into({&b33}, {&b31});
    laderman_matrix_mul_accum(u, v, m[17], mbuf, uvbuf);

    // m19 = a12 * b21 
    laderman_matrix_mul_accum(a12, b21, m[18], mbuf, uvbuf);

    // m20 = a23 * b32 
    laderman_matrix_mul_accum(a23, b32, m[19], mbuf, uvbuf);

    // m21 = a21 * b13 
    laderman_matrix_mul_accum(a21, b13, m[20], mbuf, uvbuf);

    // m22 = a31 * b12 
    laderman_matrix_mul_accum(a31, b12, m[21], mbuf, uvbuf);

    // m23 = a33 * b33 
    laderman_matrix_mul_accum(a33, b33, m[22], mbuf, uvbuf);
  }

  // Assemble results into C submatrices
  // C11 = M6 + M14 + M19
  c11.matrix_add_accum_into({&m[5], &m[13], &m[18]});
  // C12 = M1 + M4 + M5 + M6 + M12 + M14 + M15
  c12.matrix_add_accum_into({&m[0], &m[3], &m[4], &m[5], &m[11], &m[13], &m[14]});
  // C13 = M6 + M7 + M9 + M11 + M14 + M16 + M18
  c13.matrix_add_accum_into({&m[5], &m[6], &m[8], &m[10], &m[13], &m[15], &m[17]});
  // C21 = M2 + M3 + M4 + M6 + M14 + M16 + M17
  c21.matrix_add_accum_into({&m[1], &m[2], &m[3], &m[5], &m[13], &m[15], &m[16]});
  // C22 = M2 + M4 + M5 + M6 + M20
  c22.matrix_add_accum_into({&m[1], &m[3], &m[4], &m[5], &m[19]});
  // C23 = M14 + M16 + M17 + M18 + M21
  c23.matrix_add_accum_into({&m[13], &m[15], &m[16], &m[17], &m[20]});
  // C31 = M6 + M7 + M8 + M11 + M12 + M13 + M14
  c31.matrix_add_accum_into({&m[5], &m[6], &m[7], &m[10], &m[11], &m[12], &m[13]});
  // C32 = M12 + M13 + M14 + M15 + M22
  c32.matrix_add_accum_into({&m[11], &m[12], &m[13], &m[14], &m[21]});
  // C33 = M6 + M7 + M8 + M9 + M23
  c33.matrix_add_accum_into({&m[5], &m[6], &m[7], &m[8], &m[22]});

  if (n_is3div) { // happy case, we are done
    return;
  }

  const auto bound = 3 * n_div;

  for (std::size_t i = 0; i < bound; ++i) {
      for (std::size_t j = 0; j < bound; ++j) {
          T sum = static_cast<T>(0);

          for (std::size_t k = bound; k < n; ++k) {
              sum += a(i, k) * b(k, j);
          }

          acc(i, j) += sum;
      }
  }

  for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = bound; j < n; ++j) {
          T sum = static_cast<T>(0);

          for (std::size_t k = 0; k < n; ++k) {
              sum += a(i, k) * b(k, j);
          }

          acc(i, j) += sum;
      }
  }

  for (std::size_t i = bound; i < n; ++i) {
      for (std::size_t j = 0; j < bound; ++j) {
          T sum = static_cast<T>(0);

          for (std::size_t k = 0; k < n; ++k) {
              sum += a(i, k) * b(k, j);
          }

          acc(i, j) += sum;
      }
  }
}

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

    Matrix2x<T> dst(res_rows, res_cols);
    Matrix2xView<T> dst_view(dst.data(), res_cols, 0, 0, res_rows, res_cols);

    static constexpr auto SafetyBufMargin = 32uz;
    const auto total_size = res_rows * res_cols;

    switch (alg) {
    case enumMatMult::Default: {
        default_matrix_mul_accum(a_view, b_view, dst_view);
        break;
    }
    case enumMatMult::Binet: {
        binet::binet_matrix_mul_accum(a_view, b_view, dst_view);
        break;
    }
    case enumMatMult::Strassen: {
        const auto pbuf_size = static_cast<std::size_t>((7. / 3.) * total_size) + SafetyBufMargin;
        const auto uvbuf_size = static_cast<std::size_t>((2. / 3.) * total_size) + SafetyBufMargin;

        auto pbuf = utils::MulIntermediateBuffer<T>(pbuf_size);
        auto uvbuf = utils::MulIntermediateBuffer<T>(uvbuf_size);

        strassen::strassen_matrix_mul_accum(a_view, b_view, dst_view, pbuf, uvbuf);
        break;
    }
    case enumMatMult::Laderman: {
        const auto mbuf_size = static_cast<std::size_t>(2.875 * total_size) + SafetyBufMargin;
        const auto uvbuf_size = static_cast<std::size_t>(0.25 * total_size) + SafetyBufMargin;

        auto mbuf = utils::MulIntermediateBuffer<T>(mbuf_size);
        auto uvbuf = utils::MulIntermediateBuffer<T>(uvbuf_size);
        
        laderman::laderman_matrix_mul_accum(a_view, b_view, dst_view, mbuf, uvbuf);
        break;
    }
    default: 
        assert(false);
    }

    return dst;
}

template Matrix2x<mfloat> matrix_mul<mfloat>(const Matrix2x<mfloat>&, const Matrix2x<mfloat>&, enumMatMult alg);
template Matrix2x<mdouble> matrix_mul<mdouble>(const Matrix2x<mdouble>&, const Matrix2x<mdouble>&, enumMatMult alg);
