#include "matrix.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace utils {

template <typename T> class MulIntermediateBuffer {
public:
  MulIntermediateBuffer() = delete;
  explicit MulIntermediateBuffer(std::size_t buf_size)
      : m_size(buf_size), m_data(::new T[buf_size]) {}

  Matrix2xView<T> query_matrix_view(std::size_t view_rows,
                                    std::size_t view_cols) {
    const std::size_t req = view_rows * view_cols;

#ifdef DEBUG
    check_pointer_overflow(req);
#endif

    const auto mv = Matrix2xView<T>(m_data.get() + m_pointer, view_cols, 0, 0,
                                    view_rows, view_cols);

    m_pointer += req;
    return mv;
  }

  std::size_t get_mark_pointer() const { return m_pointer; }

  void revert_to_mark(std::size_t mark) {
    assert(mark <= m_pointer);
    m_pointer = mark;
  }

private:
#ifdef DEBUG
  void check_pointer_overflow(std::size_t req) {
    if (m_pointer + req > m_size) {
      throw std::runtime_error("MulIntermediateBuffer: Pointer overflow");
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
void default_matrix_mul_accum(const Matrix2xView<T> &a,
                              const Matrix2xView<T> &b, Matrix2xView<T> &acc) {
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
void strassen_matrix_mul_accum(const Matrix2xView<T> &a,
                               const Matrix2xView<T> &b, Matrix2xView<T> &acc,
                               utils::MulIntermediateBuffer<T> &pbuf,
                               utils::MulIntermediateBuffer<T> &uvbuf) {
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

  // Mark current stack pointer, then return to that mark
  const auto p_mark = pbuf.get_mark_pointer();
  const auto uv_mark = uvbuf.get_mark_pointer();

  for (auto &pi : p) {
    pi = pbuf.query_matrix_view(n_div, n_div);
    pi.fill(static_cast<T>(0));
  }

  // Create intermediate buffers for operation results
  auto u = uvbuf.query_matrix_view(n_div, n_div);
  auto v = uvbuf.query_matrix_view(n_div, n_div);

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
    pbuf.revert_to_mark(p_mark);
    uvbuf.revert_to_mark(uv_mark);
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

  pbuf.revert_to_mark(p_mark);
  uvbuf.revert_to_mark(uv_mark);
}

} // namespace strassen

namespace binet {

template <typename T>
void binet_matrix_mul_accum(const Matrix2xView<T> &a, const Matrix2xView<T> &b,
                            Matrix2xView<T> &acc) {
  assert(a.cols() == b.rows());
  // Note: sub-blocks may be rectangular when n is odd, so only the
  // dimension-compatibility asserts apply here.
  assert(a.rows() == acc.rows() && b.cols() == acc.cols());

  static constexpr auto BinetBaseBlockSize = 4uz;

  const auto n = a.rows();

  if (n <= BinetBaseBlockSize) {
    default_matrix_mul_accum(a, b, acc);
    return;
  }

  const auto n_rows = a.rows();
  const auto n_inner = a.cols();
  const auto n_cols = b.cols();

  // Independent row / inner / column splits: sub-blocks are rectangular
  // for odd dimensions, and each recursion level may split them differently.
  const auto rows_l = n_rows / 2;
  const auto rows_r = n_rows - rows_l;
  const auto inner_l = n_inner / 2;
  const auto inner_r = n_inner - inner_l;
  const auto cols_l = n_cols / 2;
  const auto cols_r = n_cols - cols_l;

  // Submatrices of C
  auto c11 = Matrix2xView<T>(acc, 0, 0, rows_l, cols_l);
  auto c12 = Matrix2xView<T>(acc, 0, cols_l, rows_l, cols_r);
  auto c21 = Matrix2xView<T>(acc, rows_l, 0, rows_r, cols_l);
  auto c22 = Matrix2xView<T>(acc, rows_l, cols_l, rows_r, cols_r);

  // Submatrices of A
  const auto a11 = Matrix2xView<T>(a, 0, 0, rows_l, inner_l);
  const auto a12 = Matrix2xView<T>(a, 0, inner_l, rows_l, inner_r);
  const auto a21 = Matrix2xView<T>(a, rows_l, 0, rows_r, inner_l);
  const auto a22 = Matrix2xView<T>(a, rows_l, inner_l, rows_r, inner_r);

  // Submatrices of B
  const auto b11 = Matrix2xView<T>(b, 0, 0, inner_l, cols_l);
  const auto b12 = Matrix2xView<T>(b, 0, cols_l, inner_l, cols_r);
  const auto b21 = Matrix2xView<T>(b, inner_l, 0, inner_r, cols_l);
  const auto b22 = Matrix2xView<T>(b, inner_l, cols_l, inner_r, cols_r);

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
void laderman_matrix_mul_accum(const Matrix2xView<T> &a,
                               const Matrix2xView<T> &b, Matrix2xView<T> &acc,
                               utils::MulIntermediateBuffer<T> &mbuf,
                               utils::MulIntermediateBuffer<T> &uvbuf) {
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

  // Mark current stack pointer, then return to that mark
  const auto m_mark = mbuf.get_mark_pointer();
  const auto uv_mark = uvbuf.get_mark_pointer();

  for (auto &mi : m) {
    mi = mbuf.query_matrix_view(n_div, n_div);
    mi.fill(static_cast<T>(0));
  }

  // Create intermediate buffers for operation results
  auto u = uvbuf.query_matrix_view(n_div, n_div);
  auto v = uvbuf.query_matrix_view(n_div, n_div);

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
  c12.matrix_add_accum_into(
      {&m[0], &m[3], &m[4], &m[5], &m[11], &m[13], &m[14]});
  // C13 = M6 + M7 + M9 + M11 + M14 + M16 + M18
  c13.matrix_add_accum_into(
      {&m[5], &m[6], &m[8], &m[9], &m[13], &m[15], &m[17]});
  // C21 = M2 + M3 + M4 + M6 + M14 + M16 + M17
  c21.matrix_add_accum_into(
      {&m[1], &m[2], &m[3], &m[5], &m[13], &m[15], &m[16]});
  // C22 = M2 + M4 + M5 + M6 + M20
  c22.matrix_add_accum_into({&m[1], &m[3], &m[4], &m[5], &m[19]});
  // C23 = M14 + M16 + M17 + M18 + M21
  c23.matrix_add_accum_into({&m[13], &m[15], &m[16], &m[17], &m[20]});
  // C31 = M6 + M7 + M8 + M11 + M12 + M13 + M14
  c31.matrix_add_accum_into(
      {&m[5], &m[6], &m[7], &m[10], &m[11], &m[12], &m[13]});
  // C32 = M12 + M13 + M14 + M15 + M22
  c32.matrix_add_accum_into({&m[11], &m[12], &m[13], &m[14], &m[21]});
  // C33 = M6 + M7 + M8 + M9 + M23
  c33.matrix_add_accum_into({&m[5], &m[6], &m[7], &m[8], &m[22]});

  if (n_is3div) { // happy case, we are done
    mbuf.revert_to_mark(m_mark);
    uvbuf.revert_to_mark(uv_mark);
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

  mbuf.revert_to_mark(m_mark);
  uvbuf.revert_to_mark(uv_mark);
}

} // namespace laderman

template <typename T>
Matrix2x<T> matrix_mul(const Matrix2x<T> &a, const Matrix2x<T> &b,
                       enumMatMult alg) {
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
    const auto pbuf_size =
        static_cast<std::size_t>((7. / 3.) * total_size) + SafetyBufMargin;
    const auto uvbuf_size =
        static_cast<std::size_t>((2. / 3.) * total_size) + SafetyBufMargin;

    auto pbuf = utils::MulIntermediateBuffer<T>(pbuf_size);
    auto uvbuf = utils::MulIntermediateBuffer<T>(uvbuf_size);

    strassen::strassen_matrix_mul_accum(a_view, b_view, dst_view, pbuf, uvbuf);
    break;
  }
  case enumMatMult::Laderman: {
    const auto mbuf_size =
        static_cast<std::size_t>(2.875 * total_size) + SafetyBufMargin;
    const auto uvbuf_size =
        static_cast<std::size_t>(0.25 * total_size) + SafetyBufMargin;

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

template Matrix2x<mfloat> matrix_mul<mfloat>(const Matrix2x<mfloat> &,
                                             const Matrix2x<mfloat> &,
                                             enumMatMult alg);
template Matrix2x<mdouble> matrix_mul<mdouble>(const Matrix2x<mdouble> &,
                                               const Matrix2x<mdouble> &,
                                               enumMatMult alg);

template Matrix2x<mfloat> inverse<mfloat>(Matrix2xView<mfloat> &, enumMatMult);
template Matrix2x<mdouble> inverse<mdouble>(Matrix2xView<mdouble> &,
                                            enumMatMult);

template Matrix2x<mfloat> inverse_laderman<mfloat>(Matrix2xView<mfloat> &);
template Matrix2x<mdouble> inverse_laderman<mdouble>(Matrix2xView<mdouble> &);

// Multiple views
template <typename T>
void multiple_views_binet(Matrix2xView<T> &a, Matrix2xView<T> &b,
                          Matrix2xView<T> &acc) {
  acc.fill(T{});
  binet::binet_matrix_mul_accum(a, b, acc);
};

template <typename T>
void multiple_views_strassen(Matrix2xView<T> &a, Matrix2xView<T> &b,
                             Matrix2xView<T> &acc) {
  acc.fill(T{});
  constexpr std::size_t margin = 32uz;

  if (a.rows() != a.cols() || b.rows() != b.cols()) {
    // Add padding
    const std::size_t p = a.rows();
    const std::size_t q = a.cols();
    const std::size_t r = b.cols();
    std::size_t m = std::max({p, q, r});

    Matrix2x<T> a_padded(m, m);
    Matrix2x<T> b_padded(m, m);

    for (std::size_t i = 0; i < p; ++i)
      for (std::size_t j = 0; j < q; ++j)
        a_padded(i, j) = a(i, j);

    for (std::size_t i = 0; i < q; ++i)
      for (std::size_t j = 0; j < r; ++j)
        b_padded(i, j) = b(i, j);

    auto ap = a_padded.submatrix_view(0, 0, m, m);
    auto bp = b_padded.submatrix_view(0, 0, m, m);

    Matrix2x<T> result_padded(m, m);
    auto rp = result_padded.submatrix_view(0, 0, m, m);
    multiple_views_strassen(ap, bp, rp);

    for (std::size_t i = 0; i < p; ++i)
      for (std::size_t j = 0; j < r; ++j)
        acc(i, j) = result_padded(i, j);
    return;
  }

  const std::size_t total_size = a.rows() * b.cols();

  auto pbuf = utils::MulIntermediateBuffer<T>(
      static_cast<std::size_t>((7. / 3.) * total_size) + margin);
  auto uvbuf = utils::MulIntermediateBuffer<T>(
      static_cast<std::size_t>((2. / 3.) * total_size) + margin);

  strassen::strassen_matrix_mul_accum(a, b, acc, pbuf, uvbuf);
}

template <typename T>
void multiple_views_laderman(Matrix2xView<T> &a, Matrix2xView<T> &b,
                             Matrix2xView<T> &acc) {
  acc.fill(T{});

  if (a.rows() != a.cols() || b.rows() != b.cols()) {
    const std::size_t p = a.rows();
    const std::size_t q = a.cols();
    const std::size_t r = b.cols();
    std::size_t m = std::max({p, q, r});

    Matrix2x<T> a_padded(m, m);
    Matrix2x<T> b_padded(m, m);

    for (std::size_t i = 0; i < p; ++i)
      for (std::size_t j = 0; j < q; ++j)
        a_padded(i, j) = a(i, j);

    for (std::size_t i = 0; i < q; ++i)
      for (std::size_t j = 0; j < r; ++j)
        b_padded(i, j) = b(i, j);

    auto ap = a_padded.submatrix_view(0, 0, m, m);
    auto bp = b_padded.submatrix_view(0, 0, m, m);

    Matrix2x<T> result_padded(m, m);
    auto rp = result_padded.submatrix_view(0, 0, m, m);
    multiple_views_laderman(ap, bp, rp);

    for (std::size_t i = 0; i < p; ++i)
      for (std::size_t j = 0; j < r; ++j)
        acc(i, j) = result_padded(i, j);
    return;
  }

  constexpr std::size_t margin = 32uz;
  const auto total_size = acc.rows() * acc.cols();

  auto mbuf = utils::MulIntermediateBuffer<T>(
      static_cast<std::size_t>(2.875 * total_size) + margin);

  auto uvbuf = utils::MulIntermediateBuffer<T>(
      static_cast<std::size_t>(0.25 * total_size) + margin);

  laderman::laderman_matrix_mul_accum(a, b, acc, mbuf, uvbuf);
}

template <typename T>
void multiple_views(Matrix2xView<T> &a, Matrix2xView<T> &b,
                    Matrix2xView<T> &acc, enumMatMult type) {
  if (type == enumMatMult::Binet)
    multiple_views_binet(a, b, acc);
  else if (type == enumMatMult::Strassen)
    multiple_views_strassen(a, b, acc);
  else
    multiple_views_laderman(a, b, acc);
};

template <typename T>
Matrix2x<T> mul_views(Matrix2xView<T> &a, Matrix2xView<T> &b) {
  Matrix2x<T> result(a.rows(), b.cols());

  auto result_view = result.submatrix_view(0, 0, a.rows(), b.cols());

  multiple_views(a, b, result_view);

  return result;
}

template <typename T> Matrix2x<T> inverse_laderman(Matrix2xView<T> &a) {
  const std::size_t n = a.rows();

  if (n != a.cols() || n == 0) {
    throw std::invalid_argument("inverse_laderman: invalid matrix size");
  }

  if (n == 1) {
    if (a(0, 0) == T{}) {
      throw std::invalid_argument("inverse_laderman: singular matrix");
    }
    Matrix2x<T> result(1, 1);
    result(0, 0) = T{1} / a(0, 0);
    return result;
  }

  if (n == 2) {
    T det = a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0);
    if (det == T{}) {
      throw std::invalid_argument("inverse_laderman: singular matrix");
    }
    Matrix2x<T> result(2, 2);
    result(0, 0) = a(1, 1) / det;
    result(0, 1) = -a(0, 1) / det;
    result(1, 0) = -a(1, 0) / det;
    result(1, 1) = a(0, 0) / det;
    return result;
  }

  const std::size_t n1 = n / 3;
  const std::size_t n2 = n / 3;
  const std::size_t n3 = n - n1 - n2;

  const std::size_t r0 = 0, r1 = n1, r2 = n1 + n2;
  const std::size_t c0 = 0, c1 = n1, c2 = n1 + n2;

  auto a11 = a.submatrix_view(r0, c0, n1, n1);
  auto a12 = a.submatrix_view(r0, c1, n1, n2);
  auto a13 = a.submatrix_view(r0, c2, n1, n3);
  auto a21 = a.submatrix_view(r1, c0, n2, n1);
  auto a22 = a.submatrix_view(r1, c1, n2, n2);
  auto a23 = a.submatrix_view(r1, c2, n2, n3);
  auto a31 = a.submatrix_view(r2, c0, n3, n1);
  auto a32 = a.submatrix_view(r2, c1, n3, n2);
  auto a33 = a.submatrix_view(r2, c2, n3, n3);

  auto mul = [](Matrix2xView<T> &x, Matrix2xView<T> &y) -> Matrix2x<T> {
    Matrix2x<T> res(x.rows(), y.cols());
    auto res_view = res.submatrix_view(0, 0, x.rows(), y.cols());
    multiple_views_laderman(x, y, res_view);
    return res;
  };

  // X1 = inv(A11)
  auto X1 = inverse_laderman(a11);
  auto X1_view = X1.submatrix_view(0, 0, n1, n1);

  auto L21 = mul(a21, X1_view);
  auto L21_view = L21.submatrix_view(0, 0, n2, n1);

  auto L31 = mul(a31, X1_view);
  auto L31_view = L31.submatrix_view(0, 0, n3, n1);

  auto U12 = mul(X1_view, a12);
  auto U12_view = U12.submatrix_view(0, 0, n1, n2);

  auto U13 = mul(X1_view, a13);
  auto U13_view = U13.submatrix_view(0, 0, n1, n3);

  // Schur 1
  auto L21_A12 = mul(L21_view, a12);
  Matrix2x<T> S22(n2, n2);
  for (std::size_t i = 0; i < n2; ++i)
    for (std::size_t j = 0; j < n2; ++j)
      S22(i, j) = a22(i, j) - L21_A12(i, j);

  auto L21_A13 = mul(L21_view, a13);
  Matrix2x<T> S23(n2, n3);
  for (std::size_t i = 0; i < n2; ++i)
    for (std::size_t j = 0; j < n3; ++j)
      S23(i, j) = a23(i, j) - L21_A13(i, j);

  auto L31_A12 = mul(L31_view, a12);
  Matrix2x<T> S32(n3, n2);
  for (std::size_t i = 0; i < n3; ++i)
    for (std::size_t j = 0; j < n2; ++j)
      S32(i, j) = a32(i, j) - L31_A12(i, j);

  auto L31_A13 = mul(L31_view, a13);
  Matrix2x<T> S33(n3, n3);
  for (std::size_t i = 0; i < n3; ++i)
    for (std::size_t j = 0; j < n3; ++j)
      S33(i, j) = a33(i, j) - L31_A13(i, j);

  // INV(S22)
  auto S22_view = S22.submatrix_view(0, 0, n2, n2);
  auto X2 = inverse_laderman(S22_view);
  auto X2_view = X2.submatrix_view(0, 0, n2, n2);

  auto S32_view = S32.submatrix_view(0, 0, n3, n2);
  auto S23_view = S23.submatrix_view(0, 0, n2, n3);

  auto L32 = mul(S32_view, X2_view);
  auto L32_view = L32.submatrix_view(0, 0, n3, n2);

  auto U23 = mul(X2_view, S23_view);
  auto U23_view = U23.submatrix_view(0, 0, n2, n3);

  // Schur 2
  auto L32_S23 = mul(L32_view, S23_view);
  Matrix2x<T> S33_prime(n3, n3);
  for (std::size_t i = 0; i < n3; ++i)
    for (std::size_t j = 0; j < n3; ++j)
      S33_prime(i, j) = S33(i, j) - L32_S23(i, j);

  // inv(S33)
  auto S33_prime_view = S33_prime.submatrix_view(0, 0, n3, n3);
  auto X3 = inverse_laderman(S33_prime_view);
  auto X3_view = X3.submatrix_view(0, 0, n3, n3);

  // tilde
  Matrix2x<T> L_tilde_21(n2, n1);
  for (std::size_t i = 0; i < n2; ++i)
    for (std::size_t j = 0; j < n1; ++j)
      L_tilde_21(i, j) = -L21(i, j);

  Matrix2x<T> L_tilde_32(n3, n2);
  for (std::size_t i = 0; i < n3; ++i)
    for (std::size_t j = 0; j < n2; ++j)
      L_tilde_32(i, j) = -L32(i, j);

  auto L32_L21 = mul(L32_view, L21_view);
  Matrix2x<T> L_tilde_31(n3, n1);
  for (std::size_t i = 0; i < n3; ++i)
    for (std::size_t j = 0; j < n1; ++j)
      L_tilde_31(i, j) = L32_L21(i, j) - L31(i, j);

  Matrix2x<T> U_tilde_12(n1, n2);
  for (std::size_t i = 0; i < n1; ++i)
    for (std::size_t j = 0; j < n2; ++j)
      U_tilde_12(i, j) = -U12(i, j);

  Matrix2x<T> U_tilde_23(n2, n3);
  for (std::size_t i = 0; i < n2; ++i)
    for (std::size_t j = 0; j < n3; ++j)
      U_tilde_23(i, j) = -U23(i, j);

  auto U12_U23 = mul(U12_view, U23_view);
  Matrix2x<T> U_tilde_13(n1, n3);
  for (std::size_t i = 0; i < n1; ++i)
    for (std::size_t j = 0; j < n3; ++j)
      U_tilde_13(i, j) = U12_U23(i, j) - U13(i, j);

  // W = U_tilde * D^(-1)
  auto U_tilde_12_view = U_tilde_12.submatrix_view(0, 0, n1, n2);
  auto W12 = mul(U_tilde_12_view, X2_view);
  auto W12_view = W12.submatrix_view(0, 0, n1, n2);

  auto U_tilde_13_view = U_tilde_13.submatrix_view(0, 0, n1, n3);
  auto W13 = mul(U_tilde_13_view, X3_view);
  auto W13_view = W13.submatrix_view(0, 0, n1, n3);

  auto U_tilde_23_view = U_tilde_23.submatrix_view(0, 0, n2, n3);
  auto W23 = mul(U_tilde_23_view, X3_view);
  auto W23_view = W23.submatrix_view(0, 0, n2, n3);

  // B = W * L_tilde
  auto L_tilde_21_view = L_tilde_21.submatrix_view(0, 0, n2, n1);
  auto L_tilde_32_view = L_tilde_32.submatrix_view(0, 0, n3, n2);
  auto L_tilde_31_view = L_tilde_31.submatrix_view(0, 0, n3, n1);

  auto B32 = mul(X3_view, L_tilde_32_view);
  auto B31 = mul(X3_view, L_tilde_31_view);

  auto W23_L32 = mul(W23_view, L_tilde_32_view);
  Matrix2x<T> B22(n2, n2);
  for (std::size_t i = 0; i < n2; ++i)
    for (std::size_t j = 0; j < n2; ++j)
      B22(i, j) = X2(i, j) + W23_L32(i, j);

  auto X2_L21 = mul(X2_view, L_tilde_21_view);
  auto W23_L31 = mul(W23_view, L_tilde_31_view);
  Matrix2x<T> B21(n2, n1);
  for (std::size_t i = 0; i < n2; ++i)
    for (std::size_t j = 0; j < n1; ++j)
      B21(i, j) = X2_L21(i, j) + W23_L31(i, j);

  auto W13_L32 = mul(W13_view, L_tilde_32_view);
  Matrix2x<T> B12(n1, n2);
  for (std::size_t i = 0; i < n1; ++i)
    for (std::size_t j = 0; j < n2; ++j)
      B12(i, j) = W12(i, j) + W13_L32(i, j);

  auto W12_L21 = mul(W12_view, L_tilde_21_view);
  auto W13_L31 = mul(W13_view, L_tilde_31_view);
  Matrix2x<T> B11(n1, n1);
  for (std::size_t i = 0; i < n1; ++i)
    for (std::size_t j = 0; j < n1; ++j)
      B11(i, j) = X1(i, j) + W12_L21(i, j) + W13_L31(i, j);

  Matrix2x<T> result(n, n);
  auto put_block = [&result](std::size_t row, std::size_t col,
                             const Matrix2x<T> &block) {
    for (std::size_t i = 0; i < block.rows(); ++i) {
      for (std::size_t j = 0; j < block.cols(); ++j) {
        result(row + i, col + j) = block(i, j);
      }
    }
  };

  put_block(r0, c0, B11);
  put_block(r0, c1, B12);
  put_block(r0, c2, W13);

  put_block(r1, c0, B21);
  put_block(r1, c1, B22);
  put_block(r1, c2, W23);

  put_block(r2, c0, B31);
  put_block(r2, c1, B32);
  put_block(r2, c2, X3);

  return result;
}

template <typename T>
Matrix2x<T> inverse(Matrix2xView<T> &a, enumMatMult type) {
  const std::size_t n = a.rows();

  if (n != a.cols() || n == 0) {
    throw std::invalid_argument("inverse: invalid matrix size");
  }

  if (type == enumMatMult::Laderman) {
    return inverse_laderman(a);
  }

  if (n == 1) {
    if (a(0, 0) == T{}) {
      throw std::invalid_argument("inverse: singular matrix");
    }

    Matrix2x<T> result(1, 1);
    result(0, 0) = T{1} / a(0, 0);
    return result;
  }

  const std::size_t n1 = n / 2;
  const std::size_t n2 = n - n1;

  // Buffer block
  auto buff1 = Matrix2x<T>(n, n);
  auto buff2 = Matrix2x<T>(n, n);
  //

  auto a11 = a.submatrix_view(0, 0, n1, n1);
  auto a12 = a.submatrix_view(0, n1, n1, n2);
  auto a21 = a.submatrix_view(n1, 0, n2, n1);
  auto a22 = a.submatrix_view(n1, n1, n2, n2);

  // inv(a11)
  auto inv_a11 = inverse(a11, type);
  auto inv_a11_view = inv_a11.submatrix_view(0, 0, n1, n1);

  // S22 = A22 - A21 * inv(A11) * A12.
  auto buff1_view = buff1.submatrix_view(0, 0, n2, n1);
  multiple_views(a21, inv_a11_view, buff1_view, type);

  auto buff2_view = buff2.submatrix_view(0, 0, n2, n2);
  multiple_views(buff1_view, a12, buff2_view, type);

  Matrix2x<T> S22(n2, n2);

  for (std::size_t i = 0; i < n2; ++i) {
    for (std::size_t j = 0; j < n2; ++j) {
      S22(i, j) = a22(i, j) - buff2(i, j);
    }
  }

  // inv(S22)
  auto S22_view = S22.submatrix_view(0, 0, n2, n2);
  auto inv_S22 = inverse(S22_view, type);
  auto inv_S22_view = inv_S22.submatrix_view(0, 0, n2, n2);

  // B12 = -inv(A11) * A12 * inv(S22)
  auto buff1_b12_view = buff1.submatrix_view(0, 0, n1, n2);
  multiple_views(inv_a11_view, a12, buff1_b12_view, type);

  auto buff2_b12_view = buff2.submatrix_view(0, 0, n1, n2);
  multiple_views(buff1_b12_view, inv_S22_view, buff2_b12_view, type);

  Matrix2x<T> b12(n1, n2);

  for (std::size_t i = 0; i < n1; ++i) {
    for (std::size_t j = 0; j < n2; ++j) {
      b12(i, j) = -buff2(i, j);
    }
  }

  // B11 = inv(A11) + inv(A11) * A12 * inv(S22) * A21 * inv(A11)
  auto buff1_b11_view = buff1.submatrix_view(0, 0, n1, n1);
  multiple_views(buff2_b12_view, a21, buff1_b11_view, type);

  auto buff2_b11_view = buff2.submatrix_view(0, 0, n1, n1);
  multiple_views(buff1_b11_view, inv_a11_view, buff2_b11_view, type);

  Matrix2x<T> b11(n1, n1);

  for (std::size_t i = 0; i < n1; ++i) {
    for (std::size_t j = 0; j < n1; ++j) {
      b11(i, j) = inv_a11(i, j) + buff2(i, j);
    }
  }

  // B21 = -inv(S22) * A21 * inv(A11)
  auto buff1_b21_view = buff1.submatrix_view(0, 0, n2, n1);
  multiple_views(inv_S22_view, a21, buff1_b21_view, type);

  auto buff2_b21_view = buff2.submatrix_view(0, 0, n2, n1);
  multiple_views(buff1_b21_view, inv_a11_view, buff2_b21_view, type);

  Matrix2x<T> b21(n2, n1);

  for (std::size_t i = 0; i < n2; ++i) {
    for (std::size_t j = 0; j < n1; ++j) {
      b21(i, j) = -buff2(i, j);
    }
  }

  // B22 = inv(S22)
  auto b22 = inv_S22;

  // Joining b blocks
  Matrix2x<T> result(n, n);

  auto put_block = [&result](std::size_t row, std::size_t col,
                             const Matrix2x<T> &block) {
    for (std::size_t i = 0; i < block.rows(); ++i) {
      for (std::size_t j = 0; j < block.cols(); ++j) {
        result(row + i, col + j) = block(i, j);
      }
    }
  };

  put_block(0, 0, b11);
  put_block(0, n1, b12);
  put_block(n1, 0, b21);
  put_block(n1, n1, b22);

  return result;
}

template std::pair<Matrix2x<mfloat>, Matrix2x<mfloat>>
LU<mfloat>(Matrix2xView<mfloat> &, enumMatMult);

template std::pair<Matrix2x<mdouble>, Matrix2x<mdouble>>
LU<mdouble>(Matrix2xView<mdouble> &, enumMatMult);

template <typename T>
std::pair<Matrix2x<T>, Matrix2x<T>> LU(Matrix2xView<T> &a, enumMatMult type) {
  const std::size_t n = a.rows();

  // Case n = 1
  if (n == 1) {
    Matrix2x<T> L(1, 1);
    Matrix2x<T> U(1, 1);

    L(0, 0) = T{1};
    U(0, 0) = a(0, 0);

    return {std::move(L), std::move(U)};
  }

  const std::size_t n1 = n / 2;
  const std::size_t n2 = n - n1;

  // Split to blocks
  auto a11 = a.submatrix_view(0, 0, n1, n1);
  auto a12 = a.submatrix_view(0, n1, n1, n2);
  auto a21 = a.submatrix_view(n1, 0, n2, n1);
  auto a22 = a.submatrix_view(n1, n1, n2, n2);

  // [L11, U11] = LU(A11)
  auto [L11, U11] = LU(a11, type);

  // inv(U11)
  auto U11_view = U11.submatrix_view(0, 0, n1, n1);
  auto inv_U11 = inverse(U11_view, type);
  auto inv_U11_view = inv_U11.submatrix_view(0, 0, n1, n1);

  // L21 = A21 * inv(U11)
  Matrix2x<T> L21(n2, n1);
  auto L21_view = L21.submatrix_view(0, 0, n2, n1);

  multiple_views(a21, inv_U11_view, L21_view, type);

  // inv(L11)
  auto L11_view = L11.submatrix_view(0, 0, n1, n1);
  auto inv_L11 = inverse(L11_view, type);
  auto inv_L11_view = inv_L11.submatrix_view(0, 0, n1, n1);

  // U12 = inv(L11) * A12
  Matrix2x<T> U12(n1, n2);
  auto U12_view = U12.submatrix_view(0, 0, n1, n2);

  multiple_views(inv_L11_view, a12, U12_view, type);

  // S = A22 - L21 * U12
  Matrix2x<T> product(n2, n2);
  auto product_view = product.submatrix_view(0, 0, n2, n2);

  multiple_views(L21_view, U12_view, product_view, type);

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

  L_view.fill(T{});
  U_view.fill(T{});
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