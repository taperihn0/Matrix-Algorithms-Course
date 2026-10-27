#include "lab1/src/matrix.hpp"
#include "lab1/src/intermediate_buffer.hpp"
#include "lab1/src/matrix_utils.hpp"
#include <cassert>

template <typename T, enumMatMult Alg>
void inverse_impl(Matrix2xView<T>& a,
                  Matrix2xView<T>& acc,
                  IntermediateBuffer<T>& buf) {
  const auto n = acc.rows();

  if (n == 1) {
    acc(0, 0) = static_cast<T>(1) / a(0, 0);
    return;
  }

  const auto n_l = n / 2;
  const auto n_r = n - n_l;

  const auto buf_mark = buf.mark();

  auto ainv = buf.query_view(n_l, n_l);
  auto s = buf.query_view(n_r, n_r);

  auto u = buf.query_view(n_r, n_r);
  auto v = buf.query_view(n_r, n_r);
  auto w = buf.query_view(n_r, n_r);

  auto a11 = Matrix2xView<T>(a, 0, 0, n_l, n_l);
  auto a12 = Matrix2xView<T>(a, 0, n_l, n_l, n_r);
  auto a21 = Matrix2xView<T>(a, n_l, 0, n_r, n_l);
  auto a22 = Matrix2xView<T>(a, n_l, n_l, n_r, n_r);

  auto b11 = Matrix2xView<T>(acc, 0, 0, n_l, n_l);
  auto b12 = Matrix2xView<T>(acc, 0, n_l, n_l, n_r);
  auto b21 = Matrix2xView<T>(acc, n_l, 0, n_r, n_l);
  auto b22 = Matrix2xView<T>(acc, n_l, n_l, n_r, n_r);

  auto u_nl_nr = Matrix2xView<T>(u, 0, 0, n_l, n_r);
  auto v_nl_nl = Matrix2xView<T>(v, 0, 0, n_l, n_l);
  auto w_nl_nl = Matrix2xView<T>(w, 0, 0, n_l, n_l);
  auto v_nl_nr = Matrix2xView<T>(v, 0, 0, n_l, n_r);
  auto u_nr_nl = Matrix2xView<T>(u, 0, 0, n_r, n_l);
  auto v_nr_nl = Matrix2xView<T>(v, 0, 0, n_r, n_l);

  static constexpr matrix_mul = [](auto&&... args) {
    // ...
  };

  // A_INV = A11^-1
  inverse_impl<T, Alg>(a11, ainv, buf);

  // S = (A22 - A21 * A11^-1 * A12)^-1
  mutils::matrix_mul_alg<T, Alg>(a21, ainv, u_nr_nl);
  mutils::matrix_mul_alg<T, Alg>(u_nr_nl, a12, s);

  s.matrix_addsub_accum_into({&a22}, {&s});
  inverse_impl<T, Alg>(s, s, buf);

  // B11 = A11^-1 * (I + A12 * S^-1 * A21 * A11^-1)
  mutils::matrix_mul_alg<T, Alg>(a12, s, u_nl_nr);
  mutils::matrix_mul_alg<T, Alg>(u_nl_nr, a21, v_nl_nl);
  mutils::matrix_mul_alg<T, Alg>(v_nl_nl, ainv, w_nl_nl);

  for (std::size_t i = 0; i < n_l; ++i) {
    w_nl_nl(i, i) += static_cast<T>(1);
  }

  mutils::matrix_mul_alg<T, Alg>(ainv, w_nl_nl, b11);

  // B12 = -A11^-1 * A12 * S^-1
  mutils::matrix_mul_alg<T, Alg>(ainv, a12, v_nl_nr);
  mutils::matrix_mul_alg<T, Alg>(v_nl_nr, s, u_nl_nr);

  b12.looped_elem_assign([&](std::size_t i, std::size_t j) {
    return -u_nl_nr(i, j);
  });

  // B21 = -S^-1 * A21 * A11^-1
  mutils::matrix_mul_alg<T, Alg>(s, a21, v_nr_nl);
  mutils::matrix_mul_alg<T, Alg>(v_nr_nl, ainv, u_nr_nl);

  b21.looped_elem_assign([&](std::size_t i, std::size_t j) {
    return -u_nr_nl(i, j);
  });

  // B22 = S^-1
  a22 = s;

  buf.revert_to(buf_mark);
}

template <typename T>
Matrix2x<T> &Matrix2x<T>::inverse(Matrix2x<T> &acc, enumMatMult alg) {
  assert(rows() == cols());
  assert(acc.rows() == rows() && acc.cols() == cols());

  Matrix2xView<T> a_view(data(), cols(), 0, 0, rows(), cols());
  Matrix2xView<T> acc_view(acc.data(), acc.cols(), 0, 0, acc.rows(), acc.cols());

  IntermediateBuffer<T> buf(rows() * cols());

  switch (alg) {
  case enumMatMult::Default:
    inverse_impl<T, enumMatMult::Default>(a_view, acc_view, buf);
    break;
  case enumMatMult::Binet:
    inverse_impl<T, enumMatMult::Binet>(a_view, acc_view, buf);
    break;
  case enumMatMult::Strassen:
    inverse_impl<T, enumMatMult::Strassen>(a_view, acc_view, buf);
    break;
  case enumMatMult::Laderman:
    inverse_impl<T, enumMatMult::Laderman>(a_view, acc_view, buf);
    break;
  default:
    assert(false);
    break;
  }

  return acc;
}

template Matrix2x<mfloat> &Matrix2x<mfloat>::inverse(Matrix2x<mfloat> &, enumMatMult);
template Matrix2x<mdouble> &Matrix2x<mdouble>::inverse(Matrix2x<mdouble> &, enumMatMult);