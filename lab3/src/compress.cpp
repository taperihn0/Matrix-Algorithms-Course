#include "compress.hpp"
#include "vector.hpp"

#include <cstddef>
#include <cstdint>

namespace mutils::compress
{

template <typename T>
Vector<T> mgs(Vector<T> v, const std::vector<Vector<T>> &vs)
{
  std::ranges::for_each(vs, [&v](const auto& vprev) {
    v -= vprev * v.dot(vprev);
  });
  return v;
}

template <typename T>
SvdDecompMatrices<T> svd(const Matrix2xView<T> &m, std::size_t b, T delta)
{
  const auto cols = m.cols();
  const auto rows = m.rows();

  auto res = SvdDecompMatrices<T>();

  if (rows == 0 || cols == 0 || b == 0 || delta < 0) {
    return res;
  }

  auto r = Matrix2x<T>(m);

  std::vector<Vector<T>> vs;
  std::vector<Vector<T>> us;
  std::vector<T> sigmas;

  vs.reserve(b);
  us.reserve(b);
  sigmas.reserve(b);

  static constexpr auto Epsilon = 1e-8;
  static constexpr auto IterationLimit = 512;

  while (sigmas.size() < b) {
    auto v = mgs(Vector<T>::random(cols), vs);
    const auto r_trans = r.transpose();

    for (auto iter = 0uz; iter < IterationLimit; ++iter) {
      auto w = r * v;
      auto z = mgs(r_trans * w, vs);

      const auto z_norm = z.norm();

      if (z_norm <= Epsilon) {
        v = Vector<T>::zeros(cols);
        break;
      }

      const auto v_next = z / z_norm;
      const auto diff = (v_next - v).norm();

      v = v_next;

      if (diff <= Epsilon) {
        break;
      }
    }

    const auto w = r * v;
    const auto sigma = w.norm();

    if (sigma <= delta) {
      break;
    }

    us.push_back(w / sigma);
    vs.push_back(v);
    sigmas.push_back(sigma);

    r = r - us.back().outer(vs.back()) * sigma;
  }

  const auto rank = sigmas.size();

  res.b = rank;
  res.u = Matrix2x<T>(rows, rank);
  res.v_trans = Matrix2x<T>(rank, cols);
  res.sigma = std::move(sigmas);

  for (auto k = 0uz; k < rank; ++k) {
    for (auto i = 0uz; i < rows; ++i) {
      res.u(i, k) = us[k][i];
    }

    for (auto j = 0uz; j < cols; ++j) {
      res.v_trans(k, j) = vs[k][j];
    }
  }

  return res;
}

template <typename T>
SvdDecompMatrices<T> svd(const Matrix2x<T> &m, std::size_t b, T delta)
{
  return svd(Matrix2xView<T>(m.data(), m.cols(), 0, 0, m.rows(), m.cols()),
             b, delta);
}

template <typename T>
void compress_impl(const Matrix2xView<T>& m,
                   QuadNode<T>& node,
                   T delta,
                   std::size_t rank_limit)
{
  const auto rows = m.rows();
  const auto cols = m.cols();

  node.is_leaf = false;
  node.view = m;

  // compute svd and stop on conditions
  const auto factorize = svd(m, rank_limit, delta);

  if (factorize.b > rank_limit || factorize.sigma.back() < delta) {
    node = QuadNode<T>::leaf(factorize);
    return;
  }

  const auto rows_d2 = rows / 2;
  const auto cols_d2 = cols / 2;

  auto m11 = m.submatrix_view(0, 0, rows_d2, cols_d2);
  auto m12 = m.submatrix_view(0, cols_d2 + 1, rows_d2, cols_d2);
  auto m21 = m.submatrix_view(rows_d2 + 1, 0, rows_d2, cols_d2);
  auto m22 = m.submatrix_view(rows_d2 + 1, cols_d2 + 1, rows_d2, cols_d2);

  compress_impl(m11, node.children[0], delta, rank_limit);
  compress_impl(m12, node.children[1], delta, rank_limit);
  compress_impl(m21, node.children[2], delta, rank_limit);
  compress_impl(m22, node.children[3], delta, rank_limit);
}

template <typename T>
CompressTree<T> compress(const Matrix2x<T>& m,
                         T delta,                 
                         std::size_t rank_limit)
{
  auto root = QuadNode<T>{};

  compress_impl(m.submatrix_view(0, 0, m.rows(), m.cols()), 
                root, 
                delta, 
                rank_limit);

  return CompressTree<T>{
    .compressed = m,
    .root = std::move(root)
  };
}

template SvdDecompMatrices<float> svd(const Matrix2xView<float> &,
                                      std::size_t, float);
template SvdDecompMatrices<double> svd(const Matrix2xView<double> &,
                                       std::size_t, double);
template SvdDecompMatrices<mfloat> svd(const Matrix2xView<mfloat> &,
                                       std::size_t, mfloat);
template SvdDecompMatrices<mdouble> svd(const Matrix2xView<mdouble> &,
                                        std::size_t, mdouble);
template SvdDecompMatrices<float> svd(const Matrix2x<float> &,
                                     std::size_t, float);
template SvdDecompMatrices<double> svd(const Matrix2x<double> &,
                                       std::size_t, double);
template SvdDecompMatrices<mfloat> svd(const Matrix2x<mfloat> &,
                                       std::size_t, mfloat);
template SvdDecompMatrices<mdouble> svd(const Matrix2x<mdouble> &,
                                        std::size_t, mdouble);

} // namespace mutils::compress
