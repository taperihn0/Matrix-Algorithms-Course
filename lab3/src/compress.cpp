#include "compress.hpp"

namespace mutils::compress
{

template <typename T>
void compress_impl(const Matrix2xView<T>& m,
                   QuadNode<T>& node,
                   T delta,
                   std::size_t max_rank)
{
  node.leaf = false;
  node.view = m;

  // compute svd
  // stop on conditions

  const auto n_l = n / 2;
  const auto n_r = n - n_l;

  auto m11 = m.submatrix_view(0, 0, n_l, n_l);
  auto m12 = m.submatrix_view(0, n_l, n_l, n_r);
  auto m21 = m.submatrix_view(n_l, 0, n_r, n_l);
  auto m22 = m.submatrix_view(n_l, n_l, n_r, n_r);

  compress_impl(m11, node.children[0], delta, max_rank);
  compress_impl(m12, node.children[1], delta, max_rank);
  compress_impl(m21, node.children[2], delta, max_rank);
  compress_impl(m22, node.children[3], delta, max_rank);
}

template <typename T>
CompressTree<T> compress(const Matrix2x<T>& m,
                         T delta,                 
                         std::size_t max_rank)
{
  auto root = QuadNode<T>{};

  compress_impl(m.submatrix_view(0, 0, m.rows(), m.cols()), 
                root, 
                delta, 
                max_rank);

  return CompressTree<T>{
    .compressed = m,
    .root = std::move(root)
  };
}

} // namespace mutils::compress