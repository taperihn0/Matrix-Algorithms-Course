#pragma once

#include "lab1/src/matrix_utils.hpp"

namespace mutils::compress
{

// stores result of SVD decompossition.
template <typename T>
struct SvdDecompResults
{
  Matrix2x<T> u;
  Matrix2x<T> d;
  Matrix2x<T> v_trans;
};

template <typename T>
struct QuadNode
{
  static inline QuadNode leaf(const SvdDecompResults& svd_decomp) {
    return QuadNode {
      .leaf = true,
      .svd_decomp = svd_decomp,
    };
  }

  // that field is used only in leaf nodes.
  std::optional<SvdDecompResults> svd_decomp;
  const Matrix2xView<T> view;
  std::array<QuadNode, 4> children;
  bool leaf = false;
};

template <typename T>
struct CompressTree
{
  const std::reference_wrapper<Matrix2x<T>> compressed;
  QuadNode root;
};    

template <typename T>
static CompressTree<T> compress(const Matrix2x<T>& m,
                                T delta,                 // minimum singular value
                                std::size_t max_rank);   // maximum rank

}; // namespace mutils::compress
