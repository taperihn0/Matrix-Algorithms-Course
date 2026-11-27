#pragma once

#include "lab1/src/matrix_utils.hpp"

#include <array>
#include <optional>
#include <vector>

namespace mutils::compress
{

// stores result of SVD decompossition.
template <typename T>
struct SvdDecompMatrices
{
  Matrix2x<T> u;
  std::vector<T> sigma;
  Matrix2x<T> v_trans;
  std::size_t b;
};

template <typename T>
struct QuadNode
{
  static inline QuadNode leaf(const SvdDecompMatrices<T>& svd_decomp) {
    return QuadNode {
      .svd_decomp = svd_decomp,
      .leaf = true,
    };
  }

  // that field is used only in leaf nodes.
  std::optional<SvdDecompMatrices<T>> svd_decomp;
  const Matrix2xView<T> view;
  std::array<QuadNode, 4> children;
  bool is_leaf = false;
};

template <typename T>
struct CompressTree
{
  const std::reference_wrapper<Matrix2x<T>> compressed;
  QuadNode<T> root;
};    

template <typename T>
SvdDecompMatrices<T> svd(const Matrix2xView<T>& m,
                         std::size_t b,
                         T delta);

template <typename T>
SvdDecompMatrices<T> svd(const Matrix2x<T>& m,
                         std::size_t b,
                         T delta);

template <typename T>
static CompressTree<T> compress(const Matrix2x<T>& m,
                                T delta,                 // minimum singular value
                                std::size_t rank_limit); // maximum rank

}; // namespace mutils::compress
