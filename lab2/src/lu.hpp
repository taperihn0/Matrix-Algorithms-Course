#pragma once

#include "lab1/src/matrix.hpp"

#include <utility>

// Block LU factorization without pivoting: A = L * U.
// The block products are computed with the requested multiplication
// algorithm (Binet or Strassen). Requires a square, non-singular matrix.
template <typename T>
std::pair<Matrix2x<T>, Matrix2x<T>> LU(Matrix2xView<T> &a, enumMatMult type);
