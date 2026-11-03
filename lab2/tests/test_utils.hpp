#pragma once

#include "lab1/src/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

inline int &test_failures() {
  static int failures = 0;
  return failures;
}

inline void check_true(bool condition, const std::string &what) {
  if (!condition) {
    std::fprintf(stderr, "[FAIL] %s\n", what.c_str());
    ++test_failures();
  }
}

template <typename T> Matrix2x<T> identity(std::size_t n) {
  Matrix2x<T> result(n, n);

  for (std::size_t i = 0; i < n; ++i) {
    result(i, i) = static_cast<T>(1);
  }

  return result;
}

// Deterministic, symmetric positive definite matrix: M^T * M + n * I.
// Being SPD it has an LU factorization without pivoting.
template <typename T> Matrix2x<T> make_spd(std::size_t n, unsigned seed) {
  Matrix2x<T> m(n, n);

  unsigned state = seed;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      state = state * 1664525u + 1013904223u;
      const float value =
          static_cast<float>((state >> 8) % 2001) / 1000.0f - 1.0f;
      m(i, j) = static_cast<T>(value);
    }
  }

  Matrix2x<T> result(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      T sum = static_cast<T>(0);
      for (std::size_t k = 0; k < n; ++k) {
        sum += m(k, i) * m(k, j);
      }
      result(i, j) = sum;
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    result(i, i) += static_cast<T>(static_cast<float>(n));
  }

  return result;
}

template <typename T>
void check_close(const Matrix2x<T> &got, const Matrix2x<T> &expected,
                 double eps, const std::string &what) {
  if (got.rows() != expected.rows() || got.cols() != expected.cols()) {
    std::fprintf(stderr, "[FAIL] %s: size mismatch (%zux%zu vs %zux%zu)\n",
                 what.c_str(), got.rows(), got.cols(), expected.rows(),
                 expected.cols());
    ++test_failures();
    return;
  }

  double max_error = 0.0;
  for (std::size_t i = 0; i < got.rows(); ++i) {
    for (std::size_t j = 0; j < got.cols(); ++j) {
      const double diff = std::abs(static_cast<double>(got(i, j).value()) -
                                   static_cast<double>(expected(i, j).value()));
      max_error = std::max(max_error, diff);
    }
  }

  if (!(max_error <= eps)) {
    std::fprintf(stderr, "[FAIL] %s: max error %.3g > %.3g\n", what.c_str(),
                 max_error, eps);
    ++test_failures();
  }
}

inline int test_result(const char *suite) {
  if (test_failures() == 0) {
    std::printf("[PASS] %s\n", suite);
    return 0;
  }

  std::printf("[FAIL] %s: %d failed check(s)\n", suite, test_failures());
  return 1;
}
