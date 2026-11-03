#pragma once

#include "lab1/src/matrix.hpp"
#include "lu.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>

static int g_failures = 0;

static void check(bool ok, const char *expr, int line) {
  if (!ok) {
    ++g_failures;
    std::cerr << "FAILED (line " << line << "): " << expr << '\n';
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

using Matd = Matrix2x<mdouble>;

static Matd make_identity(Matd::size_type n) {
  Matd m(n, n);

  for (Matd::size_type i = 0; i < n; ++i) {
    m(i, i) = static_cast<double>(1);
  }

  return m;
}

// Deterministic, symmetric positive definite matrix: M^T * M + n * I.
// Being SPD it has an LU factorization without pivoting.
static Matd make_spd(Matd::size_type n, unsigned seed) {
  Matd m(n, n);

  unsigned state = seed;
  for (Matd::size_type i = 0; i < n; ++i) {
    for (Matd::size_type j = 0; j < n; ++j) {
      state = state * 1664525u + 1013904223u;
      const float value =
          static_cast<float>((state >> 8) % 2001) / 1000.0f - 1.0f;
      m(i, j) = static_cast<double>(value);
    }
  }

  Matd result(n, n);
  for (Matd::size_type i = 0; i < n; ++i) {
    for (Matd::size_type j = 0; j < n; ++j) {
      mdouble sum(static_cast<double>(0));
      for (Matd::size_type k = 0; k < n; ++k) {
        sum += m(k, i) * m(k, j);
      }
      result(i, j) = sum;
    }
  }

  for (Matd::size_type i = 0; i < n; ++i) {
    result(i, i) += mdouble(static_cast<double>(n));
  }

  return result;
}

static void check_close(const Matd &got, const Matd &expected, double eps,
                        const std::string &what, int line) {
  if (got.rows() != expected.rows() || got.cols() != expected.cols()) {
    ++g_failures;
    std::cerr << "FAILED (line " << line << "): " << what << ": size mismatch\n";
    return;
  }

  double max_error = 0.0;
  for (Matd::size_type i = 0; i < got.rows(); ++i) {
    for (Matd::size_type j = 0; j < got.cols(); ++j) {
      const double diff = std::abs(got(i, j).value() - expected(i, j).value());
      max_error = std::max(max_error, diff);
    }
  }

  if (!(max_error <= eps)) {
    ++g_failures;
    std::cerr << "FAILED (line " << line << "): " << what << ": max error "
              << max_error << " > " << eps << '\n';
  }
}

#define CHECK_CLOSE(got, expected, eps, what)                                  \
  check_close((got), (expected), (eps), (what), __LINE__)

// Bounds, powers of two, odd/even sizes and multiples of 3 - they hit the
// base cases, the rectangular padding and the recursion remainders.
static constexpr Matd::size_type kTestSizes[] = {1, 2, 3, 4, 5, 6,
                                                 7, 8, 9, 10, 12, 16};

static void test_inverse_algorithm(enumMatMult alg, const char *name) {
  for (const auto n : kTestSizes) {
    Matd a = make_spd(n, static_cast<unsigned>(n) * 17u + 3u);
    const Matd id = make_identity(n);
    const Matd a_original = a;

    // Out-of-place: the input must stay untouched.
    Matd inverse(n, n);
    a.inverse(inverse, alg);

    CHECK_CLOSE(matrix_mul(inverse, a, alg), id, 1e-9,
                std::string(name) + " inverse n=" + std::to_string(n));
    CHECK_CLOSE(a, a_original, 0.0,
                std::string(name) + " leaves input untouched n=" +
                    std::to_string(n));

    // In-place: acc aliases the input, which the recursion must tolerate.
    Matd inplace = a;
    inplace.inverse(inplace, alg);

    CHECK_CLOSE(matrix_mul(inplace, a, alg), id, 1e-9,
                std::string(name) + " in-place inverse n=" + std::to_string(n));
  }
}

static void test_inverse_default() {
  test_inverse_algorithm(enumMatMult::Default, "default");
}

static void test_inverse_binet() {
  test_inverse_algorithm(enumMatMult::Binet, "binet");
}

static void test_inverse_strassen() {
  test_inverse_algorithm(enumMatMult::Strassen, "strassen");
}

static void test_inverse_laderman() {
  test_inverse_algorithm(enumMatMult::Laderman, "laderman");
}

static void test_lu_algorithm(enumMatMult alg, const char *name) {
  for (const auto n : kTestSizes) {
    Matd a = make_spd(n, static_cast<unsigned>(n) * 29u + 7u);
    auto a_view = a.submatrix_view(0, 0, n, n);

    auto [L, U] = LU(a_view, alg);

    CHECK_CLOSE(matrix_mul(L, U, alg), a, 1e-9,
                std::string(name) + " L*U n=" + std::to_string(n));

    // L is unit lower triangular, U is upper triangular.
    double max_off_triangle = 0.0;
    for (Matd::size_type i = 0; i < n; ++i) {
      for (Matd::size_type j = 0; j < n; ++j) {
        if (j > i) {
          max_off_triangle =
              std::max(max_off_triangle, std::abs(L(i, j).value()));
        }
        if (i == j) {
          max_off_triangle =
              std::max(max_off_triangle, std::abs(L(i, j).value() - 1.0));
        }
        if (j < i) {
          max_off_triangle =
              std::max(max_off_triangle, std::abs(U(i, j).value()));
        }
      }
    }

    check(max_off_triangle <= 1e-9,
          (std::string(name) + " L unit lower / U upper triangular n=" +
           std::to_string(n))
              .c_str(),
          __LINE__);
  }
}

static void test_lu_binet() { test_lu_algorithm(enumMatMult::Binet, "binet"); }

static void test_lu_strassen() {
  test_lu_algorithm(enumMatMult::Strassen, "strassen");
}

#ifdef DEBUG

inline int run_tests() {
  test_inverse_default();
  test_inverse_binet();
  test_inverse_strassen();
  test_inverse_laderman();
  test_lu_binet();
  test_lu_strassen();
  return g_failures;
}

const inline bool tests_status = []() {
  const bool passed = run_tests() == 0;

  if (!passed) {
    std::cout << "[TESTS] Tests failed" << std::endl;
    assert(false);
  } else {
    std::cout << "[TESTS] Tests passed" << std::endl;
  }

  return passed;
}();

#endif
