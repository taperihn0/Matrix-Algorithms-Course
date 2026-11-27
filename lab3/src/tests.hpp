#pragma once

#include "compress.hpp"
#include "lab1/src/matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const char *expr, int line) {
  if (!ok) {
    ++g_failures;
    std::cerr << "FAILED (line " << line << "): " << expr << '\n';
  }
}

static void check_le(double value, double limit, const std::string &what,
                     int line) {
  if (!(value <= limit)) {
    ++g_failures;
    std::cerr << "FAILED (line " << line << "): " << what << ": " << value
              << " > " << limit << '\n';
  }
}

#define CHECK(cond) check((cond), #cond, __LINE__)
#define CHECK_LE(value, limit, what) check_le((value), (limit), (what), __LINE__)

using Matd = Matrix2x<double>;

// the picture channels: 8 bit values in [0, 255].
using ChannelPicture = Matrix2x<std::uint8_t>;

#ifndef LAB3_SVD_TEST_DATA_FILE
#define LAB3_SVD_TEST_DATA_FILE "tests/data/svd_cases.txt"
#endif

// tolerance of the singular values, relative to their magnitude.
constexpr double kSigmaTol = 1e-9;
// entries of u and v_trans are unit vectors, so absolute.
constexpr double kVectorTol = 1e-7;
constexpr double kOrthonormalTol = 1e-8;
// relative to the channel range, so |a| <= 255.
constexpr double kReconstructionTol = 1e-8;

struct SvdCase {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::size_t max_rank = 0;
  std::size_t rank = 0;
  double delta = 0.0;
  // frobenius norm of the components the reference factorization dropped.
  double reconstruction_tail = 0.0;
  std::vector<std::uint8_t> values;
  std::vector<double> sigma;
  std::vector<double> u;
  std::vector<double> v_trans;
};

// Reads tests/data/svd_cases.txt, written by tools/generate_svd_tests.py:
// rows cols max_rank delta reconstruction_tail, A row-major (integers in
// [0, 255]), rank, the singular values, U row-major and V^T row-major,
// repeating.
static std::vector<SvdCase> load_svd_cases() {
  std::vector<SvdCase> cases;
  std::ifstream in(LAB3_SVD_TEST_DATA_FILE);

  if (!in) {
    ++g_failures;
    std::cerr << "FAILED: cannot open svd test data " << LAB3_SVD_TEST_DATA_FILE
              << '\n';
    return cases;
  }

  while (true) {
    SvdCase test_case;

    if (!(in >> test_case.rows)) {
      break;
    }

    in >> test_case.cols >> test_case.max_rank >> test_case.delta >>
        test_case.reconstruction_tail;

    const std::size_t count = test_case.rows * test_case.cols;

    test_case.values.resize(count);

    // read through int: operator>> on uint8_t would read characters.
    for (std::uint8_t &value : test_case.values) {
      int raw = 0;
      in >> raw;

      if (!in || raw < 0 || raw > 255) {
        ++g_failures;
        std::cerr << "FAILED: invalid svd test data "
                  << LAB3_SVD_TEST_DATA_FILE << '\n';
        return {};
      }

      value = static_cast<std::uint8_t>(raw);
    }

    in >> test_case.rank;

    test_case.sigma.resize(test_case.rank);
    test_case.u.resize(test_case.rows * test_case.rank);
    test_case.v_trans.resize(test_case.rank * test_case.cols);

    for (double &value : test_case.sigma) {
      in >> value;
    }
    for (double &value : test_case.u) {
      in >> value;
    }
    for (double &value : test_case.v_trans) {
      in >> value;
    }

    if (!in) {
      ++g_failures;
      std::cerr << "FAILED: truncated svd test data "
                << LAB3_SVD_TEST_DATA_FILE << '\n';
      return {};
    }

    cases.push_back(std::move(test_case));
  }

  if (!in.eof() || cases.empty()) {
    ++g_failures;
    std::cerr << "FAILED: invalid svd test data " << LAB3_SVD_TEST_DATA_FILE
              << '\n';
    return {};
  }

  return cases;
}

static Matd to_matrix(const std::vector<double> &values, std::size_t rows,
                      std::size_t cols) {
  Matd m(rows, cols);

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      m(i, j) = values[i * cols + j];
    }
  }

  return m;
}

static double max_abs_diff(const Matd &got, const Matd &expected) {
  double max_error = 0.0;

  for (std::size_t i = 0; i < got.rows(); ++i) {
    for (std::size_t j = 0; j < got.cols(); ++j) {
      max_error = std::max(max_error, std::abs(got(i, j) - expected(i, j)));
    }
  }

  return max_error;
}

static double max_abs_diff(const std::vector<double> &got,
                           const std::vector<double> &expected) {
  double max_error = 0.0;

  for (std::size_t i = 0; i < got.size(); ++i) {
    max_error = std::max(max_error, std::abs(got[i] - expected[i]));
  }

  return max_error;
}

// A singular triplet (sigma, u, v) is only defined up to the sign of u and v,
// so flip every component the same way numpy did when writing the fixture.
static void canonicalize_signs(Matd &u, Matd &v_trans) {
  for (std::size_t k = 0; k < u.cols(); ++k) {
    std::size_t pivot = 0;

    for (std::size_t i = 1; i < u.rows(); ++i) {
      if (std::abs(u(i, k)) > std::abs(u(pivot, k))) {
        pivot = i;
      }
    }

    if (u(pivot, k) >= 0.0) {
      continue;
    }

    for (std::size_t i = 0; i < u.rows(); ++i) {
      u(i, k) = -u(i, k);
    }
    for (std::size_t j = 0; j < v_trans.cols(); ++j) {
      v_trans(k, j) = -v_trans(k, j);
    }
  }
}

static void test_svd_cases() {
  for (const auto &test_case : load_svd_cases()) {
    const std::string what = "svd " + std::to_string(test_case.rows) + "x" +
                             std::to_string(test_case.cols);

    // the fixture is a picture: store the channels exactly as loaded from a
    // file, then widen them, because 8 bit arithmetic overflows.
    ChannelPicture pixels(test_case.rows, test_case.cols);

    for (std::size_t i = 0; i < test_case.rows; ++i) {
      for (std::size_t j = 0; j < test_case.cols; ++j) {
        pixels(i, j) = test_case.values[i * test_case.cols + j];
      }
    }

    Matd a(test_case.rows, test_case.cols);

    for (std::size_t i = 0; i < test_case.rows; ++i) {
      for (std::size_t j = 0; j < test_case.cols; ++j) {
        a(i, j) = static_cast<double>(pixels(i, j));
      }
    }

    const auto got =
        mutils::compress::svd(a, test_case.max_rank, test_case.delta);

    CHECK(got.sigma.size() == test_case.rank);

    if (got.sigma.size() != test_case.rank) {
      continue;
    }

    CHECK(got.u.rows() == test_case.rows);
    CHECK(got.u.cols() == got.sigma.size());
    CHECK(got.v_trans.rows() == got.sigma.size());
    CHECK(got.v_trans.cols() == test_case.cols);

    const double sigma_limit =
        kSigmaTol * (1.0 + (test_case.sigma.empty() ? 0.0 : test_case.sigma[0]));
    const double sigma_error = max_abs_diff(got.sigma, test_case.sigma);

    auto u = got.u;
    auto v_trans = got.v_trans;
    canonicalize_signs(u, v_trans);

    const double u_error =
        max_abs_diff(u, to_matrix(test_case.u, test_case.rows, got.sigma.size()));
    const double v_error = max_abs_diff(
        v_trans, to_matrix(test_case.v_trans, got.sigma.size(), test_case.cols));

    // our own factors have to be orthonormal as well
    double orth_error = 0.0;

    for (std::size_t p = 0; p < got.sigma.size(); ++p) {
      for (std::size_t q = 0; q < got.sigma.size(); ++q) {
        const double want = p == q ? 1.0 : 0.0;

        double u_dot = 0.0;
        for (std::size_t i = 0; i < test_case.rows; ++i) {
          u_dot += u(i, p) * u(i, q);
        }
        orth_error = std::max(orth_error, std::abs(u_dot - want));

        double v_dot = 0.0;
        for (std::size_t j = 0; j < test_case.cols; ++j) {
          v_dot += v_trans(p, j) * v_trans(q, j);
        }
        orth_error = std::max(orth_error, std::abs(v_dot - want));
      }
    }

    Matd reconstruction(test_case.rows, test_case.cols);

    for (std::size_t i = 0; i < test_case.rows; ++i) {
      for (std::size_t j = 0; j < test_case.cols; ++j) {
        double sum = 0.0;

        for (std::size_t k = 0; k < got.sigma.size(); ++k) {
          sum += got.sigma[k] * got.u(i, k) * got.v_trans(k, j);
        }

        reconstruction(i, j) = sum;
      }
    }

    const double reconstruction_error = max_abs_diff(reconstruction, a);

    std::cout << "[TESTS] " << what << " rank=" << test_case.rank
              << " sigma_err=" << sigma_error << " u_err=" << u_error
              << " v_err=" << v_error << " orth_err=" << orth_error
              << " recon_err=" << reconstruction_error << '\n';

    CHECK_LE(sigma_error, sigma_limit, what + " sigma");
    CHECK_LE(u_error, kVectorTol, what + " u");
    CHECK_LE(v_error, kVectorTol, what + " v_trans");
    CHECK_LE(orth_error, kOrthonormalTol, what + " orthonormality");
    CHECK_LE(reconstruction_error,
             test_case.reconstruction_tail + kReconstructionTol * (1.0 + 255.0),
             what + " reconstruction");
  }
}

inline int run_tests() {
  test_svd_cases();
  return g_failures;
}

// Runs the tests at startup and reports the result.
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
