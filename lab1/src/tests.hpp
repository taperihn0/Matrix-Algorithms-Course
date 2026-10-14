#pragma once

#include <cstdint>
#include <sstream>
#include "matrix.hpp"
#include "types.hpp"

static int g_failures = 0;

static void check(bool ok, const char* expr, int line) {
    if (!ok) {
        ++g_failures;
        std::cerr << "FAILED (line " << line << "): " << expr;
        std::cerr << '\n';
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

using Matf = Matrix2x<mfloat>;

static Matf make_matrix(const float* values, Matf::size_type rows, Matf::size_type cols) {
    Matf m(rows, cols);
    for (Matf::size_type i = 0; i < rows; ++i) {
        for (Matf::size_type j = 0; j < cols; ++j) {
            m(i, j) = values[i * cols + j];
        }
    }
    return m;
}

static void test_element_access() {
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Matf m = make_matrix(values, 2, 3);

    CHECK(m(0, 0) == mfloat(1.0f));
    CHECK(m(0, 2) == mfloat(3.0f));
    CHECK(m(1, 0) == mfloat(4.0f));
    CHECK(m(1, 2) == mfloat(6.0f));
    CHECK(m.at(1, 1) == mfloat(5.0f));

    const Matf& cm = m;
    CHECK(cm(0, 1) == mfloat(2.0f));
    CHECK(cm.at(1, 2) == mfloat(6.0f));

    m(0, 0) = 42.0f;
    CHECK(m.at(0, 0) == mfloat(42.0f));
}

static void test_arithmetic() {
    const float av[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float bv[] = {5.0f, 6.0f, 7.0f, 8.0f};
    Matf a = make_matrix(av, 2, 2);
    Matf b = make_matrix(bv, 2, 2);

    auto sum = a + b;
    CHECK(sum(0, 0) == mfloat(6.0f));
    CHECK(sum(1, 1) == mfloat(12.0f));

    auto elem = a * b;
    CHECK(elem(0, 1) == mfloat(12.0f));
    CHECK(elem(1, 0) == mfloat(21.0f));

    auto scaled = a * mfloat(2.0f);
    CHECK(scaled(1, 1) == mfloat(8.0f));

    CHECK(a != b);
    CHECK(a == a);
}

static void test_ostream() {
    const float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Matf m = make_matrix(values, 2, 3);

    std::ostringstream oss;
    oss << m;
    CHECK(oss.str() == "1 2 3\n4 5 6\n");
}

static void test_view() {
    Matf m(4, 4);
    for (Matf::size_type i = 0; i < m.rows(); ++i) {
        for (Matf::size_type j = 0; j < m.cols(); ++j) {
            m(i, j) = static_cast<float>(i * m.cols() + j);
        }
    }

    Matrix2xView<mfloat> view(m.data(), m.cols(), 0, 0, 4, 4);
    CHECK(view.rows() == 4);
    CHECK(view.cols() == 4);
    CHECK(view(0, 0) == mfloat(0.0f));
    CHECK(view(3, 3) == mfloat(15.0f));

    Matrix2xView<mfloat> sub(view, 1, 1, 2, 2);
    CHECK(sub.rows() == 2);
    CHECK(sub.cols() == 2);
    CHECK(sub(0, 0) == m.at(1, 1));
    CHECK(sub(1, 1) == m.at(2, 2));

    Matrix2xView<mfloat> nested(sub, 0, 1, 1, 1);
    CHECK(nested(0, 0) == m.at(1, 2));

    nested(0, 0) = 99.0f;
    CHECK(m.at(1, 2) == mfloat(99.0f));
}

static void test_matrix_mul_default() {
    const float av[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float bv[] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    Matf a = make_matrix(av, 2, 3);
    Matf b = make_matrix(bv, 3, 2);

    auto c = matrix_mul<mfloat>(a, b, enumMatMult::Default);
    CHECK(c.rows() == 2);
    CHECK(c.cols() == 2);
    CHECK(c(0, 0) == mfloat(58.0f));
    CHECK(c(0, 1) == mfloat(64.0f));
    CHECK(c(1, 0) == mfloat(139.0f));
    CHECK(c(1, 1) == mfloat(154.0f));
}

// Integer-valued matrices keep all intermediate float arithmetic exact,
// so results may be compared with == against the reference algorithm.
static Matf make_pattern(Matf::size_type n, bool transposed = false) {
    Matf m(n, n);
    for (Matf::size_type i = 0; i < n; ++i) {
        for (Matf::size_type j = 0; j < n; ++j) {
            m(i, j) = static_cast<float>(transposed ? (j * n + i + 1) : (i * n + j + 1));
        }
    }
    return m;
}

static Matf make_identity(Matf::size_type n) {
    Matf m(n, n);
    for (Matf::size_type i = 0; i < n; ++i) {
        for (Matf::size_type j = 0; j < n; ++j) {
            m(i, j) = (i == j) ? 1.0f : 0.0f;
        }
    }
    return m;
}

static void test_matrix_mul_strassen() {
    // 8x8: power-of-two size above the 4x4 base block -> exercises recursion
    constexpr Matf::size_type n = 8;
    const Matf a = make_pattern(n);
    const Matf b = make_pattern(n, /*transposed=*/true);
    const Matf id = make_identity(n);

    const auto a_times_id = matrix_mul<mfloat>(a, id, enumMatMult::Strassen);
    CHECK(a_times_id.rows() == n);
    CHECK(a_times_id.cols() == n);
    CHECK(a_times_id == a);

    const auto expected = matrix_mul<mfloat>(a, b, enumMatMult::Default);
    const auto got = matrix_mul<mfloat>(a, b, enumMatMult::Strassen);
    CHECK(got == expected);
}

static void test_matrix_mul_laderman() {
    // 9x9: divisible by 3 -> exercises the 3x3 block recursion
    constexpr Matf::size_type n9 = 9;
    const Matf a = make_pattern(n9);
    const Matf b = make_pattern(n9, /*transposed=*/true);
    const Matf id = make_identity(n9);

    const auto a_times_id = matrix_mul<mfloat>(a, id, enumMatMult::Laderman);
    CHECK(a_times_id.rows() == n9);
    CHECK(a_times_id.cols() == n9);
    CHECK(a_times_id == a);

    const auto expected = matrix_mul<mfloat>(a, b, enumMatMult::Default);
    const auto got = matrix_mul<mfloat>(a, b, enumMatMult::Laderman);
    CHECK(got == expected);

    // 8x8: not divisible by 3 -> exercises the remainder rows/columns path
    constexpr Matf::size_type n8 = 8;
    const Matf a8 = make_pattern(n8);
    const Matf b8 = make_pattern(n8, /*transposed=*/true);

    const auto expected8 = matrix_mul<mfloat>(a8, b8, enumMatMult::Default);
    const auto got8 = matrix_mul<mfloat>(a8, b8, enumMatMult::Laderman);
    CHECK(got8 == expected8);
}

// Pseudo-random integer matrices with values in [-16, 16] (negatives and
// zeros included). All intermediate float arithmetic stays well below 2^24,
// so every algorithm computes exact integer results comparable with ==.
static Matf make_pseudo_random(Matf::size_type n, std::uint32_t seed) {
    Matf m(n, n);
    std::uint32_t state = seed;
    for (Matf::size_type i = 0; i < n; ++i) {
        for (Matf::size_type j = 0; j < n; ++j) {
            state = state * 1664525u + 1013904223u;
            m(i, j) = static_cast<float>(static_cast<int>((state >> 16) % 33u) - 16);
        }
    }
    return m;
}

// Independent naive O(n^3) reference, living entirely in the test code.
static Matf reference_mul(const Matf& a, const Matf& b) {
    Matf r(a.rows(), b.cols());
    for (Matf::size_type i = 0; i < a.rows(); ++i) {
        for (Matf::size_type j = 0; j < b.cols(); ++j) {
            mfloat sum = mfloat(0.0f);
            for (Matf::size_type k = 0; k < a.cols(); ++k) {
                sum += a(i, k) * b(k, j);
            }
            r(i, j) = sum;
        }
    }
    return r;
}

// Sizes spread over [15, 30]: bounds, powers of two, primes and multiples
// of 2/3/4/5, hitting every algorithm's odd/even and remainder paths.
static constexpr Matf::size_type kComplexSizes[] = {15, 16, 17, 23, 24, 25, 30};

static void run_complex_mul_case(enumMatMult alg, const char* name, Matf::size_type n) {
    const Matf a = make_pseudo_random(n, 0x9E3779B9u ^ static_cast<std::uint32_t>(n));
    const Matf b = make_pseudo_random(n, 0x85EBCA6Bu + static_cast<std::uint32_t>(n) * 17u);

    const Matf expected = reference_mul(a, b);
    const Matf got = matrix_mul<mfloat>(a, b, alg);

    const std::string ctx = std::string(name) + " n=" + std::to_string(n);
    check(got.rows() == n && got.cols() == n, (ctx + ": dims").c_str(), __LINE__);
    check(got == expected, (ctx + ": equals reference").c_str(), __LINE__);
}

static void test_matrix_mul_default_complex() {
    for (const auto n : kComplexSizes) {
        run_complex_mul_case(enumMatMult::Default, "default", n);
    }
}

static void test_matrix_mul_binet_complex() {
    for (const auto n : kComplexSizes) {
        run_complex_mul_case(enumMatMult::Binet, "binet", n);
    }
}

static void test_matrix_mul_strassen_complex() {
    for (const auto n : kComplexSizes) {
        run_complex_mul_case(enumMatMult::Strassen, "strassen", n);
    }
}

static void test_matrix_mul_laderman_complex() {
    for (const auto n : kComplexSizes) {
        run_complex_mul_case(enumMatMult::Laderman, "laderman", n);
    }
}

static void test_matrix_mul_binet() {
    Matf a(8, 8);
    Matf id(8, 8);

    for (Matf::size_type i = 0; i < a.rows(); ++i) {
        for (Matf::size_type j = 0; j < a.cols(); ++j) {
            a(i, j) = static_cast<float>(i * a.cols() + j + 1);
            id(i, j) = (i == j) ? 1.0f : 0.0f;
        }
    }

    auto c = matrix_mul<mfloat>(a, id, enumMatMult::Binet);
    CHECK(c.rows() == 8);
    CHECK(c.cols() == 8);

    bool all_match = true;
    for (Matf::size_type i = 0; i < c.rows(); ++i) {
        for (Matf::size_type j = 0; j < c.cols(); ++j) {
            if (c(i, j) != a(i, j)) {
                all_match = false;
            }
        }
    }
    CHECK(all_match);
}

#ifdef DEBUG

static void test_range_check() {
    Matf m(2, 2);

    bool threw_row = false;
    try {
        (void)m.at(5, 0);
    } catch (const std::out_of_range&) {
        threw_row = true;
    }
    CHECK(threw_row);

    bool threw_col = false;
    try {
        (void)m.at(0, 5);
    } catch (const std::out_of_range&) {
        threw_col = true;
    }
    CHECK(threw_col);
}

inline int run_tests() {
    test_element_access();
    test_arithmetic();
    test_ostream();
    test_view();
    test_matrix_mul_default();
    test_matrix_mul_binet();
    test_matrix_mul_strassen();
    test_matrix_mul_laderman();
    test_matrix_mul_default_complex();
    test_matrix_mul_binet_complex();
    test_matrix_mul_strassen_complex();
    test_matrix_mul_laderman_complex();
    test_range_check();
    return g_failures;
}

const inline bool tests_status = []() {
    const bool passed = run_tests() == 0;

    if (!passed) {
        std::cout << "[TESTS] Tests failed" << std::endl;
        assert(false);
    }
    else {
        std::cout << "[TESTS] Tests passed" << std::endl;
    }

    return passed;
}();

#endif