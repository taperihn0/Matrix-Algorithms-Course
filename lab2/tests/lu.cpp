#include "lab1/src/matrix.hpp"
#include "lu.hpp"
#include "test_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace {

constexpr std::size_t TestSizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16};

void test_algorithm(enumMatMult alg, const char *name) {
  for (const auto n : TestSizes) {
    auto a = make_spd<mdouble>(n, static_cast<unsigned>(n) * 29u + 7u);
    auto a_view = a.submatrix_view(0, 0, n, n);

    auto [L, U] = LU(a_view, alg);

    check_close(matrix_mul(L, U, alg), a, 1e-9,
                std::string(name) + " L*U, n=" + std::to_string(n));

    double max_off_triangle = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const double value = static_cast<double>(L(i, j).value());
        if (j > i) {
          max_off_triangle = std::max(max_off_triangle, std::abs(value));
        }
        if (i == j) {
          max_off_triangle =
              std::max(max_off_triangle, std::abs(value - 1.0));
        }

        const double upper = static_cast<double>(U(i, j).value());
        if (j < i) {
          max_off_triangle = std::max(max_off_triangle, std::abs(upper));
        }
      }
    }

    check_true(max_off_triangle <= 1e-9,
               std::string(name) + " L unit lower / U upper triangular, n=" +
                   std::to_string(n));
  }
}

} // namespace

int main() {
  test_algorithm(enumMatMult::Binet, "Binet");
  test_algorithm(enumMatMult::Strassen, "Strassen");

  return test_result("lu");
}
