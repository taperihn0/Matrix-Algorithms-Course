#include "lab1/src/matrix.hpp"
#include "test_utils.hpp"

#include <cstddef>

namespace {

constexpr std::size_t TestSizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16};

void test_algorithm(enumMatMult alg, const char *name) {
  for (const auto n : TestSizes) {
    auto a = make_spd<mdouble>(n, static_cast<unsigned>(n) * 17u + 3u);
    const auto id = identity<mdouble>(n);
    const auto a_original = a;

    // Out-of-place: the input must stay untouched.
    Matrix2x<mdouble> inverse(n, n);
    a.inverse(inverse, alg);

    check_close(matrix_mul(inverse, a, alg), id, 1e-9,
                std::string(name) + " inverse, n=" + std::to_string(n));

    check_close(a, a_original, 0.0,
                std::string(name) + " leaves input untouched, n=" +
                    std::to_string(n));

    // In-place: acc aliases the input, which the recursion must tolerate.
    auto inplace = a;
    inplace.inverse(inplace, alg);

    check_close(matrix_mul(inplace, a, alg), id, 1e-9,
                std::string(name) + " in-place inverse, n=" + std::to_string(n));
  }
}

} // namespace

int main() {
  test_algorithm(enumMatMult::Default, "Default");
  test_algorithm(enumMatMult::Binet, "Binet");
  test_algorithm(enumMatMult::Strassen, "Strassen");
  test_algorithm(enumMatMult::Laderman, "Laderman");

  return test_result("inverse");
}
