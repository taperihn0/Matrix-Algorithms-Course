#include "benchmark.hpp"
#include "matrix.hpp"
#include <iostream>
#include <string>

int main() {
  // Jawne użycie typu mfloat dla każdego elementu
  /*
  Matrix2x<mfloat> m({{mfloat(1.0f), mfloat(2.0f)},
                      {mfloat(3.0f), mfloat(1.0f)},
                      {mfloat(2.0f), mfloat(3.0f)}});

  Matrix2x<mfloat> n({{mfloat(1.0f), mfloat(2.0f), mfloat(3.0f)},
                      {mfloat(1.0f), mfloat(2.0f), mfloat(3.0f)}});

  std::cout << m << std::endl;

  Matrix2x<mfloat> result = matrix_mul(m, n, enumMatMult::Default);

  std::cout << "HEJKA\n" << result << std::endl;
  */
  /*

  bench.generate_test_matrix("/Users/kajetanserwecinski/Documents/Programming/"
                             "Matrix-Algorithms-Course/lab1/lab1/",
                             2, 1000, 1);
  */
  Benchmark bench;
  for (int i = 2; i < 1000; ++i) {
    Matrix2x<mfloat> a(i, i);
    Matrix2x<mfloat> b(i, i);

    std::string a_path =
        "./lab1/lab1/test_matrixes/matrix_" + std::to_string(i) + "/a.txt";
    std::string b_path =
        "./lab1/lab1/test_matrixes/matrix_" + std::to_string(i) + "/b.txt";
    std::string c_path =
        "./lab1/lab1/test_matrixes/matrix_" + std::to_string(i) + "/c.txt";

    bench.load_matrix(a_path, b_path, a, b);

    Matrix2x<mfloat> c = matrix_mul(a, b, enumMatMult::Default);

    bench.save_matrix(c_path, c, i);

    if (i % 50 == 0)
      std::cout << i << "\n";
  }

  return 0;
}