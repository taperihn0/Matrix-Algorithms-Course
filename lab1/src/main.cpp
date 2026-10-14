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
  
  Benchmark bench;

  /*
  bench.generate_test_matrix("/home/kserwecinski/Matrix-Algorithms-Course/lab1/matrixes",
                             2, 100, 1);
  */

  for (int i = 2; i < 3; ++i) {
    Matrix2x<mfloat> a(i, i);
    Matrix2x<mfloat> b(i, i);

    std::string a_path =
        "./matrixes/matrix_" + std::to_string(i) + "/a.txt";
    std::string b_path =
        "./matrixes/matrix_" + std::to_string(i) + "/b.txt";
    std::string c_path =
        "./matrixes/matrix_" + std::to_string(i) + "/c.txt";

    bench.load_matrix(a_path, b_path, a, b);

    //Matrix2x<mfloat> c = matrix_mul(a, b, enumMatMult::Default);

    double elapsed = bench.perform_benchmark([&]() {
        return matrix_mul(a, b, enumMatMult::Default);
    });

    std::cout << elapsed << std::endl;

    //bench.save_matrix(c_path, c, i);

    if (i % 50 == 0)
      std::cout << i << "\n";
  }

  return 0;
}