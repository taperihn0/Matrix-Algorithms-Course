#include "benchmark.hpp"
#include "matrix.hpp"
#include "tests.hpp"

int main() {
  Benchmark bench;
  /*
  bench.generate_test_matrix("/home/kserwecinski/Matrix-Algorithms-Course/lab1/matrixes",
                             2, 100, 1);
  */
  bench.bench("./lab1/lab1/test_matrixes/", "./stats/time/", "./stats/op/");

  return 0;
}
