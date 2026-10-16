#pragma once

#include "matrix.hpp"
#include <chrono>
#include <fstream>
#include <string>

class Benchmark {
public:
  void bench(const std::string &s, const std::string &stats,
             const std::string &op);

  void generate_test_matrix(const std::string &s, int start, int end, int step);
  void load_matrix(const std::string &a, const std::string &b,
                   Matrix2x<mfloat> &ma, Matrix2x<mfloat> &mb);
  void save_matrix(const std::string &s, Matrix2x<mfloat> &m, int size);

  template <typename Op>
  double perform_benchmark(Op operation, std::ofstream &of) {
    mfloat::reset_counters();

    const auto start = std::chrono::steady_clock::now();
    auto result = operation();
    const auto end = std::chrono::steady_clock::now();

    double time =
        std::chrono::duration<double, std::milli>(end - start).count();

    of << mfloat::add << "," << mfloat::sub << "," << mfloat::mul << ","
       << mfloat::divs << "\n";

    return time;
  }

private:
};