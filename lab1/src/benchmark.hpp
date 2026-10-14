#pragma once

#include "matrix.hpp"
#include <string>
#include <chrono>

class Benchmark {
public:
  void generate_test_matrix(const std::string &s, int start, int end, int step);
  void load_matrix(const std::string &a, const std::string &b,
                   Matrix2x<mfloat> &ma, Matrix2x<mfloat> &mb);
  void save_matrix(const std::string &s, Matrix2x<mfloat> &m, int size);

  template <typename Op>
  double perform_benchmark(Op operation){
    mfloat::reset_counters();

    const auto start = std::chrono::steady_clock::now();
    auto result = operation();
    const auto end = std::chrono::steady_clock::now();

    double time = std::chrono::duration<double, std::milli>(
                      end - start
                  ).count();
    
    std::cout << "Add: " << mfloat::add << " Sub: " << mfloat::sub << " Mul: " << mfloat::mul << " Div: " << mfloat::divs << std::endl;
    
    return time;
  }
private:
};