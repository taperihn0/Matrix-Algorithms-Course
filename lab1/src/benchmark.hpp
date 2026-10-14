#pragma once

#include "matrix.hpp"
#include <string>
class Benchmark {
public:
  void generate_test_matrix(const std::string &s, int start, int end, int step);
  void load_matrix(const std::string &a, const std::string &b,
                   Matrix2x<mfloat> &ma, Matrix2x<mfloat> &mb);
  void save_matrix(const std::string &s, Matrix2x<mfloat> &m, int size);

private:
};