#include "benchmark.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <chrono>

void Benchmark::generate_test_matrix(const std::string &s, int start, int end,
                                     int step) {
  for (int i = start; i < end; i++) {
    std::string dir_name =
        s + "/lab1/test_matrixes/matrix_" + std::to_string(i) + "/";

    std::filesystem::create_directories(dir_name);

    std::string a =
        s + "/lab1/test_matrixes/matrix_" + std::to_string(i) + "/a.txt";
    std::string b =
        s + "/lab1/test_matrixes/matrix_" + std::to_string(i) + "/b.txt";

    std::ofstream a_file(a);
    std::ofstream b_file(b);

    a_file << i << "\n";
    b_file << i << "\n";

    for (int l = 0; l < i; l++) {
      for (int w = 0; w < i; w++) {
        a_file << (rand() % 1000) / 100.0 << " ";
      }
      a_file << "\n";
    }
    a_file.close();

    for (int l = 0; l < i; l++) {
      for (int w = 0; w < i; w++) {
        b_file << (rand() % 1000) / 100.0 << " ";
      }
      b_file << "\n";
    }
    b_file.close();
  }
}

void Benchmark::load_matrix(const std::string &a, const std::string &b,
                            Matrix2x<mfloat> &ma, Matrix2x<mfloat> &mb) {
  std::ifstream a_file(a);
  std::ifstream b_file(b);
  // std::cerr << "Odczyt udany: " << !a_file.fail() << '\n';
  a_file >> ma;
  b_file >> mb;
}

void Benchmark::save_matrix(const std::string &s, Matrix2x<mfloat> &m,
                            int size) {
  std::ofstream f(s);
  f << size << "\n";
  f << m;
}