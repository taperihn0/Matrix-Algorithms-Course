#include "benchmark.hpp"
#include "matrix.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

void Benchmark::bench(const std::string &s, const std::string &stats,
                      const std::string &op) {
  std::ofstream fstandard(stats + "standart.out");
  std::ofstream fbinet(stats + "binet.out");
  std::ofstream fladerman(stats + "laderman.out");
  std::ofstream fstrassen(stats + "strassen.out");

  std::ofstream fstandard_op(op + "standart.out");
  std::ofstream fbinet_op(op + "binet.out");
  std::ofstream fladerman_op(op + "laderman.out");
  std::ofstream fstrassen_op(op + "strassen.out");

  for (int i = 2; i < 500; ++i) {
    Matrix2x<mfloat> a(i, i);
    Matrix2x<mfloat> b(i, i);

    std::string a_path = "./matrixes/matrix_" + std::to_string(i) + "/a.txt";
    std::string b_path = "./matrixes/matrix_" + std::to_string(i) + "/b.txt";
    std::string c_path = "./matrixes/matrix_" + std::to_string(i) + "/c.txt";

    this->load_matrix(a_path, b_path, a, b);

    double standard = this->perform_benchmark(
        [&]() -> Matrix2x<mfloat> {
          return matrix_mul(a, b, enumMatMult::Default);
        },
        fstandard_op);

    double binet = this->perform_benchmark(
        [&]() -> Matrix2x<mfloat> {
          return matrix_mul(a, b, enumMatMult::Binet);
        },
        fbinet_op);

    double strassen = this->perform_benchmark(
        [&]() -> Matrix2x<mfloat> {
          return matrix_mul(a, b, enumMatMult::Strassen);
        },
        fstrassen_op);

    double laderman = this->perform_benchmark(
        [&]() -> Matrix2x<mfloat> {
          return matrix_mul(a, b, enumMatMult::Laderman);
        },
        fladerman_op);

    fstandard << i << "," << standard << "\n";
    fbinet << i << "," << binet << "\n";
    fstrassen << i << "," << strassen << "\n";
    fladerman << i << "," << laderman << "\n";
  }
}

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