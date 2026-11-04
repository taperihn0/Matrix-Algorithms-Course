#pragma once

#include <cstddef>
#include <string>

// Generates the timing and operation-count data consumed by tools/main.py.
//
// For every algorithm and trial two files are written:
//   <stats>/time/<name><trial>.out    n,time_seconds
//   <stats>/op/<name><trial>.out      n,add,sub,mul,div
class Benchmark {
public:
  struct Config {
    std::size_t start = 2;      // first matrix size
    std::size_t end = 350;      // one past the last matrix size
    std::size_t step = 1;       // size increment
    unsigned trials = 3;        // repetitions per size
    std::string stats_dir = "./stats";
  };

  void run(const Config &cfg);
};
