#pragma once

#include <cstddef>
#include <string>

class Benchmark {
public:
  struct Config {
    std::size_t start = 2;
    std::size_t end = 1700;
    std::size_t step = 25; 
    unsigned trials = 3;  
    std::string stats_dir = "./stats";
  };

  void run(const Config &cfg);
};
