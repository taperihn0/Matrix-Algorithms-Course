#include "benchmark.hpp"
#include "tests.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage(const char *program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "  --start <n>     first matrix size (default 2)\n"
            << "  --end <n>       one past the last matrix size (default 257)\n"
            << "  --step <n>      size increment (default 1)\n"
            << "  --trials <n>    repetitions per size (default 3)\n"
            << "  --stats <dir>   output directory (default ./stats)\n"
            << "  --help          show this message\n";
}

} // namespace

int main(int argc, char **argv) {
  Benchmark::Config cfg;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];

    auto next_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << '\n';
        std::exit(1);
      }
      return argv[++i];
    };

    if (arg == "--start") {
      cfg.start = std::stoul(next_value("--start"));
    } else if (arg == "--end") {
      cfg.end = std::stoul(next_value("--end"));
    } else if (arg == "--step") {
      cfg.step = std::stoul(next_value("--step"));
    } else if (arg == "--trials") {
      cfg.trials = static_cast<unsigned>(std::stoul(next_value("--trials")));
    } else if (arg == "--stats") {
      cfg.stats_dir = next_value("--stats");
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << arg << '\n';
      print_usage(argv[0]);
      return 1;
    }
  }

  if (cfg.step == 0 || cfg.trials == 0 || cfg.start == 0 || cfg.start >= cfg.end) {
    std::cerr << "Invalid benchmark range\n";
    print_usage(argv[0]);
    return 1;
  }

  Benchmark bench;
  bench.run(cfg);

  return 0;
}
