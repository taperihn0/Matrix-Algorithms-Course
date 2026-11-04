#include "benchmark.hpp"

#include "lab1/src/matrix.hpp"
#include "lu.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;

// Keeps the measured results alive so the optimizer cannot discard the work.
volatile double g_sink = 0.0;

struct Series {
  std::string name;
  enumMatMult mult;
  bool lu;
  std::ofstream time_out;
  std::ofstream op_out;
};

// Deterministic symmetric positive definite matrix: M^T * M + n * I.
// SPD means it is non-singular, so LU without pivoting is well defined.
template <typename T>
Matrix2x<T> make_spd(std::size_t n, unsigned seed) {
  Matrix2x<T> m(n, n);

  unsigned state = seed;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      state = state * 1664525u + 1013904223u;
      const float value = static_cast<float>((state >> 8) % 2001) / 1000.0f - 1.0f;
      m(i, j) = static_cast<T>(value);
    }
  }

  Matrix2x<T> result(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      T sum = static_cast<T>(0);
      for (std::size_t k = 0; k < n; ++k) {
        sum += m(k, i) * m(k, j);
      }
      result(i, j) = sum;
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    result(i, i) += static_cast<T>(n);
  }

  return result;
}

// Times fn() in seconds and adds its value to the sink.
template <typename F> double timed(F &&fn) {
  const auto begin = Clock::now();
  const double value = fn();
  const auto end = Clock::now();

  g_sink += value;

  return std::chrono::duration<double>(end - begin).count();
}

} // namespace

void Benchmark::run(const Config &cfg) {
  const std::filesystem::path stats_dir(cfg.stats_dir);
  const auto time_dir = stats_dir / "time";
  const auto op_dir = stats_dir / "op";

  std::filesystem::create_directories(time_dir);
  std::filesystem::create_directories(op_dir);

  std::array<Series, 6> series = {{
      {"inverse_default", enumMatMult::Default, false, {}, {}},
      {"inverse_binet", enumMatMult::Binet, false, {}, {}},
      {"inverse_strassen", enumMatMult::Strassen, false, {}, {}},
      {"inverse_laderman", enumMatMult::Laderman, false, {}, {}},
      {"lu_binet", enumMatMult::Binet, true, {}, {}},
      {"lu_strassen", enumMatMult::Strassen, true, {}, {}},
  }};

  for (unsigned trial = 0; trial < cfg.trials; ++trial) {
    for (auto &s : series) {
      const auto suffix = std::to_string(trial) + ".out";

      s.time_out.open(time_dir / (s.name + suffix));
      s.op_out.open(op_dir / (s.name + suffix));
    }

    for (std::size_t n = cfg.start; n < cfg.end; n += cfg.step) {
      const auto base =
          make_spd<mfloat>(n, static_cast<unsigned>(n) * 17u + trial * 131u + 3u);

      for (auto &s : series) {
        Matrix2x<mfloat> work = base; // every algorithm gets the same input
        Matrix2x<mfloat> acc(n, n);

        mfloat::reset_counters();

        double seconds;

        if (!s.lu) {
          seconds = timed([&] {
            work.inverse(acc, s.mult);
            return static_cast<double>(acc(0, 0).value());
          });
        } else {
          auto work_view = work.submatrix_view(0, 0, n, n);
          seconds = timed([&] {
            auto [L, U] = mutils::LU(work_view, s.mult);
            return static_cast<double>(L(0, 0).value() + U(0, 0).value());
          });
        }

        s.time_out << n << ',' << seconds << '\n';
        s.op_out << n << ',' << mfloat::add << ',' << mfloat::sub << ','
                 << mfloat::mul << ',' << mfloat::divs << '\n';
      }
    }

    for (auto &s : series) {
      s.time_out.close();
      s.op_out.close();
    }

    std::cout << "trial " << (trial + 1) << '/' << cfg.trials << " done\n";
  }

  std::cout << "Results written to: "
            << std::filesystem::absolute(stats_dir).string() << '\n';
}
