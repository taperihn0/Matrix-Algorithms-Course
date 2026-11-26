# lab2 tools

Plotting utilities for lab2, in the spirit of `lab1/tools`.

## Setup

```bash
cd lab2/tools

uv sync
# or
pip install -e .
```

## Generating data

The figures are built from `lab2/stats/`, produced by the lab2 executable:

```bash
./build/release/MatrixMulLabs --start 2 --end 257 --trials 3 --stats ../stats
```

## Usage

```bash
uv run main.py
# or
python main.py
```

Figures are written to `lab2/tools/plots/`:

Each metric is saved as two separate figures: a linear plot and a log-log plot.

| File | Content |
| --- | --- |
| `01_czas_odwracania.png` | `t(n)` for the inverse algorithms, linear |
| `01_czas_odwracania_loglog.png` | `t(n)` for the inverse algorithms, log-log |
| `02_czas_lu.png` | `t(n)` for the LU factorizations, linear |
| `02_czas_lu_loglog.png` | `t(n)` for the LU factorizations, log-log |
| `03_operacje_odwracania.png` | `op_num(n)` for the inverse algorithms, linear |
| `03_operacje_odwracania_loglog.png` | `op_num(n)` for the inverse algorithms, log-log |
| `04_operacje_lu.png` | `op_num(n)` for the LU factorizations, linear |
| `04_operacje_lu_loglog.png` | `op_num(n)` for the LU factorizations, log-log |
| `05_flops_odwracania.png` | `op_num(n) / t(n)` for the inverse algorithms, linear |
| `05_flops_odwracania_loglog.png` | `op_num(n) / t(n)` for the inverse algorithms, log-log |
| `06_flops_lu.png` | `op_num(n) / t(n)` for the LU factorizations, linear |
| `06_flops_lu_loglog.png` | `op_num(n) / t(n)` for the LU factorizations, log-log |

Times are stored in seconds in `stats/time/` but plotted in milliseconds.

FLOPS is operations per second: the mean `op_num(n)` from `stats/op/` divided
by the mean time in seconds from `stats/time/` (counted add/sub/mul/div, each
worth one operation). The two files are joined on the sizes they share.

## Inverse test fixtures

`generate_inverse_tests.py` (re)writes `lab2/tests/data/inverse_cases.txt` —
fixed SPD matrices with numpy-computed inverses used by the big inverse tests
in `lab2/src/tests.hpp`. Run it after changing the sizes in the script:

```bash
uv run generate_inverse_tests.py
# or
python3 generate_inverse_tests.py
```

## Input data

The script reads `lab2/stats/` and plots the mean over all trials found. The
algorithm name is the file name with the trial index stripped, so
`inverse_strassen0.out`, `inverse_strassen1.out`, ... are one series called
`Strassen`, plotted only in the inverse figures (`lu_*` files get their own).

```text
lab2/stats/time/<name><trial>.out    n,time_seconds
lab2/stats/op/<name><trial>.out      n,add,sub,mul,div
```

`stats/op` files may also omit the size column (`add,sub,mul,div`); the sizes
are then taken as consecutive values starting at `OP_START_N` (2), as in lab1.
