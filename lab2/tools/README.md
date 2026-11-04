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

| File | Content |
| --- | --- |
| `01_czas_odwracania.png` | `t(n)` for the inverse algorithms, linear and log-log |
| `02_czas_lu.png` | `t(n)` for the LU factorizations, linear and log-log |
| `03_operacje_odwracania.png` | `op_num(n)` for the inverse algorithms |
| `04_operacje_lu.png` | `op_num(n)` for the LU factorizations |

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
