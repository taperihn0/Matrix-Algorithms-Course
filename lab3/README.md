# Matrix inverse - lab2

## Configuration

### Presets

Configure project with:

```bash
cmake --preset debug
# or
cmake --preset release
```

## Building

### After configuring presets

```bash
cmake --build --preset debug
# or
cmake --build --preset release
```

## Binary destination

After building, the executable is located at:

```text
build/MatrixMulLabs
```

## Benchmark data

Running the executable benchmarks the inverse and LU factorizations with the
counting wrappers from lab1 and writes `stats/time/` and `stats/op/`:

```bash
./build/release/MatrixMulLabs
# or with a custom range and output directory
./build/release/MatrixMulLabs --start 2 --end 129 --step 1 --trials 3 --stats ./stats
```

Options: `--start`, `--end` (exclusive), `--step`, `--trials`, `--stats`.
Each algorithm (`inverse_default`, `inverse_binet`, `inverse_strassen`,
`inverse_laderman`, `lu_binet`, `lu_strassen`) gets one file per trial.

## Plots

`tools/` contains Python utilities that plot execution time `t(n)` and operation
count `op_num(n)` from `lab2/stats/`. See `tools/README.md` for usage.
