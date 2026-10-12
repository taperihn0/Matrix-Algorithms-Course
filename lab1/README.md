# Matrix multiplication - lab1

A small CMake project that builds and runs the sources in `src/`.

## Configuration

### Presets

This repository includes CMake presets:

- `dev` — debug build
- `release` — release build

Configure with:

```bash
cmake --preset dev
# or
cmake --preset release
```

### Manual configuration

If you do not want to use presets:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
```

## Building

### With presets

```bash
cmake --build --preset dev
# or
cmake --build --preset release
```

### Manual build

```bash
cmake --build build
```

## Binary destination

After building, the executable is located at:

```text
build/MatrixMulLabs
```

Run it from the project root with:

```bash
./build/MatrixMulLabs
```

Build is performed out-of-source in `build/`.
