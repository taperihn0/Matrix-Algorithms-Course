# lab3 tools

## SVD test fixtures

`generate_svd_tests.py` (re)writes `lab3/tests/data/svd_cases.txt` - integer
test pictures in `[0, 255]` together with the numpy `U`, `S`, `V^T` used as
the reference by the svd tests in `lab3/src/tests.hpp`. Run it after changing
`CASES` in the script:

```bash
cd lab3/tools

uv run generate_svd_tests.py
# or
python3 generate_svd_tests.py
```

The C++ side reads the fixture in Debug builds and compares its own
decomposition against it.
