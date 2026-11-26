"""Generate fixed inverse test cases for the C++ tests in src/tests.hpp.

The script writes tests/data/inverse_cases.txt with, for every size:

    n
    n*n entries of A, row-major
    n*n entries of A^-1, row-major

A is a deterministic, well-conditioned SPD matrix (R * R^T + n * I), so it has
an inverse and every recursive block inverse is well defined.  The C++ side
checks all inverse algorithms against the numpy result.

Regenerate with:

    cd lab2/tools
    python3 generate_inverse_tests.py
"""

from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
OUT_PATH = TOOLS_DIR.parent / "tests" / "data" / "inverse_cases.txt"

# Big sizes only - smaller ones are covered by the generated matrices in
# src/tests.hpp.  The odd size and the non-powers of two exercise the
# recursion remainders at larger n.
SIZES = (32, 64, 75, 96, 120)


def make_matrix(n: int) -> np.ndarray:
    rng = np.random.default_rng(n)
    r = rng.standard_normal((n, n))
    return r @ r.T + n * np.eye(n)


def main() -> None:
    cases = []

    for n in SIZES:
        a = make_matrix(n)
        a_inv = np.linalg.inv(a)

        residual = np.max(np.abs(a @ a_inv - np.eye(n)))
        condition = np.linalg.cond(a)
        print(f"n={n:3d}  cond={condition:8.2f}  residual={residual:.2e}")

        if residual > 1e-10:
            raise SystemExit(f"numpy inverse for n={n} is not accurate enough")

        cases.append((a, a_inv))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w") as f:
        for a, a_inv in cases:
            f.write(f"{a.shape[0]}\n")
            np.savetxt(f, a, fmt="%.17g")
            np.savetxt(f, a_inv, fmt="%.17g")

    print(f"\nWrote {len(cases)} cases to {OUT_PATH}")


if __name__ == "__main__":
    main()
