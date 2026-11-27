"""Generate fixed SVD test cases for the C++ tests in src/tests.hpp.

The pictures are 8 bit channels, so every test matrix holds integers in
[0, 255] - the same data ``Matrix2x<std::uint8_t>`` gets from
``src/picture.hpp``.  numpy is the reference implementation; for every case
the script writes ``tests/data/svd_cases.txt``:

    rows cols max_rank delta reconstruction_tail
    rows*cols integers of A, row-major
    rank
    rank singular values, descending
    rows*rank entries of U, row-major
    rank*cols entries of V^T, row-major

Only components with a singular value above ``delta`` are stored, capped at
``max_rank`` - exactly what the C++ implementation is expected to return.
``reconstruction_tail`` is the frobenius norm of the dropped components, so the
C++ side knows how much reconstruction error a truncated case may have.

Columns of U are sign normalized (largest magnitude entry positive) together
with the matching row of V^T, so the comparison does not depend on the sign
convention of either implementation.  Cases are rejected (and the random ones
regenerated) when the retained singular values are not separated enough for a
sign comparison to be meaningful, or when the sign pivot is ambiguous.

Regenerate with:

    cd lab3/tools
    python3 generate_svd_tests.py
"""

from pathlib import Path

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
OUT_PATH = TOOLS_DIR.parent / "tests" / "data" / "svd_cases.txt"

# (rows, cols, max_rank, kind)
CASES = (
    (1, 1, 1, "random"),
    (2, 3, 2, "random"),
    (3, 2, 3, "random"),
    (4, 4, 4, "random"),
    (6, 8, 6, "random"),
    (8, 6, 8, "random"),
    (12, 10, 12, "random"),
    (16, 16, 16, "random"),
    (8, 8, 3, "random"),     # truncated on purpose: three dominant triplets
    (6, 7, 6, "constant"),   # rank one
    (5, 4, 5, "rank_two"),   # rank two
    (4, 5, 4, "zero"),       # rank zero
)

# A singular value below this fraction of the largest one counts as zero.
RELATIVE_DELTA = 1e-9

# Power iteration converges with the ratio of consecutive singular values, so
# the retained ones have to be clearly separated for the reference to be
# reproducible.
MAX_RATIO = 0.95

MAX_ATTEMPTS = 4000


def random_picture(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    return rng.integers(0, 256, size=(rows, cols), dtype=np.uint8)


def constant_picture(rows: int, cols: int) -> np.ndarray:
    return np.full((rows, cols), 42, dtype=np.uint8)


def rank_two_picture(rows: int, cols: int) -> np.ndarray:
    u = np.arange(1, rows + 1) % 17
    v = np.arange(1, cols + 1) % 13
    a = np.outer(u, np.ones(cols, dtype=int)) + np.outer(np.ones(rows, dtype=int), v)
    return a.astype(np.uint8)


def zero_picture(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols), dtype=np.uint8)


BUILDERS = {
    "constant": constant_picture,
    "rank_two": rank_two_picture,
    "zero": zero_picture,
}


def is_separated(sigma: np.ndarray) -> bool:
    return all(sigma[i + 1] <= MAX_RATIO * sigma[i] for i in range(sigma.size - 1))


def has_clear_sign_pivot(u: np.ndarray) -> bool:
    for k in range(u.shape[1]):
        column = u[:, k]
        magnitudes = np.abs(column)
        top = magnitudes.max()

        # entries practically tied with the pivot must share its sign, otherwise
        # a rounding difference can pick the opposite pivot and flip the sign.
        tied = column[magnitudes >= top * (1.0 - 1e-9)]

        if np.any(np.sign(tied) != np.sign(tied[0])):
            return False

    return True


def evaluate(a: np.ndarray, max_rank: int):
    u, sigma, v_trans = np.linalg.svd(a.astype(np.float64), full_matrices=False)

    delta = RELATIVE_DELTA * sigma[0] if sigma.size else 0.0
    kept = int(np.count_nonzero(sigma > delta)) if delta > 0.0 else 0
    rank = min(kept, max_rank)

    # the part of the picture the factorization is allowed to drop
    dropped = sigma[rank:]
    tail = float(np.sqrt(np.sum(dropped * dropped)))

    u = u[:, :rank].copy()
    v_trans = v_trans[:rank, :].copy()
    sigma = sigma[:rank].copy()

    if rank >= 2 and not is_separated(sigma):
        return None
    if rank >= 1 and not has_clear_sign_pivot(u):
        return None

    # canonical signs: the largest magnitude entry of every u column is positive
    for k in range(rank):
        pivot = int(np.argmax(np.abs(u[:, k])))
        if u[pivot, k] < 0.0:
            u[:, k] = -u[:, k]
            v_trans[k, :] = -v_trans[k, :]

    residual = np.abs(a - (u * sigma) @ v_trans).max() if rank else np.abs(a).max()

    scale = max(1.0, float(sigma[0]) if rank else 1.0)

    if residual > tail + 1e-10 * scale:
        raise SystemExit(
            f"numpy svd residual {residual:.3e} is above the dropped tail "
            f"{tail:.3e}"
        )

    return {
        "a": a,
        "max_rank": max_rank,
        "delta": delta,
        "sigma": sigma,
        "u": u,
        "v_trans": v_trans,
        "residual": residual,
        "tail": tail,
    }


def build_case(rows: int, cols: int, max_rank: int, kind: str):
    if kind != "random":
        case = evaluate(BUILDERS[kind](rows, cols), max_rank)

        if case is None:
            raise SystemExit(f"special case {kind} {rows}x{cols} is degenerate")

        return case

    rng = np.random.default_rng(rows * 1000 + cols)

    for _ in range(MAX_ATTEMPTS):
        case = evaluate(random_picture(rng, rows, cols), max_rank)

        if case is not None:
            return case

    raise SystemExit(f"no well separated random case found for {rows}x{cols}")


def write_grid(f, values: np.ndarray) -> None:
    for row in values:
        f.write(" ".join(f"{float(v):.17g}" for v in row) + "\n")


def main() -> None:
    cases = []

    for rows, cols, max_rank, kind in CASES:
        case = build_case(rows, cols, max_rank, kind)
        case["rows"] = rows
        case["cols"] = cols
        cases.append(case)

        sigma = case["sigma"]
        shown = ", ".join(f"{v:.6g}" for v in sigma)
        print(
            f"{rows:3d}x{cols:<3d} {kind:9s} max_rank={max_rank:2d} "
            f"rank={sigma.size:2d} residual={case['residual']:.2e} "
            f"tail={case['tail']:.2e} sigma=[{shown}]"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w") as f:
        for case in cases:
            a = case["a"]
            sigma = case["sigma"]

            f.write(
                f"{case['rows']} {case['cols']} {case['max_rank']} "
                f"{case['delta']:.17g} {case['tail']:.17g}\n"
            )
            f.write(" ".join(str(int(v)) for v in a.ravel()) + "\n")
            f.write(f"{sigma.size}\n")
            f.write(" ".join(f"{v:.17g}" for v in sigma) + "\n")
            write_grid(f, case["u"])
            write_grid(f, case["v_trans"])

    print(f"\nWrote {len(cases)} cases to {OUT_PATH}")


if __name__ == "__main__":
    main()
