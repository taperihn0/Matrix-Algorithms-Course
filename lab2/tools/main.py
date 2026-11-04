"""
Generowanie wykresów zależności czasu wykonania t(n) oraz liczby operacji
op_num(n) od rozmiaru macierzy, osobno dla odwracania macierzy i rozkładu LU.

Dane z folderów stats/time/ i stats/op/.
Każdy algorytm ma kilka prób (pliki 0, 1, ...), wyciągamy średnią.

File formats
------------
stats/time/<name><trial>.out:
    n,time_seconds           (one row per matrix size)

stats/op/<name><trial>.out:
    n,add,sub,mul,div        (one row per matrix size)
or the lab1 style without the size column (sizes start at OP_START_N):
    add,sub,mul,div
"""

import os
import re

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
STATS_DIR = os.path.normpath(os.path.join(TOOLS_DIR, "..", "stats"))
PLOTS_DIR = os.path.join(TOOLS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Used only for op files without an explicit size column.
OP_START_N = 2

TRIAL_SUFFIX = re.compile(r"\d+$")

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# ──────────────────────────────────────────────
# Loading data
# ──────────────────────────────────────────────
def discover_trials(kind: str) -> dict[str, list[str]]:
    """Groups ``stats/<kind>/*.out`` files by algorithm name (trial index stripped)."""
    directory = os.path.join(STATS_DIR, kind)
    if not os.path.isdir(directory):
        return {}

    groups: dict[str, list[str]] = {}

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".out"):
            continue

        stem = fname[: -len(".out")]
        name = TRIAL_SUFFIX.sub("", stem)
        groups.setdefault(name, []).append(os.path.join(directory, fname))

    return dict(sorted(groups.items()))


def average_trials(
    trials: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray] | None:
    """Mean over trials on the matrix sizes present in every trial."""
    if not trials:
        return None

    sizes = set(trials[0][0].tolist())
    for s, _ in trials[1:]:
        sizes &= set(s.tolist())

    common = np.array(sorted(sizes), dtype=int)
    if common.size == 0:
        return None

    values = []
    for s, v in trials:
        values.append(v[np.searchsorted(s, common)])

    return {"sizes": common, "values": np.mean(values, axis=0)}


def load_time_data(files: list[str]) -> dict[str, np.ndarray] | None:
    """Reads ``n,time_seconds`` files and averages them."""
    trials = []

    for path in files:
        data = np.loadtxt(path, delimiter=",", ndmin=2)
        if data.size == 0:
            continue

        order = np.argsort(data[:, 0])
        sizes = data[order, 0].astype(int)
        times = data[order, 1]
        trials.append((sizes, times))

    return average_trials(trials)


def load_op_data(files: list[str]) -> dict[str, np.ndarray] | None:
    """Reads operation counters and averages the total op_num per size."""
    trials = []

    for path in files:
        data = np.loadtxt(path, delimiter=",", ndmin=2)
        if data.size == 0:
            continue

        if data.shape[1] >= 5:
            sizes = data[:, 0].astype(int)
            ops = data[:, 1:].sum(axis=1)
        else:
            sizes = np.arange(OP_START_N, OP_START_N + data.shape[0])
            ops = data.sum(axis=1)

        order = np.argsort(sizes)
        trials.append((sizes[order], ops[order]))

    return average_trials(trials)


def label_for(name: str) -> str:
    """``inverse_strassen`` -> ``Strassen``, ``inverse_default`` -> ``Domyślny``."""
    algo = name.partition("_")[2]
    labels = {
        "default": "Domyślny",
        "binet": "Binet",
        "strassen": "Strassen",
        "laderman": "Laderman",
    }
    return labels.get(algo, algo.capitalize())


# ──────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────
def downsample(x, y, max_points=400):
    """Samples the curve down for readability."""
    if len(x) <= max_points:
        return x, y
    step = max(1, len(x) // max_points)
    return x[::step], y[::step]


def format_number(v):
    """Readable tick labels for both tiny times and huge operation counts."""
    magnitude = abs(v)
    if magnitude >= 1e6:
        return f"{v / 1e6:.1f}M"
    if magnitude >= 1e3:
        return f"{v / 1e3:.0f}k"
    if magnitude >= 1 or magnitude == 0:
        return f"{v:.0f}"
    return f"{v:.4f}"


def format_large_numbers(ax, axis="y"):
    formatter = ticker.FuncFormatter(lambda v, _: format_number(v))
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def plot_metric(kind: str, prefix: str, ylabel: str, title: str, filename: str):
    """Draws the metric in a linear and a log-log panel for one algorithm family."""
    groups = discover_trials(kind)
    series = []

    for name, files in groups.items():
        if not name.startswith(prefix + "_"):
            continue
        data = load_time_data(files) if kind == "time" else load_op_data(files)
        if data is not None:
            series.append((name, data))

    if not series:
        print(f"  ! {filename}: brak danych w {os.path.join(STATS_DIR, kind)}")
        return

    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for idx, (name, data) in enumerate(series):
        color = colors[idx % len(colors)]
        label = label_for(name)

        sx, sy = downsample(data["sizes"], data["values"])
        axes[0].plot(sx, sy, color=color, label=label, linewidth=2, alpha=0.9)

        mask = (data["sizes"] > 0) & (data["values"] > 0)
        axes[1].plot(
            data["sizes"][mask],
            data["values"][mask],
            color=color,
            label=label,
            linewidth=2,
            alpha=0.9,
        )

    for ax in axes:
        ax.set_xlabel("Rozmiar macierzy (n)")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper left", framealpha=0.9)

    axes[0].set_title(f"{title} — średnia z prób")
    axes[0].set_xlim(left=0)
    format_large_numbers(axes[0])

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title(f"{title} (skala log-log) — średnia z prób")

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {filename}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Generowanie wykresów...\n")

    plot_metric(
        kind="time",
        prefix="inverse",
        ylabel="Czas wykonania [s]",
        title="Czas wykonania odwracania macierzy t(n)",
        filename="01_czas_odwracania.png",
    )

    plot_metric(
        kind="time",
        prefix="lu",
        ylabel="Czas wykonania [s]",
        title="Czas wykonania rozkładu LU t(n)",
        filename="02_czas_lu.png",
    )

    plot_metric(
        kind="op",
        prefix="inverse",
        ylabel="Całkowita liczba operacji (dod. + odejm. + mnoż. + dziel.)",
        title="Liczba operacji odwracania macierzy op_num(n)",
        filename="03_operacje_odwracania.png",
    )

    plot_metric(
        kind="op",
        prefix="lu",
        ylabel="Całkowita liczba operacji (dod. + odejm. + mnoż. + dziel.)",
        title="Liczba operacji rozkładu LU op_num(n)",
        filename="04_operacje_lu.png",
    )

    print(f"\nWszystkie wykresy zapisane w: {os.path.abspath(PLOTS_DIR)}")
