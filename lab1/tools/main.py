"""
Generowanie wykresów zależności rozmiaru macierzy od liczby operacji i czasu
dla algorytmów: Binet, Strassen, Laderman.

Dane z folderów stats/op/ i stats/time/.
Każdy algorytm ma 5 prób (pliki 0–4), wyciągamy średnią.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────
# Konfiguracja
# ──────────────────────────────────────────────
STATS_DIR = os.path.join(os.path.dirname(__file__), "..", "stats")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

ALGORITHMS = ["binet", "strassen", "laderman"]
ALGO_LABELS = {"binet": "Binet", "strassen": "Strassen", "laderman": "Laderman"}
ALGO_COLORS = {"binet": "#2196F3", "strassen": "#E91E63", "laderman": "#4CAF50"}
ALGO_MARKERS = {"binet": "o", "strassen": "s", "laderman": "^"}
N_TRIALS = 5

# Styl wykresów
plt.rcParams.update({
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
})


# ──────────────────────────────────────────────
# Wczytywanie danych
# ──────────────────────────────────────────────
def load_op_data(algo: str) -> dict:
    """
    Wczytuje dane operacji dla danego algorytmu ze wszystkich prób.
    Zwraca dict z kluczami: sizes, add_ops, mul_ops, total_ops
    (każda wartość to średnia z prób).
    
    Format pliku op: add_ops,add_trivial,mul_ops,mul_trivial
    Linia i-ta odpowiada rozmiarowi macierzy i+2.
    """
    all_add = []
    all_mul = []

    for trial in range(N_TRIALS):
        fpath = os.path.join(STATS_DIR, "op", f"{algo}{trial}.out")
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            continue
        data = np.loadtxt(fpath, delimiter=",")
        add_ops = data[:, 0] + data[:, 1]   # add_ops + add_trivial
        mul_ops = data[:, 2] + data[:, 3]   # mul_ops + mul_trivial
        all_add.append(add_ops)
        all_mul.append(mul_ops)

    if not all_add:
        return None

    # Upewnij się, że wymiary się zgadzają
    min_len = min(len(a) for a in all_add)
    all_add = np.array([a[:min_len] for a in all_add])
    all_mul = np.array([m[:min_len] for m in all_mul])

    avg_add = np.mean(all_add, axis=0)
    avg_mul = np.mean(all_mul, axis=0)
    sizes = np.arange(2, 2 + min_len)

    return {
        "sizes": sizes,
        "add_ops": avg_add,
        "mul_ops": avg_mul,
        "total_ops": avg_add + avg_mul,
    }


def load_time_data(algo: str) -> dict:
    """
    Wczytuje dane czasowe dla danego algorytmu ze wszystkich prób.
    Zwraca dict z kluczami: sizes, time (średnia z prób).
    
    Format pliku time: matrix_size,time_seconds
    """
    all_times = []
    sizes_ref = None

    for trial in range(N_TRIALS):
        fpath = os.path.join(STATS_DIR, "time", f"{algo}{trial}.out")
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            continue
        data = np.loadtxt(fpath, delimiter=",")
        sizes = data[:, 0].astype(int)
        times = data[:, 1]

        if sizes_ref is None:
            sizes_ref = sizes
        all_times.append(times)

    if not all_times:
        return None

    min_len = min(len(t) for t in all_times)
    all_times = np.array([t[:min_len] for t in all_times])
    avg_time = np.mean(all_times, axis=0)

    return {
        "sizes": sizes_ref[:min_len],
        "time": avg_time,
    }


# ──────────────────────────────────────────────
# Funkcje pomocnicze dla wykresów
# ──────────────────────────────────────────────
def downsample(x, y, max_points=200):
    """Próbkowanie danych dla czytelności (co N-ty punkt)."""
    if len(x) <= max_points:
        return x, y
    step = max(1, len(x) // max_points)
    return x[::step], y[::step]


def format_large_numbers(ax, axis="y"):
    """Formatowanie dużych liczb na osiach."""
    if axis == "y":
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6
            else (f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}")
        ))
    else:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6
            else (f"{x/1e3:.0f}k" if x >= 1e3 else f"{x:.0f}")
        ))


# ──────────────────────────────────────────────
# Generowanie wykresów
# ──────────────────────────────────────────────
def plot_ops_comparison():
    """
    Wykres 1: Porównanie całkowitej liczby operacji (add + mul)
    dla wszystkich algorytmów.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    for algo in ALGORITHMS:
        data = load_op_data(algo)
        if data is None:
            continue
        sx, sy = downsample(data["sizes"], data["total_ops"])
        ax.plot(sx, sy,
                color=ALGO_COLORS[algo],
                label=f"{ALGO_LABELS[algo]} (średnia z {N_TRIALS} prób)",
                linewidth=2, alpha=0.9)

    ax.set_xlabel("Rozmiar macierzy (n)")
    ax.set_ylabel("Całkowita liczba operacji (dodawania + mnożenia)")
    ax.set_title("Porównanie całkowitej liczby operacji — średnia z prób")
    ax.legend(loc="upper left", framealpha=0.9)
    format_large_numbers(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "01_calkowita_liczba_operacji.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 01_calkowita_liczba_operacji.png")


def plot_add_mul_separate():
    """
    Wykres 2: Oddzielne porównanie dodawań i mnożeń.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Dodawania
    ax = axes[0]
    for algo in ALGORITHMS:
        data = load_op_data(algo)
        if data is None:
            continue
        sx, sy = downsample(data["sizes"], data["add_ops"])
        ax.plot(sx, sy,
                color=ALGO_COLORS[algo],
                label=ALGO_LABELS[algo],
                linewidth=2, alpha=0.9)
    ax.set_xlabel("Rozmiar macierzy (n)")
    ax.set_ylabel("Liczba dodawań / odejmowań")
    ax.set_title("Porównanie liczby dodawań / odejmowań")
    ax.legend(loc="upper left", framealpha=0.9)
    format_large_numbers(ax)

    # Mnożenia
    ax = axes[1]
    for algo in ALGORITHMS:
        data = load_op_data(algo)
        if data is None:
            continue
        sx, sy = downsample(data["sizes"], data["mul_ops"])
        ax.plot(sx, sy,
                color=ALGO_COLORS[algo],
                label=ALGO_LABELS[algo],
                linewidth=2, alpha=0.9)
    ax.set_xlabel("Rozmiar macierzy (n)")
    ax.set_ylabel("Liczba mnożeń")
    ax.set_title("Porównanie liczby mnożeń")
    ax.legend(loc="upper left", framealpha=0.9)
    format_large_numbers(ax)

    fig.suptitle("Dekompozycja operacji — średnia z prób", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "02_dodawania_vs_mnozenia.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 02_dodawania_vs_mnozenia.png")


def plot_ops_per_algorithm():
    """
    Wykres 3: Dla każdego algorytmu osobny wykres z dekompozycją
    na dodawania i mnożenia (stacked area).
    """
    algos_with_data = [(a, load_op_data(a)) for a in ALGORITHMS]
    algos_with_data = [(a, d) for a, d in algos_with_data if d is not None]

    n = len(algos_with_data)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, (algo, data) in zip(axes, algos_with_data):
        sx_add, sy_add = downsample(data["sizes"], data["add_ops"])
        sx_mul, sy_mul = downsample(data["sizes"], data["mul_ops"])

        ax.fill_between(sx_add, 0, sy_add,
                        alpha=0.35, color="#FF9800", label="Dodawania / odejmowania")
        ax.fill_between(sx_mul, 0, sy_mul,
                        alpha=0.35, color="#9C27B0", label="Mnożenia")
        ax.plot(sx_add, sy_add, color="#FF9800", linewidth=1.5)
        ax.plot(sx_mul, sy_mul, color="#9C27B0", linewidth=1.5)

        ax.set_xlabel("Rozmiar macierzy (n)")
        ax.set_ylabel("Liczba operacji")
        ax.set_title(f"{ALGO_LABELS[algo]}")
        ax.legend(loc="upper left", framealpha=0.9)
        format_large_numbers(ax)

    fig.suptitle("Dekompozycja operacji na algorytm — średnia z prób", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "03_dekompozycja_operacji_na_algorytm.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 03_dekompozycja_operacji_na_algorytm.png")


def plot_time_comparison():
    """
    Wykres 4: Porównanie czasu wykonania dla wszystkich algorytmów.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    for algo in ALGORITHMS:
        data = load_time_data(algo)
        if data is None:
            continue
        sx, sy = downsample(data["sizes"], data["time"])
        ax.plot(sx, sy,
                color=ALGO_COLORS[algo],
                label=f"{ALGO_LABELS[algo]} (średnia z {N_TRIALS} prób)",
                linewidth=2, alpha=0.9)

    ax.set_xlabel("Rozmiar macierzy (n)")
    ax.set_ylabel("Czas wykonania [s]")
    ax.set_title("Porównanie czasu wykonania — średnia z prób")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "04_czas_wykonania.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 04_czas_wykonania.png")


def plot_time_log_scale():
    """
    Wykres 5: Czas wykonania w skali logarytmicznej (log-log).
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    for algo in ALGORITHMS:
        data = load_time_data(algo)
        if data is None:
            continue
        sx, sy = downsample(data["sizes"], data["time"])
        # Filtruj zerowe wartości dla log scale
        mask = sy > 0
        ax.plot(sx[mask], sy[mask],
                color=ALGO_COLORS[algo],
                label=ALGO_LABELS[algo],
                linewidth=2, alpha=0.9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rozmiar macierzy (n)")
    ax.set_ylabel("Czas wykonania [s]")
    ax.set_title("Porównanie czasu wykonania (skala log-log) — średnia z prób")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "05_czas_log_scale.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 05_czas_log_scale.png")


def plot_ops_log_scale():
    """
    Wykres 6: Całkowita liczba operacji w skali log-log.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    for algo in ALGORITHMS:
        data = load_op_data(algo)
        if data is None:
            continue
        sx, sy = downsample(data["sizes"], data["total_ops"])
        mask = sy > 0
        ax.plot(sx[mask], sy[mask],
                color=ALGO_COLORS[algo],
                label=ALGO_LABELS[algo],
                linewidth=2, alpha=0.9)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rozmiar macierzy (n)")
    ax.set_ylabel("Całkowita liczba operacji")
    ax.set_title("Porównanie liczby operacji (skala log-log) — średnia z prób")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "06_operacje_log_scale.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 06_operacje_log_scale.png")


def plot_ratio_vs_binet():
    """
    Wykres 7: Stosunek liczby operacji Strassena i Ladermana
    względem Bineta.
    """
    binet_data = load_op_data("binet")
    if binet_data is None:
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    for algo in ["strassen", "laderman"]:
        data = load_op_data(algo)
        if data is None:
            continue
        min_len = min(len(data["total_ops"]), len(binet_data["total_ops"]))
        ratio = data["total_ops"][:min_len] / binet_data["total_ops"][:min_len]
        sizes = data["sizes"][:min_len]
        sx, sy = downsample(sizes, ratio)
        ax.plot(sx, sy,
                color=ALGO_COLORS[algo],
                label=f"{ALGO_LABELS[algo]} / Binet",
                linewidth=2, alpha=0.9)

    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="Binet (ref = 1.0)")
    ax.set_xlabel("Rozmiar macierzy (n)")
    ax.set_ylabel("Stosunek operacji (algorytm / Binet)")
    ax.set_title("Stosunek całkowitej liczby operacji względem algorytmu Bineta")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "07_stosunek_operacji_vs_binet.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 07_stosunek_operacji_vs_binet.png")


def plot_individual_trials_ops():
    """
    Wykres 8: Dla każdego algorytmu — poszczególne próby + średnia
    (całkowita liczba operacji). Średnia wyróżniona grubszą linią.
    """
    algos_with_data = []
    for algo in ALGORITHMS:
        avg_data = load_op_data(algo)
        if avg_data is None:
            continue
        algos_with_data.append((algo, avg_data))

    n = len(algos_with_data)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, (algo, avg_data) in zip(axes, algos_with_data):
        # Poszczególne próby (cienkie, półprzezroczyste)
        for trial in range(N_TRIALS):
            fpath = os.path.join(STATS_DIR, "op", f"{algo}{trial}.out")
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                continue
            raw = np.loadtxt(fpath, delimiter=",")
            total = raw[:, 0] + raw[:, 1] + raw[:, 2] + raw[:, 3]
            sizes = np.arange(2, 2 + len(total))
            sx, sy = downsample(sizes, total)
            ax.plot(sx, sy,
                    color=ALGO_COLORS[algo], alpha=0.2, linewidth=0.8,
                    label=f"Próba {trial}" if trial == 0 else None)

        # Średnia (gruba linia)
        sx, sy = downsample(avg_data["sizes"], avg_data["total_ops"])
        ax.plot(sx, sy,
                color=ALGO_COLORS[algo], linewidth=3, alpha=1.0,
                label="Średnia", zorder=5)

        ax.set_xlabel("Rozmiar macierzy (n)")
        ax.set_ylabel("Całkowita liczba operacji")
        ax.set_title(f"{ALGO_LABELS[algo]}")
        ax.legend(loc="upper left", framealpha=0.9)
        format_large_numbers(ax)

    fig.suptitle("Próby vs. średnia — całkowita liczba operacji", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "08_proby_vs_srednia_operacje.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 08_proby_vs_srednia_operacje.png")


def plot_individual_trials_time():
    """
    Wykres 9: Dla każdego algorytmu — poszczególne próby + średnia
    (czas wykonania). Średnia wyróżniona grubszą linią.
    """
    algos_with_data = []
    for algo in ALGORITHMS:
        avg_data = load_time_data(algo)
        if avg_data is None:
            continue
        algos_with_data.append((algo, avg_data))

    n = len(algos_with_data)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, (algo, avg_data) in zip(axes, algos_with_data):
        for trial in range(N_TRIALS):
            fpath = os.path.join(STATS_DIR, "time", f"{algo}{trial}.out")
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                continue
            raw = np.loadtxt(fpath, delimiter=",")
            sizes = raw[:, 0].astype(int)
            times = raw[:, 1]
            sx, sy = downsample(sizes, times)
            ax.plot(sx, sy,
                    color=ALGO_COLORS[algo], alpha=0.2, linewidth=0.8,
                    label=f"Próba {trial}" if trial == 0 else None)

        sx, sy = downsample(avg_data["sizes"], avg_data["time"])
        ax.plot(sx, sy,
                color=ALGO_COLORS[algo], linewidth=3, alpha=1.0,
                label="Średnia", zorder=5)

        ax.set_xlabel("Rozmiar macierzy (n)")
        ax.set_ylabel("Czas wykonania [s]")
        ax.set_title(f"{ALGO_LABELS[algo]}")
        ax.legend(loc="upper left", framealpha=0.9)

    fig.suptitle("Próby vs. średnia — czas wykonania", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "09_proby_vs_srednia_czas.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ 09_proby_vs_srednia_czas.png")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Generowanie wykresów...\n")

    plot_ops_comparison()
    plot_add_mul_separate()
    plot_ops_per_algorithm()
    plot_time_comparison()
    plot_time_log_scale()
    plot_ops_log_scale()
    plot_ratio_vs_binet()
    plot_individual_trials_ops()
    plot_individual_trials_time()

    print(f"\nWszystkie wykresy zapisane w: {os.path.abspath(PLOTS_DIR)}")
