"""
Skrypt do analizy i wizualizacji wydajności algorytmów mnożenia macierzy:
- Standardowy (naiwny / O(n^3))
- Binet (dziel i zwyciężaj / O(n^3))
- Strassen (O(n^(log2(7))) ≈ O(n^2.807))
- Laderman (O(n^(log3(23))) ≈ O(n^2.854))

Wczytuje dane ze statystyk:
- ../stats/time/*.out (czasy wykonania w ms)
- ../stats/op/*.out   (liczba operacji arytmetycznych: dodawanie, odejmowanie, mnożenie)
Generuje czytelne, wielkoformatowe wykresy porównawcze z liniami trendu,
pozwalające na empiryczne wyprowadzenie i weryfikację teoretycznej złożoności obliczeniowej.
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# Konfiguracja ścieżek i stałych
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = REPO_ROOT / "stats"
OP_DIR = STATS_DIR / "op"
TIME_DIR = STATS_DIR / "time"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Identyfikatory algorytmów w nazwach plików
ALGORITHMS = ["standart", "binet", "strassen", "laderman"]

ALG_NAMES = {
    "standart": "Standardowy",
    "binet": "Binet",
    "strassen": "Strassen",
    "laderman": "Laderman",
}

# Teoretyczne wykładniki złożoności obliczeniowej O(n^k)
THEORETICAL_EXPONENTS = {
    "standart": 3.0,
    "binet": 3.0,
    "strassen": np.log2(7.0),               # ≈ 2.80735
    "laderman": np.log(23.0) / np.log(3.0), # ≈ 2.85404
}

ALG_COLORS = {
    "standart": "#1f77b4",  # niebieski
    "binet": "#2ca02c",     # zielony
    "strassen": "#ff7f0e",  # pomarańczowy
    "laderman": "#d62728",  # czerwony
}

ALG_MARKERS = {
    "standart": "o",
    "binet": "s",
    "strassen": "^",
    "laderman": "D",
}


# ==============================================================================
# Funkcje ładowania danych do DataFrame
# ==============================================================================
def load_time_series(alg_name: str, time_dir: Path = TIME_DIR) -> pd.DataFrame:
    """
    Wczytuje dane czasowe dla danego algorytmu.
    Obsługuje format <rozmiar>,<czas_ms>. W przypadku obecności wielu rund
    (np. standart0.out, standart1.out, ...) uśrednia wyniki dla każdego rozmiaru.
    """
    pattern = str(time_dir / f"{alg_name}*.out")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Nie znaleziono plików czasowych dla {alg_name} w {time_dir}")

    dfs = []
    for f in files:
        # on_bad_lines='skip' pomija ewentualne urwane linie na końcu pliku
        df = pd.read_csv(f, header=None, names=["size", "time_ms"], on_bad_lines="skip")
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
        df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
        df = df.dropna().sort_values("size")
        df["size"] = df["size"].astype(int)
        dfs.append(df)

    if len(dfs) == 1:
        result_df = dfs[0].reset_index(drop=True)
    else:
        # Uśrednianie po rundach
        combined = pd.concat(dfs, axis=0)
        result_df = (
            combined.groupby("size", as_index=False)["time_ms"]
            .mean()
            .sort_values("size")
            .reset_index(drop=True)
        )

    return result_df


def load_op_series(alg_name: str, op_dir: Path = OP_DIR) -> pd.DataFrame:
    """
    Wczytuje liczbę operacji arytmetycznych dla danego algorytmu.
    Format pliku op: <add>,<sub>,<mul>,<divs>.
    Zgodnie z poleceniem: pomija liczbę dzieleń (nie występują w benchmarku).
    Wiersze odpowiadają kolejnym rozmiarom macierzy począwszy od n=2.
    """
    pattern = str(op_dir / f"{alg_name}*.out")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Nie znaleziono plików operacji dla {alg_name} w {op_dir}")

    dfs = []
    for f in files:
        # Wczytanie 4 kolumn (ostatnia to divs)
        df = pd.read_csv(
            f,
            header=None,
            names=["add", "sub", "mul", "divs"],
            on_bad_lines="skip",
        )
        for col in ["add", "sub", "mul"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["add", "sub", "mul"])
        # Pomijamy liczbę dzieleń!
        df = df[["add", "sub", "mul"]].copy()
        dfs.append(df)

    if len(dfs) == 1:
        op_df = dfs[0].reset_index(drop=True)
    else:
        # W przypadku wielu rund uśredniamy wiersz po wierszu
        min_len = min(len(d) for d in dfs)
        trimmed = [d.iloc[:min_len].reset_index(drop=True) for d in dfs]
        op_df = sum(trimmed) / len(trimmed)

    # Indeks 0 odpowiada rozmiarowi n=2 (początek pętli benchmarku w C++)
    op_df.insert(0, "size", np.arange(2, 2 + len(op_df), dtype=int))
    # Wyliczenie operacji łącznych (bez dzieleń)
    op_df["add_sub"] = op_df["add"] + op_df["sub"]
    op_df["total_ops"] = op_df["add"] + op_df["sub"] + op_df["mul"]

    return op_df


def load_all_data():
    """Wczytuje kompletne słowniki z danymi dla wszystkich algorytmów."""
    time_data = {}
    op_data = {}
    for name in ALGORITHMS:
        time_data[name] = load_time_series(name)
        op_data[name] = load_op_series(name)
    return time_data, op_data


# ==============================================================================
# Dopasowanie linii trendu i estymacja złożoności teoretycznej
# ==============================================================================
def fit_power_law(sizes: np.ndarray, values: np.ndarray, min_size: int = 10) -> dict:
    """
    Dopasowuje model potęgowy y = c * n^k w przestrzeni log-log:
    log10(y) = k * log10(n) + log10(c)
    
    Współczynnik kierunkowy k to empiryczny wykładnik złożoności O(n^k).
    Parametr min_size pozwala na odrzucenie małych rozmiarów n < 10,
    gdzie narzuty bazowe lub małe bloki zaburzają asymptotykę.
    """
    mask = (sizes >= min_size) & (values > 0)
    x_sub = sizes[mask]
    y_sub = values[mask]

    if len(x_sub) < 3:
        # Jeśli za mało punktów, dopasuj do wszystkich dostępnych dodatnich
        mask = values > 0
        x_sub = sizes[mask]
        y_sub = values[mask]

    log_x = np.log10(x_sub)
    log_y = np.log10(y_sub)

    # Regresja liniowa w skali log10
    slope, intercept = np.polyfit(log_x, log_y, 1)
    k = slope
    c = 10.0 ** intercept

    # Obliczenie współczynnika determinacji R^2
    y_pred_log = intercept + slope * log_x
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    ss_res = np.sum((log_y - y_pred_log) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    # Przewidywane wartości dla pełnego zakresu rozmiarów
    y_fitted_full = c * (sizes.astype(float) ** k)

    return {
        "k": k,
        "c": c,
        "r2": r2,
        "fitted_curve": y_fitted_full,
    }


# ==============================================================================
# Tworzenie dużych i czytelnych wykresów
# ==============================================================================
def setup_plot_style():
    """Ustawia estetyczny, duży i czytelny styl wykresów."""
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 17,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.title_fontsize": 12,
        "figure.titlesize": 19,
        "lines.linewidth": 2.2,
        "grid.linestyle": "--",
        "grid.alpha": 0.5,
    })


def plot_time_comparison(time_data: dict, plots_dir: Path):
    """
    Wykres 1: Porównanie czasu wykonania algorytmów (skala liniowa oraz log-log z liniami trendu).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    fits = {}
    for name in ALGORITHMS:
        df = time_data[name]
        label = ALG_NAMES[name]
        color = ALG_COLORS[name]
        marker = ALG_MARKERS[name]
        theo_k = THEORETICAL_EXPONENTS[name]

        fit = fit_power_law(df["size"].to_numpy(), df["time_ms"].to_numpy(), min_size=15)
        fits[name] = fit

        # Rysowanie punktów / linii danych (z krokiem markera dla czytelności)
        markevery = max(1, len(df) // 25)
        ax1.plot(
            df["size"],
            df["time_ms"],
            color=color,
            marker=marker,
            markevery=markevery,
            markersize=6,
            alpha=0.85,
            label=f"{label} (dane)",
        )
        # Linia trendu w skali liniowej
        ax1.plot(
            df["size"],
            fit["fitted_curve"],
            color=color,
            linestyle="--",
            alpha=0.9,
            label=f"{label}: trend ~ O(n^{fit['k']:.2f}) [R²={fit['r2']:.4f}]",
        )

        # Skala log-log
        ax2.loglog(
            df["size"],
            df["time_ms"],
            color=color,
            marker=marker,
            markevery=markevery,
            markersize=6,
            alpha=0.85,
            label=f"{label} (dane)",
        )
        ax2.loglog(
            df["size"],
            fit["fitted_curve"],
            color=color,
            linestyle="--",
            alpha=0.9,
            label=f"{label}: nachylenie k={fit['k']:.2f} (teoria: {theo_k:.2f})",
        )

    # Konfiguracja osi lewego wykresu (skala liniowa)
    ax1.set_title("Czas wykonania w skali liniowej z liniami trendu", pad=12, fontweight="bold")
    ax1.set_xlabel("Rozmiar macierzy n (wymiar macierzy n × n)")
    ax1.set_ylabel("Czas wykonania T(n) [ms]")
    ax1.grid(True)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    # Konfiguracja osi prawego wykresu (skala log-log)
    ax2.set_title("Czas wykonania w skali log-log (nachylenie = wykładnik O(n^k))", pad=12, fontweight="bold")
    ax2.set_xlabel("Rozmiar macierzy n [skala log₁₀]")
    ax2.set_ylabel("Czas wykonania T(n) [ms] [skala log₁₀]")
    ax2.grid(True, which="both")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.suptitle("Porównanie czasu wykonania algorytmów mnożenia macierzy", fontsize=19, y=0.98, fontweight="bold")
    plt.tight_layout()
    output_path = plots_dir / "01_porownanie_czasu_wykonania.png"
    plt.savefig(output_path, dpi=300)
    print(f"[ZAPISANO] Wykres czasu wykonania: {output_path}")
    return fits


def plot_total_operations(op_data: dict, plots_dir: Path):
    """
    Wykres 2: Porównanie całkowitej liczby operacji arytmetycznych (dodawanie + odejmowanie + mnożenie).
    Bez dzieleń. Skala liniowa i log-log z liniami trendu.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    fits = {}
    for name in ALGORITHMS:
        df = op_data[name]
        label = ALG_NAMES[name]
        color = ALG_COLORS[name]
        marker = ALG_MARKERS[name]
        theo_k = THEORETICAL_EXPONENTS[name]

        fit = fit_power_law(df["size"].to_numpy(), df["total_ops"].to_numpy(), min_size=10)
        fits[name] = fit

        markevery = max(1, len(df) // 25)
        ax1.plot(
            df["size"],
            df["total_ops"],
            color=color,
            marker=marker,
            markevery=markevery,
            markersize=6,
            alpha=0.85,
            label=f"{label} (dane)",
        )
        ax1.plot(
            df["size"],
            fit["fitted_curve"],
            color=color,
            linestyle="--",
            alpha=0.9,
            label=f"{label}: trend ~ O(n^{fit['k']:.3f}) [R²={fit['r2']:.4f}]",
        )

        ax2.loglog(
            df["size"],
            df["total_ops"],
            color=color,
            marker=marker,
            markevery=markevery,
            markersize=6,
            alpha=0.85,
            label=f"{label} (dane)",
        )
        ax2.loglog(
            df["size"],
            fit["fitted_curve"],
            color=color,
            linestyle="--",
            alpha=0.9,
            label=f"{label}: k={fit['k']:.3f} (teoria: {theo_k:.3f})",
        )

    ax1.set_title("Liczba operacji arytmetycznych (skala liniowa)", pad=12, fontweight="bold")
    ax1.set_xlabel("Rozmiar macierzy n (wymiar macierzy n × n)")
    ax1.set_ylabel("Całkowita liczba operacji (add + sub + mul)")
    ax1.grid(True)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    ax2.set_title("Liczba operacji w skali log-log (nachylenie prostej = złożoność)", pad=12, fontweight="bold")
    ax2.set_xlabel("Rozmiar macierzy n [skala log₁₀]")
    ax2.set_ylabel("Całkowita liczba operacji [skala log₁₀]")
    ax2.grid(True, which="both")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.suptitle(
        "Porównanie całkowitej liczby operacji arytmetycznych (z pominięciem dzieleń)",
        fontsize=19,
        y=0.98,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path = plots_dir / "02_porownanie_calkowitej_liczby_operacji.png"
    plt.savefig(output_path, dpi=300)
    print(f"[ZAPISANO] Wykres liczby operacji: {output_path}")
    return fits


def plot_multiplications(op_data: dict, plots_dir: Path):
    """
    Wykres 3: Porównanie liczby operacji mnożenia (mul).
    Pozwala uwidocznić kluczową redukcję mnożeń algorytmów Strassena (7 zamiast 8)
    oraz Ladermana (23 zamiast 27 dla bloków 3x3).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    fits = {}
    for name in ALGORITHMS:
        df = op_data[name]
        label = ALG_NAMES[name]
        color = ALG_COLORS[name]
        marker = ALG_MARKERS[name]
        theo_k = THEORETICAL_EXPONENTS[name]

        fit = fit_power_law(df["size"].to_numpy(), df["mul"].to_numpy(), min_size=10)
        fits[name] = fit

        markevery = max(1, len(df) // 25)
        ax1.plot(
            df["size"],
            df["mul"],
            color=color,
            marker=marker,
            markevery=markevery,
            markersize=6,
            alpha=0.85,
            label=f"{label} (dane)",
        )
        ax1.plot(
            df["size"],
            fit["fitted_curve"],
            color=color,
            linestyle="--",
            alpha=0.9,
            label=f"{label}: trend ~ O(n^{fit['k']:.3f})",
        )

        ax2.loglog(
            df["size"],
            df["mul"],
            color=color,
            marker=marker,
            markevery=markevery,
            markersize=6,
            alpha=0.85,
            label=f"{label} (dane)",
        )
        ax2.loglog(
            df["size"],
            fit["fitted_curve"],
            color=color,
            linestyle="--",
            alpha=0.9,
            label=f"{label}: k={fit['k']:.3f} (teoria: {theo_k:.3f})",
        )

    ax1.set_title("Liczba operacji mnożenia (skala liniowa)", pad=12, fontweight="bold")
    ax1.set_xlabel("Rozmiar macierzy n (wymiar macierzy n × n)")
    ax1.set_ylabel("Liczba operacji mnożenia (mul)")
    ax1.grid(True)
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)

    ax2.set_title("Liczba mnożeń w skali log-log (redukcja Strassena i Ladermana)", pad=12, fontweight="bold")
    ax2.set_xlabel("Rozmiar macierzy n [skala log₁₀]")
    ax2.set_ylabel("Liczba operacji mnożenia [skala log₁₀]")
    ax2.grid(True, which="both")
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.suptitle("Porównanie liczby operacji mnożenia w algorytmach", fontsize=19, y=0.98, fontweight="bold")
    plt.tight_layout()
    output_path = plots_dir / "03_porownanie_liczby_mnozen.png"
    plt.savefig(output_path, dpi=300)
    print(f"[ZAPISANO] Wykres liczby mnożeń: {output_path}")
    return fits


def plot_operations_breakdown(op_data: dict, plots_dir: Path):
    """
    Wykres 4: Szczegółowa dekompozycja operacji dla każdego algorytmu (siatka 2x2).
    Prezentuje: dodawanie, odejmowanie, mnożenie oraz sumę (bez dzieleń).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    axes_flat = axes.flatten()

    for ax, name in zip(axes_flat, ALGORITHMS):
        df = op_data[name]
        label = ALG_NAMES[name]
        markevery = max(1, len(df) // 20)

        # Dodawanie
        ax.plot(
            df["size"],
            df["add"],
            color="#2ca02c",
            marker="o",
            markevery=markevery,
            markersize=5,
            label="Dodawanie (add)",
        )
        # Odejmowanie - rysowane tylko jeśli występuje (> 0)
        if df["sub"].sum() > 0:
            ax.plot(
                df["size"],
                df["sub"],
                color="#e377c2",
                marker="s",
                markevery=markevery,
                markersize=5,
                label="Odejmowanie (sub)",
            )
        # Mnożenie
        ax.plot(
            df["size"],
            df["mul"],
            color="#ff7f0e",
            marker="^",
            markevery=markevery,
            markersize=5,
            label="Mnożenie (mul)",
        )
        # Całkowita suma operacji
        ax.plot(
            df["size"],
            df["total_ops"],
            color="#1f77b4",
            linestyle="--",
            label="Suma operacji (bez dzieleń)",
        )

        theo = THEORETICAL_EXPONENTS[name]
        ax.set_title(f"Model: {label} (złożoność teoretyczna: O(n^{theo:.3f}))", pad=10, fontweight="bold")
        ax.set_xlabel("Rozmiar macierzy n (n × n)")
        ax.set_ylabel("Liczba operacji")
        ax.grid(True)
        ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    fig.suptitle(
        "Dekompozycja typów operacji arytmetycznych dla poszczególnych algorytmów (bez dzieleń)",
        fontsize=18,
        y=0.99,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path = plots_dir / "04_dekompozycja_operacji_na_algorytm.png"
    plt.savefig(output_path, dpi=300)
    print(f"[ZAPISANO] Wykres dekompozycji operacji: {output_path}")


def plot_complexity_summary(time_fits: dict, op_fits: dict, plots_dir: Path):
    """
    Wykres 5: Podsumowanie wykładników złożoności O(n^k):
    Teoretyczna vs Empiryczna z operacji vs Empiryczna z pomiarów czasu.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    names = [ALG_NAMES[a] for a in ALGORITHMS]
    theo_vals = [THEORETICAL_EXPONENTS[a] for a in ALGORITHMS]
    op_vals = [op_fits[a]["k"] for a in ALGORITHMS]
    time_vals = [time_fits[a]["k"] for a in ALGORITHMS]

    x = np.arange(len(names))
    width = 0.25

    rects1 = ax.bar(x - width, theo_vals, width, label="Złożoność teoretyczna k", color="#1f77b4", alpha=0.85)
    rects2 = ax.bar(x, op_vals, width, label="Złożoność empiryczna (operacje) k", color="#2ca02c", alpha=0.85)
    rects3 = ax.bar(x + width, time_vals, width, label="Złożoność empiryczna (czas) k", color="#ff7f0e", alpha=0.85)

    ax.set_title("Porównanie wykładników złożoności obliczeniowej O(n^k)", pad=14, fontweight="bold")
    ax.set_ylabel("Wartość wykładnika potęgowego k")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight="bold")
    ax.set_ylim(0, 3.8)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    # Etykiety wartości nad słupkami
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    output_path = plots_dir / "05_porownanie_zlozonosci_teoretycznej_i_empirycznej.png"
    plt.savefig(output_path, dpi=300)
    print(f"[ZAPISANO] Wykres podsumowania złożoności: {output_path}")


def plot_relative_efficiency(time_data: dict, op_data: dict, plots_dir: Path):
    """
    Wykres 6: Względna efektywność algorytmów w odniesieniu do algorytmu standardowego.
    Lewy panel: Stosunek liczby operacji (Ops_alg / Ops_standard).
    Prawy panel: Stosunek czasu wykonania (Time_alg / Time_standard).
    Pokazuje punkty przecięcia (crossover points) oraz rzeczywisty zysk.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    base_op = op_data["standart"].set_index("size")["total_ops"]
    base_time = time_data["standart"].set_index("size")["time_ms"]

    for name in ["binet", "strassen", "laderman"]:
        label = ALG_NAMES[name]
        color = ALG_COLORS[name]
        marker = ALG_MARKERS[name]

        # Względna liczba operacji
        op_s = op_data[name].set_index("size")["total_ops"]
        common_sizes_op = op_s.index.intersection(base_op.index)
        ratio_op = op_s.loc[common_sizes_op] / base_op.loc[common_sizes_op]

        markevery_op = max(1, len(common_sizes_op) // 25)
        ax1.plot(
            common_sizes_op,
            ratio_op,
            color=color,
            marker=marker,
            markevery=markevery_op,
            markersize=6,
            label=f"{label} / Standard",
        )

        # Względny czas wykonania
        time_s = time_data[name].set_index("size")["time_ms"]
        common_sizes_time = time_s.index.intersection(base_time.index)
        ratio_time = time_s.loc[common_sizes_time] / base_time.loc[common_sizes_time]

        markevery_time = max(1, len(common_sizes_time) // 25)
        ax2.plot(
            common_sizes_time,
            ratio_time,
            color=color,
            marker=marker,
            markevery=markevery_time,
            markersize=6,
            label=f"{label} / Standard",
        )

    # Linia odniesienia 1.0 (poziom algorytmu standardowego)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Standard = 1.0")
    ax1.set_title("Stosunek liczby operacji do algorytmu Standardowego", pad=12, fontweight="bold")
    ax1.set_xlabel("Rozmiar macierzy n (wymiary n × n)")
    ax1.set_ylabel("Współczynnik (Ops_alg / Ops_standard)")
    ax1.grid(True)
    ax1.legend(loc="best", frameon=True, framealpha=0.9)

    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Standard = 1.0")
    ax2.set_title("Stosunek czasu wykonania do algorytmu Standardowego", pad=12, fontweight="bold")
    ax2.set_xlabel("Rozmiar macierzy n (wymiary n × n)")
    ax2.set_ylabel("Współczynnik (Time_alg / Time_standard)")
    ax2.grid(True)
    ax2.legend(loc="best", frameon=True, framealpha=0.9)

    fig.suptitle(
        "Efektywność względna algorytmów wobec metody Standardowej (wartości < 1.0 oznaczają przewagę)",
        fontsize=18,
        y=0.99,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path = plots_dir / "06_stosunek_wzgledem_standardowego.png"
    plt.savefig(output_path, dpi=300)
    print(f"[ZAPISANO] Wykres efektywności względnej: {output_path}")


# ==============================================================================
# Wypisanie podsumowania w konsoli
# ==============================================================================
def print_summary_table(time_fits: dict, op_fits: dict, mul_fits: dict):
    """Wyświetla estetyczną tabelę podsumowującą złożoność teoretyczną i empiryczną."""
    header = (
        f"{'Algorytm':<14} | "
        f"{'Teoria O(n^k)':<13} | "
        f"{'Empiryczne (op) k':<18} | "
        f"{'R² (op)':<8} | "
        f"{'Empiryczne (mul) k':<19} | "
        f"{'Empiryczne (czas) k':<20} | "
        f"{'R² (czas)':<8}"
    )
    separator = "-" * len(header)
    print("\n" + "=" * len(header))
    print("PODSUMOWANIE ZŁOŻONOŚCI OBLICZENIOWEJ MODELI (WYPROWADZENIE TEORETYCZNE VS EMPIRYCZNE)")
    print("=" * len(header))
    print(header)
    print(separator)

    for name in ALGORITHMS:
        display = ALG_NAMES[name]
        theo = THEORETICAL_EXPONENTS[name]
        op_fit = op_fits[name]
        mul_fit = mul_fits[name]
        t_fit = time_fits[name]

        print(
            f"{display:<14} | "
            f"O(n^{theo:<6.3f}) | "
            f"O(n^{op_fit['k']:<6.3f})       | "
            f"{op_fit['r2']:<8.4f} | "
            f"O(n^{mul_fit['k']:<6.3f})        | "
            f"O(n^{t_fit['k']:<6.3f})         | "
            f"{t_fit['r2']:<8.4f}"
        )
    print("=" * len(header) + "\n")


# ==============================================================================
# Główna funkcja wykonawcza
# ==============================================================================
def main():
    print("Rozpoczęcie ładowania danych ze statystyk /stats/time oraz /stats/op...")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    time_data, op_data = load_all_data()
    print("Wszystkie dane zostały pomyślnie załadowane do DataFrame'ów.")
    for name in ALGORITHMS:
        print(f"  - {ALG_NAMES[name]}: op={len(op_data[name])} próbek (n={op_data[name]['size'].min()}..{op_data[name]['size'].max()}), "
              f"time={len(time_data[name])} próbek (n={time_data[name]['size'].min()}..{time_data[name]['size'].max()})")

    print("\nGenerowanie wykresów o wysokiej czytelności i dopasowywanie linii trendów...")
    time_fits = plot_time_comparison(time_data, PLOTS_DIR)
    op_fits = plot_total_operations(op_data, PLOTS_DIR)
    mul_fits = plot_multiplications(op_data, PLOTS_DIR)
    plot_operations_breakdown(op_data, PLOTS_DIR)
    plot_complexity_summary(time_fits, op_fits, PLOTS_DIR)
    plot_relative_efficiency(time_data, op_data, PLOTS_DIR)

    print_summary_table(time_fits, op_fits, mul_fits)
    print(f"Wszystkie wykresy zapisano w katalogu: {PLOTS_DIR}")

    # Wyświetlenie interaktywne jeśli sesja wspiera GUI
    try:
        import matplotlib
        if matplotlib.get_backend().lower() not in ["agg", "template"]:
            plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
