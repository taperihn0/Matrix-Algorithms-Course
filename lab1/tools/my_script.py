from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

path = Path(
    "/Users/kajetanserwecinski/Documents/Programming/Matrix-Algorithms-Course/lab1/stats/time"
)

# Wyniki MATLAB-a — plik rozdzielony tabulatorami
# Wyniki MATLAB-a
matlab = pd.read_csv(
    path / "timings.out",
    header=None,
    sep="\t",
    names=["matrix_width", "time_seconds"],
)

# Zamiana na liczby i usunięcie nagłówków zapisanych wśród danych
matlab["matrix_width"] = pd.to_numeric(
    matlab["matrix_width"], errors="coerce"
)
matlab["time_seconds"] = pd.to_numeric(
    matlab["time_seconds"], errors="coerce"
)

matlab = (
    matlab.dropna(subset=["matrix_width", "time_seconds"])
    .sort_values("matrix_width")
)

print(matlab)

# Przygotowanie wykresu
fig, ax = plt.subplots(figsize=(14, 8))

# MATLAB
ax.plot(
    matlab["matrix_width"],
    matlab["time_seconds"] * 1000,
    label="MATLAB",
    color="black",
    linestyle="-",
    marker="o",
    markersize=4,
    linewidth=2,
    zorder=10,
)

# Wyniki algorytmów C++
for name in ["binet", "laderman", "strassen"]:
    samples = []

    # Wczytanie pięciu próbek
    for i in range(5):
        file_path = path / f"{name}{i}.out"

        df = pd.read_csv(
            file_path,
            header=None,
            sep=",",
            names=["matrix_width", "time_seconds"],
        )

        df["matrix_width"] = pd.to_numeric(
            df["matrix_width"], errors="raise"
        )
        df["time_seconds"] = pd.to_numeric(
            df["time_seconds"], errors="raise"
        )

        samples.append(df)

    # Średnia z pięciu próbek dla każdej szerokości macierzy
    avg = (
        pd.concat(samples)
        .groupby("matrix_width", as_index=False)["time_seconds"]
        .mean()
        .sort_values("matrix_width")
    )

    # Zapis średnich czasów w sekundach
    avg.to_csv(
        path / f"{name}_avg.out",
        index=False,
        header=False,
        sep="\t",
    )

    # Wykres w milisekundach
    ax.plot(
        avg["matrix_width"],
        avg["time_seconds"] * 1000,
        label=name.capitalize(),
        linewidth=2,
    )

    print(f"{name}:\n{avg}\n")

# Opisy osi i tytuł
ax.set_xlabel("Szerokość macierzy", fontsize=13, labelpad=10)
ax.set_ylabel("Czas mnożenia macierzy [ms] — skala logarytmiczna", fontsize=13, labelpad=10)
ax.set_title("Porównanie czasu mnożenia macierzy", fontsize=15, pad=15)

# Skala logarytmiczna osi Y
ax.set_yscale("log")

# Podziałka osi X co 25
ax.xaxis.set_major_locator(MultipleLocator(25))
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

# Czytelność
ax.tick_params(axis="x", labelsize=9, labelrotation=45)
ax.tick_params(axis="y", labelsize=10)
ax.grid(True, which="major", linestyle="--", alpha=0.35)
ax.grid(True, which="minor", axis="y", linestyle=":", alpha=0.15)
ax.legend(fontsize=11, loc="best")

fig.tight_layout()

# Zapis wykresu
output_path = path / "comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"Wykres zapisano do: {output_path}")