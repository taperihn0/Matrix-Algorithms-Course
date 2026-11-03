from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

path = Path(
    "/Users/kajetanserwecinski/Documents/Programming/Matrix-Algorithms-Course/lab1/stats/op"
)

algorithms = ["binet", "laderman", "strassen"]

columns = [
    "additions",
    "subtractions",
    "multiplications",
    "divisions",
]

results = {}

# Wczytanie pięciu próbek i obliczenie średnich
for name in algorithms:
    samples = []

    for i in range(5):
        file_path = path / f"{name}{i}.out"

        df = pd.read_csv(
            file_path,
            header=None,
            sep=",",
            names=columns,
            skipinitialspace=True,
        )

        # Pierwszy wiersz odpowiada macierzy 2x2
        df["matrix_width"] = range(2, len(df) + 2)
        samples.append(df)

    # Średnia liczba operacji dla każdej szerokości macierzy
    avg = (
        pd.concat(samples)
        .groupby("matrix_width", as_index=False)[columns]
        .mean()
        .sort_values("matrix_width")
    )

    # Łączna liczba dodawań i odejmowań
    avg["add_sub_operations"] = (
        avg["additions"] + avg["subtractions"]
    )

    results[name] = avg

    # Zapis średnich liczników operacji
    avg[columns].to_csv(
        path / f"{name}_avg.out",
        index=False,
        header=False,
        sep=",",
    )

    print(f"{name}:\n{avg}\n")


def plot_comparison(column, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(13, 7))

    for name, avg in results.items():
        ax.plot(
            avg["matrix_width"],
            avg[column],
            label=name.capitalize(),
            linewidth=2,
        )

    ax.set_xlabel("Szerokość macierzy", fontsize=13, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=15, pad=15)

    # Podziałka osi X co 25
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

    # Ograniczenie osi X do rzeczywistego zakresu danych
    min_width = min(avg["matrix_width"].min() for avg in results.values())
    max_width = max(avg["matrix_width"].max() for avg in results.values())
    ax.set_xlim(min_width, max_width)

    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=11)
    ax.margins(y=0.05)

    fig.tight_layout()

    output_path = path / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Zapisano wykres: {output_path}")


# Wykres 1: dodawania i odejmowania
plot_comparison(
    column="add_sub_operations",
    ylabel="Liczba dodawań i odejmowań",
    title="Porównanie liczby dodawań i odejmowań",
    filename="comparison_add_sub.png",
)

# Wykres 2: mnożenia
plot_comparison(
    column="multiplications",
    ylabel="Liczba mnożeń",
    title="Porównanie liczby mnożeń",
    filename="comparison_multiplications.png",
)