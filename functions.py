import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os


PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

LABEL_MAX_LEN = 20

def load_excel_file(file_path):
    """
    Load an Excel file and return a DataFrame.

    Parameters:
    file_path (str): The path to the Excel file.

    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_excel(file_path)

        # Shorter column names for easier access
        df.columns = [
            "timestamp",
            "email",
            "nombre",
            "genero",
            "facultad",
            "estamento",
            "usa_supercomputo",
            "conoce_supercomputo",
            "habilidad_jupyter",
            "habilidad_linux",
            "usa_cli",
            "asiste_taller",
        ]

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#e0e0e0",
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.bottom": False,
            "ytick.left": False,
        })

        

        return df

    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None
    
def _make_references(labels):
    """Return (short_labels, ref_text) when any label exceeds LABEL_MAX_LEN, else (labels, None)."""
    if not any(len(str(l)) > LABEL_MAX_LEN for l in labels):
        return [str(l) for l in labels], None
    keys = [chr(65 + i) for i in range(len(labels))]  # A, B, C, …
    ref_lines = [f"{k}: {l}" for k, l in zip(keys, labels)]
    return keys, "\n".join(ref_lines)


def _annotate_bars(ax, bars, values):
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(int(val)),
                ha="center", va="bottom",
                fontsize=9, color="#333333",
            )


def plot_histogram(df, column, title=None, xlabel=None, color=None, save_path=None):
    """Plot a histogram/bar chart for a dataframe column.

    Works for both numeric columns (histogram with bins) and
    categorical columns (bar chart of value counts). Long category
    labels are replaced with letter keys; the full reference table
    is shown above the chart.
    """
    series = df[column]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#f9f9f9")
    ax.set_facecolor("#f9f9f9")

    if pd.api.types.is_numeric_dtype(series):
        bar_color = color or PALETTE[0]
        n, bins, patches = ax.hist(
            series.dropna(), color=bar_color, edgecolor="white", linewidth=0.8, rwidth=0.85,
        )
        ax.set_ylim(0, max(n) * 1.18)
        _annotate_bars(ax, patches, n)
        ax.set_ylabel("Frecuencia", fontsize=10, color="#555555")
    else:
        counts = series.value_counts()
        tick_labels, ref_text = _make_references(counts.index)
        bar_colors = [color] * len(counts) if color else PALETTE[:len(counts)]
        bars = ax.bar(
            tick_labels, counts.values,
            color=bar_colors, edgecolor="white", linewidth=0.8, width=0.6,
        )
        n_lines = len(counts)
        ax.set_ylim(0, counts.values.max() * (1 + 0.065 * n_lines + 0.18))
        _annotate_bars(ax, bars, counts.values)
        ax.set_ylabel("Frecuencia", fontsize=10, color="#555555")
        ax.tick_params(axis="x", labelsize=11, labelcolor="#333333")
        if ref_text:
            ax.text(
                0.01, 0.99,
                ref_text,
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=7.5,
                family="monospace",
                color="#333333",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9),
            )

    ax.set_title(title or column, fontsize=13, fontweight="bold", color="#222222", pad=12)
    ax.set_xlabel(xlabel or column, fontsize=10, color="#555555", labelpad=8)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.tick_params(axis="y", labelsize=9, labelcolor="#777777")
    ax.spines["bottom"].set_color("#cccccc")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()

    plt.show()
    plt.close(fig)


def create_plots_sections():
    plots = [
        ("facultad",           "Distribución por Facultad"),
        ("estamento",          "Distribución por Estamento"),
        ("usa_supercomputo",   "Usa Supercómputo"),
        ("conoce_supercomputo","Conoce Supercómputo"),
        ("habilidad_jupyter",  "Habilidad con Jupyter"),
        ("habilidad_linux",    "Habilidad con Linux"),
        ("usa_cli",            "Habilidad con CLI"),
    ]

    return plots