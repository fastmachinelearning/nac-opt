import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.cm import ScalarMappable


def scatterplot_2d(
    source_csv: str,
    columns: list,
    x_axis_name: str,
    y_axis_name: str,
    output_path: str = "scatterplot_2d.png",
    limit: int = 1000,
):
    """
    source_csv - input csv file
    columns - column name mapping (e.g., ['config', 'date', 'bops', 'accuracy_top1'])
    x_axis_name - must be in columns list
    y_axis_name - must be in columns list
    output_path - png file output
    limit - how many search points to plot starting from index 0
    """

    df = pd.read_csv(source_csv)[:limit]
    df.columns = columns

    assert x_axis_name in columns, "x_axis_name missing in columns"
    assert y_axis_name in columns, "y_axis_name missing in columns"

    fig, ax = plt.subplots(figsize=(7, 5))

    # Select colormap and define count
    cm = plt.cm.get_cmap("viridis_r")
    count = [x for x in range(len(df))]

    ax.scatter(
        df[x_axis_name].values,
        df[y_axis_name].values,
        marker="^",
        alpha=0.8,
        c=count,
        cmap=cm,
        label="Discovered Model",
        s=10,
    )
    ax.set_title("Morph Search Results")
    ax.set_xlabel(x_axis_name, fontsize=13)
    ax.set_ylabel(y_axis_name, fontsize=13)
    ax.legend(fancybox=True, fontsize=10, framealpha=1, borderpad=0.2, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Eval Count bar
    norm = plt.Normalize(0, len(df))
    sm = ScalarMappable(norm=norm, cmap=cm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.85)
    cbar.ax.set_title("         Evaluation\n  Count", fontsize=8)

    fig.tight_layout(pad=2)
    plt.savefig(output_path)


def swarmplot_1d(source_csv: str, columns: list, axis_name: str, output_path: str = "swarmplot.png", limit: int = 1000):
    """
    source_csv - input csv file
    columns - column name mapping (e.g., ['config', 'date', 'bops', 'accuracy_top1'])
    axis_name - must be in columns list
    output_path - png file output
    limit - how many search points to plot starting from index 0
    """
    # Lazy Import
    import seaborn as sns

    df = pd.read_csv(source_csv)[:limit]
    df.columns = columns

    assert axis_name in columns, "axis_name missing in columns"

    sns.set(style="whitegrid")
    palette = sns.color_palette("viridis_r", n_colors=len(df))
    sns.swarmplot(x=[1] * len(df), y=axis_name, hue=df.index.tolist(), data=df, palette=palette)
    plt.legend([], [], frameon=False)
    plt.savefig(output_path)
