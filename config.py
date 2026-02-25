import os
import warnings
import numpy as np
from IPython.display import HTML, display
import matplotlib.pyplot as plt
import seaborn as sns

def setup_notebook(seed: int = 42):
    warnings.filterwarnings("ignore")
    np.random.seed(seed)
    display(HTML("<style>.output_scroll { height: auto !important; }</style>"))

def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def resolve_project_path(*paths, marker="data"):
    current = os.path.abspath(os.getcwd())
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, marker)):
            return os.path.join(current, *paths)
        current = os.path.dirname(current)
    return os.path.join(os.getcwd(), *paths)

RESIDUAL_STYLE = {
    "scatter_color": "#4C72B0",
    "alpha": 0.6,
    "edgecolor": "black",
    "marker_size": 40,
    "zero_line_color": "#C44E52",
    "zero_line_style": "--",
    "zero_line_width": 1.5,
    "hist_color": "#4C72B0",
    "bins": 30,
}

CLUSTER_COLORS = {
    0: "#1f77b4",  # blue
    1: "#ff7f0e",  # orange
    2: "#2ca02c",  # green
    3: "#d62728",  # red
    4: "#9467bd",  # purple
    5: "#8c564b",  # brown
    6: "#e377c2",  # pink
    7: "#7f7f7f",  # gray
}

def set_base_style():
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=1.0
    )
    plt.rcParams.update({
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (7, 4),
    })
