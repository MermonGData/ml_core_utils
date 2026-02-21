import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ml_core_utils.config import RESIDUAL_STYLE
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


def evaluate_regression_metrics_df(y_true, y_pred, warn=True):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Safety check for percentage metrics
    epsilon = 1e-8
    near_zero = np.abs(y_true) < 1e-6
    if warn and np.any(near_zero):
        print("Warning: `y_true` contains near-zero values - MAPE and RMSPE may be unstable.")

    # Calculations
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Percentage errors
    pct_diffs = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)
    mape = np.mean(pct_diffs) * 100
    rmspe = np.sqrt(np.mean(pct_diffs**2)) * 100

    # Correlation (Handling zero-variance edge cases)
    try:
        pearson = np.corrcoef(y_true, y_pred)[0, 1]
    except:
        pearson = np.nan

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE [%]": mape,
        "RMSPE [%]": rmspe,
        "R²": r2_score(y_true, y_pred),
        "Pearson correlation": pearson
    }

    return pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]).round(4)

def plot_residuals(y_true, y_pred, title="Residuals Distribution"):
    set_base_style()
    s = RESIDUAL_STYLE
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred

    # Histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(
        residuals,
        bins=s["bins"],
        kde=True,
        color=s["hist_color"],
        alpha=0.8
    )    

    plt.axvline(
        0,
        color=s["zero_line_color"],
        linestyle=s["zero_line_style"],
        linewidth=s["zero_line_width"]
    )

    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_residuals_vs_fitted(model, X_test, y_true, title="Residuals vs Fitted"):
    set_base_style()
    s = RESIDUAL_STYLE

    y_pred = model.predict(X_test)
    residuals = y_true - y_pred
    
    plt.scatter(
        y_pred,
        residuals,
        alpha=s["alpha"],
        s=s["marker_size"],
        edgecolor=s["edgecolor"],
        color=s["scatter_color"]
    )

    plt.axhline(
        0,
        color=s["zero_line_color"],
        linestyle=s["zero_line_style"],
        linewidth=s["zero_line_width"]
    )

    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_residuals_grid(trained_models_dict, comparison_df, dataset_name, y_test, 
                        n_models=4, figsize=(14, 10)):
    set_base_style()
    s = RESIDUAL_STYLE

    #Create a grid of residual plots for top models in a specific dataset
    dataset_models = comparison_df[comparison_df["Dataset"] == dataset_name].head(n_models)
    models_dict = trained_models_dict[dataset_name]
    
    n_rows = int(np.ceil(len(dataset_models) / 2))
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(len(dataset_models), len(axes)):
        axes[i].set_visible(False)
    
    for ax, (_, row) in zip(axes, dataset_models.iterrows()):
        model_name = row["Model"]
        
        entry = models_dict[model_name]
        model = entry["model"]
        
        if "y_pred" in entry:
            y_pred = entry["y_pred"]
        else:
            X_test = entry["X_test"]
            y_pred = model.predict(X_test)
        
        residuals = y_test.values - y_pred
        
        ax.scatter(
        y_pred,
        residuals,
        alpha=s["alpha"],
        s=s["marker_size"],
        edgecolor=s["edgecolor"],
        color=s["scatter_color"]
    )
        ax.axhline(
        0,
        color=s["zero_line_color"],
        linestyle=s["zero_line_style"],
        linewidth=s["zero_line_width"]
    )
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        ax.set_title(f"{model_name} ({dataset_name})")
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(np.mean(residuals**2))
        ax.text(0.05, 0.95, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig, axes