import pandas as pd

def get_missingness_report(df, info_df=None):
    """
    Generates a comprehensive diagnostic report of missing values.
    Optionally maps descriptions from a variables_info DataFrame.
    """
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        return "No missing values detected."

    report = pd.DataFrame({
        "missing_count": null_counts,
        "missing_pct": (null_counts / len(df)) * 100
    }).loc[lambda x: x["missing_count"] > 0]

    if info_df is not None:
        mapping = info_df.set_index("Variable")["Description"]
        report["description"] = report.index.map(mapping)

    return report.sort_values("missing_pct", ascending=False)