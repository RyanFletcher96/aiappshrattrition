"""Label-free drift monitoring: compare a new monthly batch of employees with the training data.

Four kinds of drift matter for this model:
- data drift: the inputs change (e.g. more people working overtime)
- concept drift: the link between inputs and leaving changes (needs outcome labels)
- prediction drift: the model's risk scores change
- label drift: the attrition rate itself changes (needs outcome labels)

Outcome labels arrive months after scoring, so this module checks only data drift and
prediction drift. An alert means "a person should look at this", not "retrain automatically".
"""

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency, ks_2samp
from config_loader import load_config
from features import (
    NOMINAL_COLUMNS, NUMERIC_COLUMNS, ORDINAL_COLUMNS, load_features, split_data,
)

# Only model inputs are monitored. Gender and MaritalStatus are not model inputs, but their
# group mix could be monitored separately to check the fairness results still apply.
CONTINUOUS_COLUMNS = NUMERIC_COLUMNS + ORDINAL_COLUMNS
CATEGORICAL_COLUMNS = NOMINAL_COLUMNS

# Common PSI rule of thumb
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25

# Stops log(0) and division by zero when a bin or category is empty in one dataset
EPSILON = 1e-4

BATCH_SIZE = 300

def _psi_from_proportions(ref_prop: np.ndarray, cur_prop: np.ndarray) -> float:
    """PSI = sum((current - reference) * ln(current / reference)) over bins."""
    ref_prop = np.clip(ref_prop, EPSILON, None)
    cur_prop = np.clip(cur_prop, EPSILON, None)
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))

def psi_numeric(reference, current, bins: int = 10) -> float:
    """Population Stability Index for a numeric feature, using reference quantiles as bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Inner edges only: the outer bins are open-ended so current values outside the
    # reference range still land in a bin. np.unique removes repeated edges, which happen
    # with ordinal features like JobSatisfaction (only values 1-4).
    inner_edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    n_bins = len(inner_edges) + 1

    # searchsorted gives each value its bin number: 0 = below the first edge, and so on
    ref_counts = np.bincount(np.searchsorted(inner_edges, reference, side="right"), minlength=n_bins)
    cur_counts = np.bincount(np.searchsorted(inner_edges, current, side="right"), minlength=n_bins)
    return _psi_from_proportions(ref_counts / len(reference), cur_counts / len(current))

def psi_categorical(reference, current) -> float:
    """Population Stability Index for a categorical feature, based on category proportions."""
    reference, current = pd.Series(reference), pd.Series(current)
    # Union, so a category that is new (or has disappeared) in the current batch still counts
    categories = sorted(set(reference) | set(current))
    ref_prop = reference.value_counts(normalize=True).reindex(categories, fill_value=0)
    cur_prop = current.value_counts(normalize=True).reindex(categories, fill_value=0)
    return _psi_from_proportions(ref_prop.to_numpy(), cur_prop.to_numpy())

def psi_status(psi: float) -> str:
    """Turn a PSI value into a plain-English status using the common PSI bands."""
    if psi < PSI_MODERATE:
        return "stable"
    if psi <= PSI_SIGNIFICANT:
        return "moderate - monitor"
    return "significant - investigate"

def chi2_p_value(reference, current) -> float:
    """Chi-squared test p-value on a 2-row table of category counts (reference vs current)."""
    table = pd.DataFrame({
        "reference": pd.Series(reference).value_counts(),
        "current": pd.Series(current).value_counts(),
    }).fillna(0).T
    if table.shape[1] < 2:
        # Only one category in both datasets, so there is nothing to compare
        return 1.0
    return float(chi2_contingency(table)[1])

def drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """One row per monitored model input: PSI, a statistical test p-value and a status."""
    rows = []
    for col in CONTINUOUS_COLUMNS:
        rows.append({
            "feature": col,
            "type": "ordinal" if col in ORDINAL_COLUMNS else "numeric",
            "psi": psi_numeric(reference_df[col], current_df[col]),
            "test": "ks",
            "p_value": float(ks_2samp(reference_df[col], current_df[col]).pvalue),
        })
    for col in CATEGORICAL_COLUMNS:
        rows.append({
            "feature": col,
            "type": "nominal",
            "psi": psi_categorical(reference_df[col], current_df[col]),
            "test": "chi2",
            "p_value": chi2_p_value(reference_df[col], current_df[col]),
        })
    report = pd.DataFrame(rows)
    # Status comes from PSI (size of change) rather than the p-value, because with
    # large batches even tiny, harmless changes become "statistically significant"
    report["status"] = report["psi"].apply(psi_status)
    return report.sort_values("psi", ascending=False).reset_index(drop=True)

def prediction_drift(reference_scores, current_scores, threshold: float) -> dict:
    """Compare the model's risk scores and the share flagged between reference and current."""
    reference_scores = np.asarray(reference_scores)
    current_scores = np.asarray(current_scores)
    ref_flagged = float((reference_scores >= threshold).mean())
    cur_flagged = float((current_scores >= threshold).mean())
    psi = psi_numeric(reference_scores, current_scores)
    return {
        "score_psi": psi,
        "score_status": psi_status(psi),
        "reference_flagged_share": ref_flagged,
        "current_flagged_share": cur_flagged,
        "flagged_share_change": cur_flagged - ref_flagged,
    }

def make_shifted_batch(X: pd.DataFrame, seed: int, overtime_share: float = 0.5,
                       income_factor: float = 0.85) -> pd.DataFrame:
    """Simulate a changed month: more people on overtime and monthly income cut by 15%.

    This is a deliberate, made-up change so we can check the monitor catches it.
    """
    batch = X.sample(n=BATCH_SIZE, random_state=seed).copy()
    rng = np.random.default_rng(seed)

    # Move randomly chosen "No" rows to "Yes" until overtime_share of the batch is on overtime
    no_rows = batch.index[batch["OverTime"] == "No"]
    n_to_flip = max(int(round(overtime_share * len(batch))) - int((batch["OverTime"] == "Yes").sum()), 0)
    batch.loc[rng.choice(no_rows, size=n_to_flip, replace=False), "OverTime"] = "Yes"

    # Keep income a whole number and at least 1, matching the cleaning rules
    batch["MonthlyIncome"] = (batch["MonthlyIncome"] * income_factor).round().astype(int).clip(lower=1)
    return batch

def save_psi_plot(report: pd.DataFrame, output_path, title: str) -> None:
    """Horizontal bar chart of PSI per feature, with the 0.1 and 0.25 alert lines."""
    colours = {
        "stable": "#b5b4aa",
        "moderate - monitor": "#fab219",
        "significant - investigate": "#d03b3b",
    }
    report = report.sort_values("psi")
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh(report["feature"], report["psi"], color=report["status"].map(colours))
    ax.axvline(PSI_MODERATE, color="#6b6a63", linestyle="--", linewidth=1)
    ax.axvline(PSI_SIGNIFICANT, color="#6b6a63", linestyle="-", linewidth=1)
    # Name the lines at the top of the chart (x in data units, y as a fraction of the axes)
    for x, label in [(PSI_MODERATE, "monitor"), (PSI_SIGNIFICANT, "investigate")]:
        ax.text(x, 1.005, f"{x} {label}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color="#6b6a63")

    # Label only the alerted bars so the important values are easy to read
    for _, row in report[report["psi"] >= PSI_MODERATE].iterrows():
        ax.text(row["psi"], row["feature"], f"  {row['psi']:.2f}", va="center", fontsize=9)

    ax.set_xlim(0, max(report["psi"].max() * 1.15, PSI_SIGNIFICANT * 1.3))
    ax.set_xlabel("Population Stability Index (PSI) - higher means a bigger change")
    ax.set_ylabel("")
    ax.set_title(title, pad=22)
    ax.legend(handles=[
        Patch(color=colours["stable"], label="Stable (PSI below 0.1)"),
        Patch(color=colours["moderate - monitor"], label="Moderate, monitor (0.1 to 0.25)"),
        Patch(color=colours["significant - investigate"], label="Significant, investigate (above 0.25)"),
    ], loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    cfg = load_config()
    seed = cfg["split"]["seed"]
    figures_dir = cfg["paths"]["figures"]

    X, y = load_features(cfg["paths"]["processed_data"])
    # The test set has already been used for the final evaluation, so the reference
    # and both simulated batches come from the training split only
    X_train, _, _, _ = split_data(X, y, test_size=cfg["split"]["test_size"], seed=seed)

    saved = joblib.load(cfg["paths"]["model"])
    pipe, threshold = saved["pipeline"], saved["threshold"]
    model_inputs = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
    # These are in-sample scores (the model was fitted on X_train); fine as a reference
    # here because both demo batches are also drawn from X_train
    reference_scores = pipe.predict_proba(X_train[model_inputs])[:, 1]

    # Both batches start from the same 300 sampled rows (same seed), so any difference
    # between their reports comes only from the simulated change
    batches = {
        "stable_batch": X_train.sample(n=BATCH_SIZE, random_state=seed),
        "shifted_batch": make_shifted_batch(X_train, seed=seed),
    }

    reports, metrics = [], {}
    for name, batch in batches.items():
        report = drift_report(X_train, batch)
        scores = pipe.predict_proba(batch[model_inputs])[:, 1]
        pred = prediction_drift(reference_scores, scores, threshold)
        alerts = report[report["psi"] >= PSI_MODERATE]

        print(f"\n===== {name} ({len(batch)} employees vs {len(X_train)} in training) =====")
        print("Top 5 features by PSI:")
        print(report.head(5).round(4).to_string(index=False))
        print(f"Feature alerts (PSI >= {PSI_MODERATE}): "
              f"{', '.join(alerts['feature']) if len(alerts) else 'none'}")
        print(f"Prediction drift: score PSI {pred['score_psi']:.3f} ({pred['score_status']}), "
              f"flagged share {pred['reference_flagged_share']:.1%} -> "
              f"{pred['current_flagged_share']:.1%} ({pred['flagged_share_change']:+.1%})")

        report.insert(0, "batch", name)
        reports.append(report)
        metrics.update({
            f"{name}_max_feature_psi": float(report["psi"].max()),
            f"{name}_n_moderate": int((report["status"] == "moderate - monitor").sum()),
            f"{name}_n_significant": int((report["status"] == "significant - investigate").sum()),
            f"{name}_score_psi": pred["score_psi"],
            f"{name}_flagged_share": pred["current_flagged_share"],
            f"{name}_flagged_share_change": pred["flagged_share_change"],
        })

    pd.concat(reports).round(4).to_csv(figures_dir / "drift_report.csv", index=False)
    save_psi_plot(
        reports[1], figures_dir / "drift_psi.png",
        "Feature drift: simulated shifted batch vs training data\n"
        "(more overtime, monthly income cut by 15%)",
    )
    print(f"\nSaved {figures_dir / 'drift_report.csv'} and {figures_dir / 'drift_psi.png'}")

    mlflow.set_experiment("hr-attrition")
    with mlflow.start_run(run_name="drift_monitoring_demo"):
        mlflow.log_metrics(metrics)
        mlflow.log_params({"batch_size": BATCH_SIZE, "psi_moderate": PSI_MODERATE,
                           "psi_significant": PSI_SIGNIFICANT})
        mlflow.set_tag("stage", "monitoring")

    print("\nAlerts trigger an investigation by the model owner, not automatic retraining.")
