"""Try ways to reduce the age bias, using out-of-fold predictions on the training set only.

The test set is never used here, so it stays a fair final check.
"""

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from config_loader import load_config
from fairness import fairness_gaps, sensitive_features
from features import build_pipeline, load_features, split_data
from train import choose_threshold, get_models

# Inputs removed in each variant. TotalWorkingYears is tested because it correlates with Age.
VARIANTS = {
    "baseline_with_age": (),
    "no_age": ("Age",),
    "no_age_or_total_working_years": ("Age", "TotalWorkingYears"),
}

def per_group_thresholds(y_true, proba, groups: pd.Series, min_recall: float):
    """Choose a separate threshold for each group so every group reaches min_recall."""
    flags = np.zeros(len(proba), dtype=int)
    thresholds = {}
    for group in groups.unique():
        in_group = (groups == group).to_numpy()
        threshold, _, _ = choose_threshold(y_true[in_group], proba[in_group], min_recall)
        thresholds[str(group)] = round(threshold, 3)
        flags[in_group] = (proba[in_group] >= threshold).astype(int)
    return flags, thresholds

def summarise(name: str, y_true, proba, flags, groups: pd.DataFrame) -> dict:
    """Overall performance plus the fairness gaps for every sensitive attribute."""
    row = {
        "variant": name,
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, flags, zero_division=0),
        "recall": recall_score(y_true, flags, zero_division=0),
    }
    for feature in groups.columns:
        by_group, gaps = fairness_gaps(y_true, flags, groups[feature])
        row[f"{feature}_recall_gap"] = gaps["recall_gap"]
        row[f"{feature}_fpr_gap"] = gaps["fpr_gap"]
        if feature == "AgeBand":
            for band, recall in by_group["recall"].items():
                # MLflow names can't contain < or >, so "<30" becomes "under_30"
                safe_band = str(band).replace("<", "under_").replace(">", "over_")
                row[f"recall_age_{safe_band}"] = recall
    return row

if __name__ == "__main__":
    cfg = load_config()
    seed = cfg["split"]["seed"]
    min_recall = cfg["success_criteria"]["min_recall"]
    bins, labels = cfg["fairness"]["age_bins"], cfg["fairness"]["age_labels"]

    X, y = load_features(cfg["paths"]["processed_data"])
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=cfg["split"]["test_size"], seed=seed
    )
    groups = sensitive_features(X_train, bins, labels)
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    mlflow.set_experiment("hr-attrition")

    rows = []
    baseline_proba = None
    for name, excluded in VARIANTS.items():
        pipe = build_pipeline(get_models(seed)["logistic_regression"], exclude_columns=excluded)
        proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        # Each variant gets its own threshold, tuned to the same recall target
        threshold, _, _ = choose_threshold(y_train, proba, min_recall)
        flags = (proba >= threshold).astype(int)
        row = summarise(name, y_train, proba, flags, groups)
        row["threshold"] = threshold
        rows.append(row)
        if name == "baseline_with_age":
            baseline_proba = proba

    # Post-processing alternative: keep the baseline model, but one threshold per age band
    flags, band_thresholds = per_group_thresholds(
        y_train, baseline_proba, groups["AgeBand"], min_recall
    )
    row = summarise("baseline_threshold_per_age_band", y_train, baseline_proba, flags, groups)
    row["threshold"] = str(band_thresholds)
    rows.append(row)

    results = pd.DataFrame(rows)
    for row in rows:
        with mlflow.start_run(run_name=f"fairness_experiment_{row['variant']}"):
            mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, float)})
            mlflow.set_tag("stage", "fairness_experiment")

    results.round(3).to_csv(cfg["paths"]["figures"] / "fairness_experiments.csv", index=False)
    columns = ["variant", "precision", "recall", "AgeBand_recall_gap", "AgeBand_fpr_gap",
               "MaritalStatus_recall_gap", "Gender_recall_gap"]
    print("Fairness experiments (training set, out-of-fold, recall target "
          f"{min_recall}, fairness limit {cfg['success_criteria']['max_fairness_gap']}):\n")
    print(results[columns].round(3).to_string(index=False))
    print("\nRecall by age band:")
    age_cols = ["variant"] + [c for c in results.columns if c.startswith("recall_age_")]
    print(results[age_cols].round(3).to_string(index=False))
    print(f"\nPer-band thresholds: {band_thresholds}")
