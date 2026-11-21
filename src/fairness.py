"""Check whether the model's errors are spread fairly across Gender, AgeBand and MaritalStatus."""

import json
import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from fairlearn.metrics import MetricFrame, count, false_positive_rate, selection_rate
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from config_loader import load_config
from features import age_band, build_pipeline, load_features, split_data
from train import get_models

def recall(y_true, y_pred) -> float:
    """Recall that returns 0 instead of warning when a group has no leavers."""
    return recall_score(y_true, y_pred, zero_division=0)

def leavers(y_true, y_pred) -> int:
    """Number of actual leavers in the group. Recall is based only on these people."""
    return int(y_true.sum())

FAIRNESS_METRICS = {
    "recall": recall,
    "fpr": false_positive_rate,
    "selection_rate": selection_rate,
    "count": count,
    "leavers": leavers,
}

def sensitive_features(X: pd.DataFrame, bins: list, labels: list) -> pd.DataFrame:
    """Return the attributes we check fairness across. AgeBand is derived from Age."""
    return pd.DataFrame({
        "Gender": X["Gender"],
        "AgeBand": age_band(X["Age"], bins, labels),
        "MaritalStatus": X["MaritalStatus"],
    })

def fairness_gaps(y_true, y_pred, groups: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Return metrics per group and the largest gap between groups for recall and FPR."""
    frame = MetricFrame(
        metrics=FAIRNESS_METRICS, y_true=y_true, y_pred=y_pred, sensitive_features=groups
    )
    by_group = frame.by_group
    if isinstance(groups.dtype, pd.CategoricalDtype):
        # Keep age bands in age order rather than alphabetical order
        by_group = by_group.reindex(groups.cat.categories)
    gaps = {
        "recall_gap": float(by_group["recall"].max() - by_group["recall"].min()),
        "fpr_gap": float(by_group["fpr"].max() - by_group["fpr"].min()),
    }
    return by_group, gaps

def save_fairness_plot(tables: dict, output_path) -> None:
    """Bar charts of recall and false positive rate for each group."""
    fig, axes = plt.subplots(1, len(tables), figsize=(5 * len(tables), 4), sharey=True)
    for ax, (feature, table) in zip(axes, tables.items()):
        table[["recall", "fpr"]].plot.bar(ax=ax, rot=0)
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylim(0, 1)
        # Show group sizes, because small groups give unreliable rates
        ax.set_xticklabels([
            f"{g}\n(n={int(row['count'])}, leavers={int(row['leavers'])})"
            for g, row in table.iterrows()
        ], fontsize=8)
    axes[0].set_ylabel("Rate")
    fig.suptitle("Recall and false positive rate by group (test set)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    cfg = load_config()
    seed = cfg["split"]["seed"]
    max_gap = cfg["success_criteria"]["max_fairness_gap"]
    bins, labels = cfg["fairness"]["age_bins"], cfg["fairness"]["age_labels"]
    figures_dir = cfg["paths"]["figures"]

    X, y = load_features(cfg["paths"]["processed_data"])
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=cfg["split"]["test_size"], seed=seed
    )
    saved = joblib.load(cfg["paths"]["model"])
    pipe, threshold, model_name = saved["pipeline"], saved["threshold"], saved["model_name"]

    # Two views: out-of-fold on the training set (1176 people, bigger groups)
    # and the held-out test set (294 people, the final check)
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    oof_proba = cross_val_predict(
        build_pipeline(get_models(seed)[model_name]), X_train, y_train,
        cv=cv, method="predict_proba",
    )[:, 1]
    test_proba = pipe.predict_proba(X_test)[:, 1]

    datasets = {
        "train_oof": (X_train, y_train, (oof_proba >= threshold).astype(int)),
        "test": (X_test, y_test, (test_proba >= threshold).astype(int)),
    }

    rows, all_gaps, test_tables = [], {}, {}
    for dataset, (X_part, y_part, flags) in datasets.items():
        groups = sensitive_features(X_part, bins, labels)
        print(f"\n===== {dataset} ({len(y_part)} employees) =====")
        for feature in groups.columns:
            by_group, gaps = fairness_gaps(y_part, flags, groups[feature])
            print(f"\n{feature}")
            print(by_group.round(3).to_string())
            for gap_name, gap in gaps.items():
                status = "PASS" if gap <= max_gap else "FAIL"
                print(f"{gap_name}: {gap:.3f} ({status}, limit {max_gap})")
                all_gaps[f"{dataset}_{feature}_{gap_name}"] = round(gap, 4)

            table = by_group.reset_index().rename(columns={feature: "group"})
            table.insert(0, "feature", feature)
            table.insert(0, "dataset", dataset)
            rows.append(table)
            if dataset == "test":
                test_tables[feature] = by_group

    pd.concat(rows).round(3).to_csv(figures_dir / "fairness_results.csv", index=False)
    save_fairness_plot(test_tables, figures_dir / "fairness_test.png")

    # Saved separately from metrics.json so the CI quality gate can read both
    fairness_path = cfg["paths"]["metrics"].parent / "fairness.json"
    with open(fairness_path, "w", encoding="utf-8") as f:
        json.dump({"model_name": model_name, "max_gap": max_gap, "gaps": all_gaps}, f, indent=2)
    print(f"\nSaved fairness results to {fairness_path}")

    mlflow.set_experiment("hr-attrition")
    with mlflow.start_run(run_name=f"{model_name}_fairness"):
        mlflow.log_metrics(all_gaps)
        mlflow.set_tag("stage", "fairness_evaluation")
