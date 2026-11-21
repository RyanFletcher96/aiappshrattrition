"""Evaluate the saved model once on the held-out test set and record the results."""

import json
import joblib
import matplotlib.pyplot as plt
import mlflow
from datetime import datetime, timezone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from config_loader import load_config
from features import load_features, split_data

def threshold_metrics(y_true, proba, threshold: float) -> dict:
    """Return metrics for both the risk scores and the yes/no flags at the threshold."""
    flagged = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "recall": recall_score(y_true, flagged, zero_division=0),
        "precision": precision_score(y_true, flagged, zero_division=0),
        "f1": f1_score(y_true, flagged, zero_division=0),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }

def check_criteria(metrics: dict, criteria: dict, baseline_pr_auc: float) -> dict:
    """Compare test metrics with the success criteria in config.yaml."""
    return {
        "recall_target_met": bool(metrics["recall"] >= criteria["min_recall"]),
        "precision_target_met": bool(metrics["precision"] >= criteria["min_precision"]),
        "beats_dummy_baseline": bool(metrics["pr_auc"] > baseline_pr_auc),
    }

def save_pr_curve(y_true, proba, threshold: float, metrics: dict, model_name: str, output_path) -> None:
    """Plot the precision-recall curve with the chosen threshold and the baseline marked."""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    baseline = y_true.mean()
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"{model_name} (PR-AUC {metrics['pr_auc']:.3f})")
    plt.axhline(baseline, color="grey", linestyle="--", label=f"Dummy baseline ({baseline:.3f})")
    plt.scatter(metrics["recall"], metrics["precision"], color="red", zorder=3,
                label=f"Threshold {threshold:.3f}")
    plt.xlabel("Recall (share of leavers flagged)")
    plt.ylabel("Precision (share of flags that were leavers)")
    plt.title("Precision-recall curve (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def save_confusion_matrix(y_true, proba, threshold: float, output_path) -> None:
    """Plot the confusion matrix at the chosen threshold."""
    flagged = (proba >= threshold).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_true, flagged, display_labels=["Stayed", "Left"], cmap="Blues", colorbar=False
    )
    plt.title(f"Confusion matrix (test set, threshold {threshold:.3f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    cfg = load_config()
    figures_dir = cfg["paths"]["figures"]

    # Recreate the exact same split as train.py (same seed), then use only the test side
    X, y = load_features(cfg["paths"]["processed_data"])
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=cfg["split"]["test_size"], seed=cfg["split"]["seed"]
    )

    saved = joblib.load(cfg["paths"]["model"])
    pipe, threshold, model_name = saved["pipeline"], saved["threshold"], saved["model_name"]
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics = threshold_metrics(y_test, proba, threshold)
    # A model that always predicts the base rate gets PR-AUC equal to the attrition rate
    baseline_pr_auc = float(y_test.mean())
    criteria = check_criteria(metrics, cfg["success_criteria"], baseline_pr_auc)

    print(f"Test set: {len(y_test)} employees, {int(y_test.sum())} leavers")
    print(f"Model: {model_name}, threshold {threshold:.3f}\n")
    for name, value in metrics.items():
        print(f"{name:<16} {value:.3f}" if isinstance(value, float) else f"{name:<16} {value}")
    print(f"\nDummy baseline PR-AUC: {baseline_pr_auc:.3f}")
    for name, passed in criteria.items():
        print(f"{name:<22} {'PASS' if passed else 'FAIL'}")
    print("\n" + classification_report(
        y_test, (proba >= threshold).astype(int), target_names=["Stayed", "Left"]
    ))

    save_pr_curve(y_test, proba, threshold, metrics, model_name, figures_dir / "pr_curve_test.png")
    save_confusion_matrix(y_test, proba, threshold, figures_dir / "confusion_matrix_test.png")

    results = {
        "model_name": model_name,
        "threshold": threshold,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_size": len(y_test),
        "baseline_pr_auc": baseline_pr_auc,
        # float() turns numpy numbers into plain floats so json can save them
        "metrics": {k: float(v) if isinstance(v, float) else v for k, v in metrics.items()},
        "criteria": criteria,
    }
    with open(cfg["paths"]["metrics"], "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {cfg['paths']['metrics']}")

    mlflow.set_experiment("hr-attrition")
    with mlflow.start_run(run_name=f"{model_name}_test_evaluation"):
        mlflow.log_param("threshold", threshold)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.set_tag("stage", "final_test_evaluation")
