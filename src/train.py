"""Train and compare models with cross-validation on the training set, then save the chosen one."""

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, make_scorer, precision_recall_curve, precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from config_loader import load_config
from features import build_pipeline, load_features, split_data

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    # zero_division=0 stops warnings when the dummy model predicts no leavers
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": "f1",
}

def get_models(seed: int) -> dict:
    """Return the unfitted models to compare, with a dummy baseline first."""
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed
        ),
    }

def evaluate_cv(pipe, X_train: pd.DataFrame, y_train: pd.Series, cv) -> dict:
    """Cross-validate a pipeline and return the mean and std of each metric."""
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING)
    results = {}
    for metric in SCORING:
        fold_scores = scores[f"test_{metric}"]
        results[f"{metric}_mean"] = fold_scores.mean()
        results[f"{metric}_std"] = fold_scores.std()
    return results

def choose_threshold(y_true, proba, min_recall: float) -> tuple[float, float, float]:
    """Pick the threshold with the highest precision among those with recall >= min_recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision and recall have one more value than thresholds, so drop the last one
    precision, recall = precision[:-1], recall[:-1]

    meets_recall = recall >= min_recall
    if not meets_recall.any():
        raise ValueError(f"No threshold reaches recall >= {min_recall}")

    best = np.argmax(np.where(meets_recall, precision, -1))
    return float(thresholds[best]), float(precision[best]), float(recall[best])

def save_calibration_plot(y_true, proba, output_path: Path) -> None:
    """Plot predicted probability against observed attrition rate."""
    observed, predicted = calibration_curve(y_true, proba, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.plot(predicted, observed, "o-", label="Model (out-of-fold)")
    plt.xlabel("Predicted probability of leaving")
    plt.ylabel("Observed attrition rate")
    plt.title("Calibration curve (training set, out-of-fold)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    cfg = load_config()
    seed = cfg["split"]["seed"]
    criteria = cfg["success_criteria"]
    figures_dir = cfg["paths"]["figures"]

    X, y = load_features(cfg["paths"]["processed_data"])
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=cfg["split"]["test_size"], seed=seed
    )
    # The test set is not used in this file. It is evaluated once, in evaluate.py.

    # Same folds for every model, so the comparison is fair
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    mlflow.set_experiment("hr-attrition")

    # 1. Compare models with cross-validation
    rows = []
    for name, model in get_models(seed).items():
        with mlflow.start_run(run_name=name):
            results = evaluate_cv(build_pipeline(model), X_train, y_train, cv)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(results)
            mlflow.set_tag("include_protected", False)
        rows.append({"model": name, **results})
        print(f"{name:<20} PR-AUC {results['pr_auc_mean']:.3f} +/- {results['pr_auc_std']:.3f}")

    cv_results = pd.DataFrame(rows).round(3)
    cv_results.to_csv(figures_dir / "cv_results.csv", index=False)
    print("\nCross-validation results (training set, 5 folds):")
    print(cv_results[["model", "pr_auc_mean", "pr_auc_std", "roc_auc_mean",
                      "recall_mean", "precision_mean"]].to_string(index=False))

    # 2. Choose the best model by mean PR-AUC
    best = cv_results.sort_values("pr_auc_mean", ascending=False).iloc[0]
    best_name = best["model"]
    print(f"\nBest model: {best_name}")
    if best["pr_auc_std"] > criteria["max_cv_pr_auc_std"]:
        print(f"WARNING: PR-AUC std {best['pr_auc_std']} is above "
              f"{criteria['max_cv_pr_auc_std']}, so results vary a lot between folds")

    # 3. Comparison run: same model with Gender and MaritalStatus as inputs
    best_model = get_models(seed)[best_name]
    with mlflow.start_run(run_name=f"{best_name}_with_protected"):
        protected_results = evaluate_cv(
            build_pipeline(best_model, include_protected=True), X_train, y_train, cv
        )
        mlflow.log_params(best_model.get_params())
        mlflow.log_metrics(protected_results)
        mlflow.set_tag("include_protected", True)
    print(f"With protected attributes: PR-AUC {protected_results['pr_auc_mean']:.3f} "
          f"(without: {best['pr_auc_mean']:.3f})")

    # 4. Choose the decision threshold from out-of-fold predictions (never the test set)
    pipe = build_pipeline(get_models(seed)[best_name])
    oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold, precision, recall = choose_threshold(y_train, oof_proba, criteria["min_recall"])
    print(f"\nThreshold {threshold:.3f}: recall {recall:.3f}, precision {precision:.3f}")
    if precision < criteria["min_precision"]:
        print(f"WARNING: precision is below the target of {criteria['min_precision']}")

    # 5. Calibration
    brier = brier_score_loss(y_train, oof_proba)
    save_calibration_plot(y_train, oof_proba, figures_dir / "calibration_curve.png")
    print(f"Brier score (out-of-fold): {brier:.3f}")

    # 6. Fit on the whole training set and save
    pipe.fit(X_train, y_train)
    model_path = cfg["paths"]["model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "threshold": threshold, "model_name": best_name},
        model_path,
    )
    print(f"Saved {best_name} to {model_path}")
