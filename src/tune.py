"""Tune the regularisation strength C of the logistic regression with cross-validation.

Only the training set is used. The test set was already used once for the final
evaluation, so this is a recorded experiment that informs a future model version.
It does not change models/model.joblib.
"""

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from config_loader import load_config
from features import build_pipeline, load_features, split_data
from train import evaluate_cv

# Small grid around the default C=1, spaced evenly on a log scale.
# A grid of 7 values is enough for one parameter; a bigger search would cost
# more compute than a problem with 1,176 training rows justifies.
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
CURRENT_C = 1

def get_model(C: float) -> LogisticRegression:
    """Same settings as the chosen model in train.py, with only C changed."""
    return LogisticRegression(C=C, class_weight="balanced", max_iter=1000)

def save_tuning_plot(results: pd.DataFrame, max_std: float, output_path: Path) -> None:
    """Plot mean PR-AUC with +/- one std error bars against C."""
    plt.figure(figsize=(7, 5.5))
    plt.plot(results["C"], results["pr_auc_mean"], color="grey", alpha=0.5)
    # Colour shows the stability target, which is about the spread (error bar size)
    groups = [
        (results["meets_std_target"], "tab:green", f"Spread meets target (std <= {max_std})"),
        (~results["meets_std_target"], "tab:orange", f"Spread above target (std > {max_std})"),
    ]
    for mask, colour, label in groups:
        if mask.any():  # skip empty groups so the legend only shows what is plotted
            plt.errorbar(results.loc[mask, "C"], results.loc[mask, "pr_auc_mean"],
                         yerr=results.loc[mask, "pr_auc_std"], fmt="o", capsize=4,
                         color=colour, label=label)

    current = results[results["C"] == CURRENT_C].iloc[0]
    plt.scatter([CURRENT_C], [current["pr_auc_mean"]], s=200, facecolors="none",
                edgecolors="black", zorder=3, label=f"Current model (C={CURRENT_C})")

    plt.xscale("log")
    plt.xticks(C_GRID, [str(c) for c in C_GRID])
    plt.xlabel("C (smaller = stronger regularisation)")
    plt.ylabel("PR-AUC (mean across folds)")
    plt.title("Logistic regression: PR-AUC vs C (training set, 5-fold CV)")
    # Legend below the axes so it never hides the error bars
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
    plt.figtext(0.5, 0.01, "Error bars show +/- one standard deviation across the folds. "
                "The stability target is about this spread, not the mean.",
                ha="center", fontsize=8)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(output_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    cfg = load_config()
    seed = cfg["split"]["seed"]
    max_std = cfg["success_criteria"]["max_cv_pr_auc_std"]
    figures_dir = cfg["paths"]["figures"]

    X, y = load_features(cfg["paths"]["processed_data"])
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=cfg["split"]["test_size"], seed=seed
    )
    # The test set is not used here: it has already been used for the final evaluation

    # Same folds as train.py, so the C=1 result matches the model comparison
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    mlflow.set_experiment("hr-attrition")

    # A plain loop rather than GridSearchCV, so every C gets its own MLflow run
    rows = []
    for C in C_GRID:
        model = get_model(C)
        with mlflow.start_run(run_name=f"tune_logistic_regression_C_{C}"):
            results = evaluate_cv(build_pipeline(model), X_train, y_train, cv)
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(results)
            mlflow.set_tag("stage", "tuning")
        rows.append({"C": C, **results})
        print(f"C={C:<5} PR-AUC {results['pr_auc_mean']:.3f} +/- {results['pr_auc_std']:.3f}")

    results = pd.DataFrame(rows)
    results["meets_std_target"] = results["pr_auc_std"] <= max_std
    columns = ["C", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "recall_mean",
               "precision_mean", "meets_std_target"]
    results[columns].round(3).to_csv(figures_dir / "tuning_results.csv", index=False)
    save_tuning_plot(results, max_std, figures_dir / "tuning_c.png")
    print("\nTuning results (training set, 5 folds):")
    print(results[columns].round(3).to_string(index=False))

    # Conclusion: is the best C clearly better than the current one, or within the noise?
    best = results.sort_values("pr_auc_mean", ascending=False).iloc[0]
    current = results[results["C"] == CURRENT_C].iloc[0]
    gain = best["pr_auc_mean"] - current["pr_auc_mean"]
    # Compare with the smaller of the two stds: if the gain is below even that, it is noise
    noise = min(best["pr_auc_std"], current["pr_auc_std"])
    print(f"\nBest C by mean PR-AUC: {best['C']} "
          f"({best['pr_auc_mean']:.3f} +/- {best['pr_auc_std']:.3f})")
    print(f"Current C={CURRENT_C}: {current['pr_auc_mean']:.3f} +/- {current['pr_auc_std']:.3f}")
    if gain > noise:
        print(f"Gain of {gain:.3f} is larger than the fold-to-fold std ({noise:.3f})")
    else:
        print(f"Gain of {gain:.3f} is smaller than the fold-to-fold std ({noise:.3f}), "
              "so it is within the noise")
    stable = results.loc[results["meets_std_target"], "C"].tolist()
    if stable:
        print(f"C values meeting the std target (<= {max_std}): {stable}")
    else:
        print(f"No C value meets the std target (<= {max_std})")
    print("\nThe deployed model (models/model.joblib) is unchanged: the test set has already "
          "been used, so any new C is a recommendation for a future model version.")
