"""CI quality gate: sanity-check the committed results files and list what needs human review."""

import json
import os
import sys
from pathlib import Path

from config_loader import load_config

# Values that are rates or probabilities, so they must sit between 0 and 1
RATE_METRICS = ["pr_auc", "roc_auc", "brier", "recall", "precision", "f1"]
REQUIRED_METRICS = ["pr_auc", "recall", "precision"]


def is_number(value) -> bool:
    """Return True for ints and floats, but not for bools (which Python treats as ints)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_json(path: Path) -> tuple[dict | None, str | None]:
    """Read a JSON results file, returning (data, None) or (None, error message)."""
    if not path.exists():
        return None, f"{path.name} is missing (run src/evaluate.py and src/fairness.py)"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return None, f"{path.name} could not be parsed: {error}"
    if not isinstance(data, dict):
        return None, f"{path.name} should contain a JSON object"
    return data, None


def check_metrics(results: dict, criteria: dict) -> tuple[list[str], list[str]]:
    """Check metrics.json: broken values block CI, missed targets go to human review."""
    blocking, review = [], []

    metrics = results.get("metrics")
    if not isinstance(metrics, dict):
        return ["metrics.json has no 'metrics' section"], review

    missing = [key for key in REQUIRED_METRICS if not is_number(metrics.get(key))]
    for key in ["threshold", "baseline_pr_auc"]:
        if not is_number(results.get(key)):
            missing.append(key)
    if missing:
        # Without these values the remaining checks would be meaningless
        return [f"metrics.json is missing or has non-numeric values for: {', '.join(missing)}"], review

    # Rates outside [0, 1] mean the evaluation code is broken, not that the model is weak
    rates = {key: metrics[key] for key in RATE_METRICS if key in metrics}
    rates["baseline_pr_auc"] = results["baseline_pr_auc"]
    for key, value in rates.items():
        if not (is_number(value) and 0 <= value <= 1):
            blocking.append(f"{key} = {value} is not a rate between 0 and 1")

    threshold = results["threshold"]
    if not 0 < threshold < 1:
        blocking.append(f"threshold = {threshold} should be strictly between 0 and 1")

    # A model that cannot beat "always predict the majority" should never get this far
    if metrics["pr_auc"] <= results["baseline_pr_auc"]:
        blocking.append(
            f"PR-AUC {metrics['pr_auc']:.3f} does not beat the dummy baseline "
            f"{results['baseline_pr_auc']:.3f}"
        )

    # Missed business targets are a judgement call, so they are reported rather than blocking
    if metrics["recall"] < criteria["min_recall"]:
        review.append(f"Test recall {metrics['recall']:.3f} is below the target {criteria['min_recall']:.2f}")
    if metrics["precision"] < criteria["min_precision"]:
        review.append(
            f"Test precision {metrics['precision']:.3f} is below the target {criteria['min_precision']:.2f}"
        )

    return blocking, review


def check_fairness(fairness: dict, criteria: dict) -> tuple[list[str], list[str]]:
    """Check fairness.json: broken values block CI, large gaps go to human review."""
    blocking, review = [], []

    gaps = fairness.get("gaps")
    if not isinstance(gaps, dict) or not gaps:
        return ["fairness.json has no 'gaps' section"], review

    for name, gap in gaps.items():
        # A gap is a difference between two rates, so it must also be between 0 and 1
        if not (is_number(gap) and 0 <= gap <= 1):
            blocking.append(f"{name} = {gap} is not a gap between 0 and 1")
        elif gap > criteria["max_fairness_gap"]:
            review.append(f"{name} = {gap:.3f} is above the limit {criteria['max_fairness_gap']:.2f}")

    return blocking, review


def run_gate(metrics: dict, fairness: dict, criteria: dict) -> tuple[list[str], list[str]]:
    """Return (blocking failures, items needing human review) for the saved results."""
    metric_blocking, metric_review = check_metrics(metrics, criteria)
    fairness_blocking, fairness_review = check_fairness(fairness, criteria)
    return metric_blocking + fairness_blocking, metric_review + fairness_review


def write_summary(blocking: list[str], review: list[str], summary_path: str) -> None:
    """Append a short markdown table to the GitHub Actions job summary."""
    lines = ["## Model quality gate", "", "| Type | Finding |", "| --- | --- |"]
    lines += [f"| Blocking | {item} |" for item in blocking]
    lines += [f"| Review required | {item} |" for item in review]
    if not blocking and not review:
        lines.append("| OK | No findings |")
    lines += [
        "",
        "A green CI run is not approval: review items need documented human sign-off "
        "(e.g. in the model card) before any deployment decision.",
        "",
    ]
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    """Load the committed results, print the gate report and return the exit code."""
    cfg = load_config()
    metrics_path = cfg["paths"]["metrics"]
    # Same location src/fairness.py saves to
    fairness_path = metrics_path.parent / "fairness.json"

    metrics, metrics_error = load_json(metrics_path)
    fairness, fairness_error = load_json(fairness_path)
    load_errors = [e for e in [metrics_error, fairness_error] if e]

    if load_errors:
        blocking, review = load_errors, []
    else:
        blocking, review = run_gate(metrics, fairness, cfg["success_criteria"])

    print("=== Model quality gate ===")
    print(f"\nBlocking failures: {len(blocking)}")
    for item in blocking:
        print(f"  FAIL   {item}")
    print(f"\nReview required: {len(review)}")
    for item in review:
        print(f"  REVIEW {item}")
    if review:
        print(
            "\nThese review items do not fail CI, but they need documented human sign-off"
            "\n(e.g. in the model card) before any deployment decision."
        )
    print("\nCI passing only shows the code and results files are sane.")
    print("It does not prove the model is fair, the metric is right, or that deployment is justified.")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        write_summary(blocking, review, summary_path)

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
