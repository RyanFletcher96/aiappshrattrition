"""
test_quality_gate.py, Unit tests for the CI quality gate.
"""

import sys
import os

# Allow the test runner to find src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quality_gate import run_gate


def make_criteria():
    """Create success criteria matching the shape of config.yaml."""
    return {"min_recall": 0.70, "min_precision": 0.40, "max_cv_pr_auc_std": 0.05, "max_fairness_gap": 0.10}


def make_metrics(pr_auc=0.60, recall=0.75, precision=0.45, threshold=0.5):
    """Create a metrics.json-style dict that meets every target by default."""
    return {
        "model_name": "logistic_regression",
        "threshold": threshold,
        "baseline_pr_auc": 0.16,
        "metrics": {"pr_auc": pr_auc, "roc_auc": 0.80, "brier": 0.15,
                    "recall": recall, "precision": precision, "f1": 0.55},
    }


def make_fairness(age_gap=0.05):
    """Create a fairness.json-style dict with small gaps by default."""
    return {
        "model_name": "logistic_regression",
        "max_gap": 0.1,
        "gaps": {"test_Gender_recall_gap": 0.04, "test_AgeBand_recall_gap": age_gap},
    }


def test_good_results_pass_cleanly():
    """Results that meet every target should give no blocking failures and nothing to review."""
    blocking, review = run_gate(make_metrics(), make_fairness(), make_criteria())

    assert blocking == [], f"Unexpected blocking failures: {blocking}"
    assert review == [], f"Unexpected review items: {review}"


def test_not_beating_baseline_blocks():
    """A PR-AUC no better than the dummy baseline should block CI."""
    blocking, _ = run_gate(make_metrics(pr_auc=0.16), make_fairness(), make_criteria())

    assert any("dummy baseline" in item for item in blocking), f"Baseline check missing: {blocking}"


def test_invalid_threshold_blocks():
    """A threshold outside (0, 1) means something is broken, so CI should fail."""
    blocking, _ = run_gate(make_metrics(threshold=1.5), make_fairness(), make_criteria())

    assert any("threshold" in item for item in blocking), f"Threshold check missing: {blocking}"


def test_missed_targets_need_review_but_do_not_block():
    """Missed recall and a large fairness gap are for humans to judge, not for CI to block."""
    blocking, review = run_gate(make_metrics(recall=0.55), make_fairness(age_gap=0.3), make_criteria())

    assert blocking == [], f"Missed targets should not block CI: {blocking}"
    assert any("recall" in item for item in review), f"Recall not flagged for review: {review}"
    assert any("AgeBand" in item for item in review), f"Fairness gap not flagged for review: {review}"
