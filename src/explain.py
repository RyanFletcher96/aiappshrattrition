"""Explain the model globally (which features matter overall) and locally (why one person was flagged)."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from config_loader import load_config
from features import add_engineered_features, load_features, split_data

def transformed_frame(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Run X through every pipeline step except the model, keeping readable feature names."""
    values = pipe[:-1].transform(X)
    names = pipe.named_steps["prep"].get_feature_names_out()
    # Drop the "num__" / "cat__" prefixes the ColumnTransformer adds
    names = [name.split("__", 1)[1] for name in names]
    return pd.DataFrame(values, columns=names, index=X.index)

def readable_values(X: pd.DataFrame, Xt: pd.DataFrame) -> pd.DataFrame:
    """Original (unscaled) values for display, so plots show e.g. 2 years not -1.07."""
    raw = add_engineered_features(X)
    display = Xt.copy()
    for col in display.columns:
        if col in raw.columns:
            display[col] = raw[col].values
    return display

if __name__ == "__main__":
    cfg = load_config()
    figures_dir = cfg["paths"]["figures"]

    X, y = load_features(cfg["paths"]["processed_data"])
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=cfg["split"]["test_size"], seed=cfg["split"]["seed"]
    )
    saved = joblib.load(cfg["paths"]["model"])
    pipe, threshold, model_name = saved["pipeline"], saved["threshold"], saved["model_name"]
    model = pipe.named_steps["model"]

    Xt_train = transformed_frame(pipe, X_train)
    Xt_test = transformed_frame(pipe, X_test)

    # LinearExplainer suits logistic regression; the whole training set is the background
    masker = shap.maskers.Independent(Xt_train, max_samples=len(Xt_train))
    explainer = shap.LinearExplainer(model, masker)
    shap_values = explainer(Xt_test)
    # Explain with scaled values, but display the original ones
    shap_values.data = readable_values(X_test, Xt_test).values

    # 1. Global: which features push risk up or down across the whole test set
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("SHAP summary: feature impact on attrition risk (test set)")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    importance = pd.DataFrame({
        "feature": Xt_test.columns,
        "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.round(4).to_csv(figures_dir / "shap_importance.csv", index=False)
    print("Top 10 features by mean |SHAP| (test set):")
    print(importance.head(10).round(3).to_string(index=False))

    # Coefficients as odds ratios: per 1 standard deviation for numeric features,
    # or for having the category vs not for one-hot features
    coefficients = pd.DataFrame({
        "feature": Xt_train.columns,
        "coefficient": model.coef_[0],
        "odds_ratio": np.exp(model.coef_[0]),
    }).sort_values("coefficient", ascending=False)
    coefficients.round(3).to_csv(figures_dir / "lr_coefficients.csv", index=False)

    # 2. Local: why the highest-risk flagged employee in the test set was flagged
    proba = pipe.predict_proba(X_test)[:, 1]
    position = int(np.argmax(proba))
    print(f"\nLocal explanation for test employee at position {position}: "
          f"risk score {proba[position]:.3f} (threshold {threshold:.3f}), "
          f"actually {'left' if y_test.iloc[position] == 1 else 'stayed'}")
    shap.plots.waterfall(shap_values[position], max_display=12, show=False)
    plt.title("Why this employee was flagged (SHAP, log-odds)")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved SHAP plots and tables to {figures_dir}")
