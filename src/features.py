"""Training-side feature preparation: target, split, engineered features and the unfitted pipeline."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from config_loader import load_config

TARGET_COLUMN = "Attrition"

NUMERIC_COLUMNS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

ENGINEERED_COLUMNS = [
    "TenureRatio",
    "PromotionStagnation",
    "SatisfactionMean",
]

ORDINAL_COLUMNS = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance",
]

NOMINAL_COLUMNS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "OverTime",
]

# Kept in X for fairness evaluation, but excluded from model inputs by default
PROTECTED_COLUMNS = [
    "Gender",
    "MaritalStatus",
]

SATISFACTION_COLUMNS = [
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]

def load_features(filepath) -> tuple[pd.DataFrame, pd.Series]:
    """Load the cleaned CSV and split it into features X and a 0/1 target y."""
    df = pd.read_csv(filepath)
    y = (df[TARGET_COLUMN] == "Yes").astype(int)
    X = df.drop(columns=[TARGET_COLUMN])
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int):
    """Stratified train/test split so both sides keep the same attrition rate."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-wise engineered features. Module-level so the pipeline can be pickled."""
    df = df.copy()
    # +1 avoids dividing by zero (some employees have 0 working years or 0 years at the company)
    df["TenureRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)
    df["PromotionStagnation"] = df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
    df["SatisfactionMean"] = df[SATISFACTION_COLUMNS].mean(axis=1)
    return df

def age_band(age: pd.Series, bins: list, labels: list) -> pd.Series:
    """Group ages into bands for fairness evaluation (not a model input)."""
    return pd.cut(age, bins=bins, labels=labels, right=False)

def build_preprocessor(include_protected: bool = False) -> ColumnTransformer:
    """Scale numeric/ordinal columns and one-hot encode nominal ones. Unfitted."""
    nominal = NOMINAL_COLUMNS + PROTECTED_COLUMNS if include_protected else NOMINAL_COLUMNS
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS + ENGINEERED_COLUMNS + ORDINAL_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False),
                nominal,
            ),
        ],
        # Anything not listed (e.g. Gender, MaritalStatus) never reaches the model
        remainder="drop",
    )

def build_pipeline(model, include_protected: bool = False) -> Pipeline:
    """Chain feature engineering, preprocessing and a model. Fitted only on training data."""
    return Pipeline([
        ("features", FunctionTransformer(add_engineered_features)),
        ("prep", build_preprocessor(include_protected)),
        ("model", model),
    ])

if __name__ == "__main__":
    cfg = load_config()
    X, y = load_features(cfg["paths"]["processed_data"])
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=cfg["split"]["test_size"], seed=cfg["split"]["seed"]
    )
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Attrition rate - train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    for include_protected in (False, True):
        pipe = build_pipeline(LogisticRegression(max_iter=1000), include_protected)
        pipe.fit(X_train, y_train)
        n_features = len(pipe.named_steps["prep"].get_feature_names_out())
        print(f"include_protected={include_protected}: {n_features} model input features")
