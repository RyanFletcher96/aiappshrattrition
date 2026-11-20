import pandas as pd
from pathlib import Path
from config_loader import load_config

EXPECTED_COLUMNS = [
    "Age",
    "Attrition",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "Over18",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

DROP_COLUMNS = [
    "EmployeeCount",
    "StandardHours",
    "Over18",
    "EmployeeNumber",
]

TARGET_COLUMN = "Attrition"

VALUE_RANGES = {
    "Age": (16, 80),
    "MonthlyIncome": (1, None),
    "Education": (1, 5),
    "JobLevel": (1, 5),
    "StockOptionLevel": (0, 3),
    "EnvironmentSatisfaction": (1, 4),
    "JobInvolvement": (1, 4),
    "JobSatisfaction": (1, 4),
    "PerformanceRating": (1, 4),
    "RelationshipSatisfaction": (1, 4),
    "WorkLifeBalance": (1, 4),
}

REQUIRED_NON_NULL = [
]

for c in EXPECTED_COLUMNS:
    if c not in DROP_COLUMNS:
        REQUIRED_NON_NULL.append(c)

def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw data from a CSV file and remove index columns."""
    df = pd.read_csv(filepath)
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]

    # Handles index columns if a CSV is saved without index=False
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"Dropped unnamed columns: {unnamed_cols}")
    return df

def validate_columns(df: pd.DataFrame) -> None:
    """Raise ValueError if the data does not match the expected schema."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    unexpected = set(df.columns) - set(EXPECTED_COLUMNS)
    if unexpected:
        raise ValueError(f"Unexpected columns found: {sorted(unexpected)}")

    nulls = df[REQUIRED_NON_NULL].isnull().sum()
    if nulls.any():
        raise ValueError(f"Nulls in required columns:\n{nulls[nulls > 0]}")

    bad_target = set(df[TARGET_COLUMN].unique()) - {"Yes", "No"}
    if bad_target:
        raise ValueError(f"Invalid {TARGET_COLUMN} values: {sorted(bad_target)}")

    for col, (low, high) in VALUE_RANGES.items():
        if low is not None and (df[col] < low).any():
            raise ValueError(f"Values in {col} below minimum {low}")
        if high is not None and (df[col] > high).any():
            raise ValueError(f"Values in {col} above maximum {high}")
        
def preprocess(input_path: str, output_path: str) -> pd.DataFrame:

    df = load_raw_data(input_path)
    raw_shape = df.shape
    validate_columns(df)
    df = df.drop(columns=DROP_COLUMNS)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Raw: {raw_shape[0]} rows x {raw_shape[1]} cols -> " 
          f"Cleaned: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"Preprocessed data saved to {output_path}")
    return df

if __name__ == "__main__":
    cfg = load_config()
    preprocess(
        input_path=cfg["paths"]["raw_data"],
        output_path=cfg["paths"]["processed_data"]
    )