"""FastAPI proof of concept: score employees for attrition risk over HTTP.

Run from the project root with:
    python -m uvicorn api:app --app-dir src --reload
Then open http://127.0.0.1:8000/docs
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from config_loader import load_config
from preprocess import VALUE_RANGES
from score_batch import model_version, score_employees

MODEL = {}

# Uvicorn only configures its own loggers, so set a format here to see this module's messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Lineage fields /model-info may show. Listed explicitly so nothing else in the saved
# metadata (such as file paths) is exposed by accident.
LINEAGE_KEYS = [
    "trained_at", "git_commit", "python_version", "sklearn_version", "pandas_version",
    "training_rows", "cv_pr_auc_mean", "cv_pr_auc_std", "threshold_recall",
    "threshold_precision", "random_seed",
]

def in_range(column: str):
    """Pydantic Field using the same limits as preprocess.VALUE_RANGES (one source of truth)."""
    low, high = VALUE_RANGES[column]
    return Field(ge=low, le=high)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the saved model once when the API starts."""
    cfg = load_config()
    MODEL.update(joblib.load(cfg["paths"]["model"]))
    logger.info("Loaded model %s (version %s)", MODEL["model_name"], model_version(MODEL))
    yield
    MODEL.clear()

app = FastAPI(
    title="HR Attrition Risk API",
    description="Proof of concept. Scores support a human retention conversation; "
                "they must not be used for recruitment, redundancy or performance decisions.",
    version="1.0",
    lifespan=lifespan,
)

class Employee(BaseModel):
    """One employee's model inputs. Gender and MaritalStatus are not accepted."""

    # Reject unexpected fields, so protected attributes can't be sent by mistake
    model_config = ConfigDict(extra="forbid")

    EmployeeNumber: int = Field(ge=1)
    Age: int = in_range("Age")
    BusinessTravel: Literal["Non-Travel", "Travel_Frequently", "Travel_Rarely"]
    DailyRate: int = Field(ge=0)
    Department: Literal["Human Resources", "Research & Development", "Sales"]
    DistanceFromHome: int = Field(ge=0)
    Education: int = in_range("Education")
    EducationField: Literal[
        "Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"
    ]
    EnvironmentSatisfaction: int = in_range("EnvironmentSatisfaction")
    HourlyRate: int = Field(ge=0)
    JobInvolvement: int = in_range("JobInvolvement")
    JobLevel: int = in_range("JobLevel")
    JobRole: Literal[
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
        "Manufacturing Director", "Research Director", "Research Scientist",
        "Sales Executive", "Sales Representative",
    ]
    JobSatisfaction: int = in_range("JobSatisfaction")
    MonthlyIncome: int = in_range("MonthlyIncome")
    MonthlyRate: int = Field(ge=0)
    NumCompaniesWorked: int = Field(ge=0)
    OverTime: Literal["Yes", "No"]
    PercentSalaryHike: int = Field(ge=0)
    PerformanceRating: int = in_range("PerformanceRating")
    RelationshipSatisfaction: int = in_range("RelationshipSatisfaction")
    StockOptionLevel: int = in_range("StockOptionLevel")
    TotalWorkingYears: int = Field(ge=0)
    TrainingTimesLastYear: int = Field(ge=0)
    WorkLifeBalance: int = in_range("WorkLifeBalance")
    YearsAtCompany: int = Field(ge=0)
    YearsInCurrentRole: int = Field(ge=0)
    YearsSinceLastPromotion: int = Field(ge=0)
    YearsWithCurrManager: int = Field(ge=0)

class PredictRequest(BaseModel):
    employees: list[Employee] = Field(min_length=1, max_length=1000)

class Prediction(BaseModel):
    EmployeeNumber: int
    risk_score: float
    flagged: bool

class PredictResponse(BaseModel):
    request_id: str
    model_name: str
    model_version: str
    threshold: float
    note: str
    predictions: list[Prediction]

@app.get("/health")
def health() -> dict:
    """Check the API is running and the model is loaded."""
    return {"status": "ok", "model_loaded": "pipeline" in MODEL}

@app.get("/model-info")
def model_info() -> dict:
    """Show which model is serving and how it was built (safe metadata only, no file paths)."""
    if "pipeline" not in MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Models saved before metadata was added still load, they just show as "unversioned"
    metadata = MODEL.get("metadata", {})
    info = {
        "model_name": MODEL["model_name"],
        "threshold": round(MODEL["threshold"], 4),
        "version": model_version(MODEL),
    }
    for key in LINEAGE_KEYS:
        info[key] = metadata.get(key)
    return info

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Score a batch of employees. Higher risk_score means higher attrition risk."""
    if "pipeline" not in MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # The request ID goes in the response and the log, so one call can be traced end to end
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    X = pd.DataFrame([employee.model_dump() for employee in request.employees])
    results = score_employees(MODEL, X)
    predictions = [
        Prediction(
            EmployeeNumber=int(emp_id),
            risk_score=float(row.risk_score),
            flagged=bool(row.flagged),
        )
        for emp_id, row in zip(X["EmployeeNumber"], results.itertuples())
    ]
    latency_ms = (time.perf_counter() - start) * 1000

    # Counts and timing only: employee field values and scores are never logged
    logger.info(
        "request_id=%s employees=%d flagged=%d latency_ms=%.1f",
        request_id, len(predictions), int(results["flagged"].sum()), latency_ms,
    )
    return PredictResponse(
        request_id=request_id,
        model_name=MODEL["model_name"],
        model_version=model_version(MODEL),
        threshold=round(MODEL["threshold"], 4),
        note="risk_score ranks employees by risk and is not a calibrated probability. "
             "Flags support, and do not replace, a human retention conversation.",
        predictions=predictions,
    )
