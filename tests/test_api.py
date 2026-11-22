"""
test_api.py, Unit tests for the HR attrition FastAPI proof of concept.
"""

import pytest
import sys
import os
import uuid

# Allow the test runner to find src/ without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient
from config_loader import load_config
from api import app

# The trained model is not in Git (models/*.joblib is ignored), so skip if it hasn't been built
MODEL_PATH = load_config()["paths"]["model"]
pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Run src/train.py first to create the model"
)

# First employee in the raw dataset, without the target or protected attributes
VALID_EMPLOYEE = {
    "EmployeeNumber": 1,
    "Age": 41,
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 1102,
    "Department": "Sales",
    "DistanceFromHome": 1,
    "Education": 2,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 2,
    "HourlyRate": 94,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobRole": "Sales Executive",
    "JobSatisfaction": 4,
    "MonthlyIncome": 5993,
    "MonthlyRate": 19479,
    "NumCompaniesWorked": 8,
    "OverTime": "Yes",
    "PercentSalaryHike": 11,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 1,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 8,
    "TrainingTimesLastYear": 0,
    "WorkLifeBalance": 1,
    "YearsAtCompany": 6,
    "YearsInCurrentRole": 4,
    "YearsSinceLastPromotion": 0,
    "YearsWithCurrManager": 5,
}


def make_client():
    """Create a test client. Using it in a 'with' block loads the model."""
    return TestClient(app)


def test_health_reports_model_loaded():
    """GET /health should return 200 and confirm the model is loaded."""
    with make_client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_valid_employee_gets_risk_score():
    """A valid employee should get a risk score between 0 and 1."""
    with make_client() as client:
        response = client.post("/predict", json={"employees": [VALID_EMPLOYEE]})
    assert response.status_code == 200, response.text
    prediction = response.json()["predictions"][0]
    assert prediction["EmployeeNumber"] == 1
    assert 0 <= prediction["risk_score"] <= 1, "Risk score outside 0-1"
    assert isinstance(prediction["flagged"], bool)


def test_out_of_range_age_is_rejected():
    """Age=200 should be rejected with a 422 validation error."""
    employee = {**VALID_EMPLOYEE, "Age": 200}
    with make_client() as client:
        response = client.post("/predict", json={"employees": [employee]})
    assert response.status_code == 422


def test_unknown_category_is_rejected():
    """A JobRole the model has never seen should be rejected."""
    employee = {**VALID_EMPLOYEE, "JobRole": "Astronaut"}
    with make_client() as client:
        response = client.post("/predict", json={"employees": [employee]})
    assert response.status_code == 422


def test_protected_attribute_is_rejected():
    """Sending Gender should be rejected, because the API must not collect it."""
    employee = {**VALID_EMPLOYEE, "Gender": "Female"}
    with make_client() as client:
        response = client.post("/predict", json={"employees": [employee]})
    assert response.status_code == 422


def test_model_info_returns_version_metadata():
    """GET /model-info should name the model, its threshold and a version, without file paths."""
    with make_client() as client:
        response = client.get("/model-info")
    assert response.status_code == 200, response.text
    info = response.json()
    assert info["model_name"], "model_name missing"
    assert 0 < info["threshold"] < 1, "Threshold outside 0-1"
    assert isinstance(info["version"], str) and info["version"], "version missing"
    assert "trained_at" in info and "git_commit" in info, "Lineage fields missing"
    assert ".joblib" not in response.text, "Model file path should not be exposed"


def test_predict_response_includes_request_id_and_version():
    """Each /predict response should carry a unique request ID and the model version."""
    with make_client() as client:
        first = client.post("/predict", json={"employees": [VALID_EMPLOYEE]}).json()
        second = client.post("/predict", json={"employees": [VALID_EMPLOYEE]}).json()
        version = client.get("/model-info").json()["version"]
    assert str(uuid.UUID(first["request_id"])) == first["request_id"], "request_id is not a UUID"
    assert first["request_id"] != second["request_id"], "request_id should differ per call"
    assert first["model_version"] == version, "model_version should match /model-info"
