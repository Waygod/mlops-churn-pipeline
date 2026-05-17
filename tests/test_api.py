import json
import subprocess
import sys
from pathlib import Path
import pytest

MODEL_PATH = Path("app/model.pkl")

@pytest.fixture(scope="session", autouse=True)
def prepare_model():
    if not MODEL_PATH.exists():
        subprocess.run([sys.executable, "src/generate_data.py"], check=True)
        subprocess.run([sys.executable, "src/preprocess.py"], check=True)
        subprocess.run([sys.executable, "src/train.py"], check=True)

def test_health_endpoint():
    from app.app import app
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"

def test_prediction_endpoint_returns_prediction():
    from app.app import app
    client = app.test_client()
    sample = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 1026.0
    }

    response = client.post("/predict", data=json.dumps(sample), content_type="application/json")
    body = response.get_json()

    assert response.status_code == 200
    assert "prediction" in body
    assert "churn_probability" in body
    assert body["prediction"] in [0, 1]
