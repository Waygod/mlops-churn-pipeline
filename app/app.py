from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

REQUIRED_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "InternetService", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

app = Flask(__name__)

def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run `python src/train.py` before starting the API."
        )
    return joblib.load(MODEL_PATH)

MODEL_BUNDLE = load_model_bundle()
MODEL = MODEL_BUNDLE["model"]
METADATA = MODEL_BUNDLE.get("metadata", {})

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Customer Churn Prediction API",
        "endpoints": ["/health", "/predict"]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_type": METADATA.get("model_type", "unknown"),
        "model_version": METADATA.get("model_version", "unknown")
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)

        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400

        missing = [feature for feature in REQUIRED_FEATURES if feature not in payload]
        if missing:
            return jsonify({"error": "Missing required features.", "missing": missing}), 400

        input_df = pd.DataFrame([payload], columns=REQUIRED_FEATURES)
        prediction = int(MODEL.predict(input_df)[0])
        probability = float(MODEL.predict_proba(input_df)[0][1])

        return jsonify({
            "prediction": prediction,
            "prediction_label": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(probability, 4),
            "model_version": METADATA.get("model_version", "unknown")
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
