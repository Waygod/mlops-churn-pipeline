from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROCESSED_DATA_PATH = Path("data/processed/customer_churn_clean.csv")
MODEL_PATH = Path("app/model.pkl")
METRICS_PATH = Path("app/metrics.json")

TARGET = "Churn"

CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "InternetService",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

def load_training_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}. Run preprocessing first.")
    return pd.read_csv(path)

def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_split=8,
        random_state=42,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_score": round(float(f1_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "model_type": "RandomForestClassifier",
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "features": FEATURES,
    }

    return model, metrics

def main() -> None:
    df = load_training_data()
    model, metrics = train_model(df)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metrics}, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
