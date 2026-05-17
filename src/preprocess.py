from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/customer_churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/customer_churn_clean.csv")

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "InternetService", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}. Run src/generate_data.py first.")
    return pd.read_csv(path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    numeric_columns = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna()
    df = df[df["tenure"] >= 0]
    df = df[df["MonthlyCharges"] >= 0]
    df = df[df["TotalCharges"] >= 0]
    df = df[df["Churn"].isin([0, 1])]

    return df.reset_index(drop=True)

def main() -> None:
    df = load_raw_data()
    cleaned = preprocess_data(df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed dataset at {PROCESSED_DATA_PATH}, shape={cleaned.shape}")

if __name__ == "__main__":
    main()
