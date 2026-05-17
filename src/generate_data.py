from pathlib import Path
import numpy as np
import pandas as pd

RAW_DATA_PATH = Path("data/raw/customer_churn.csv")

def generate_churn_data(n_rows: int = 2000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    gender = rng.choice(["Male", "Female"], n_rows)
    senior = rng.choice([0, 1], n_rows, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n_rows)
    dependents = rng.choice(["Yes", "No"], n_rows, p=[0.35, 0.65])
    tenure = rng.integers(1, 73, n_rows)
    phone_service = rng.choice(["Yes", "No"], n_rows, p=[0.9, 0.1])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n_rows, p=[0.38, 0.44, 0.18])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.25, 0.20])
    paperless = rng.choice(["Yes", "No"], n_rows, p=[0.6, 0.4])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_rows,
        p=[0.35, 0.20, 0.25, 0.20],
    )

    monthly = (
        25
        + (internet_service == "DSL") * 25
        + (internet_service == "Fiber optic") * 45
        + senior * 5
        + rng.normal(0, 8, n_rows)
    ).clip(18, 120).round(2)

    total = (monthly * tenure + rng.normal(0, 50, n_rows)).clip(0, None).round(2)

    logit = (
        -1.6
        + (contract == "Month-to-month") * 1.4
        + (internet_service == "Fiber optic") * 0.7
        + (payment == "Electronic check") * 0.5
        + (paperless == "Yes") * 0.2
        + senior * 0.25
        - tenure * 0.035
        + (monthly > 80) * 0.35
        - (dependents == "Yes") * 0.2
    )

    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability, n_rows)

    return pd.DataFrame(
        {
            "customerID": [f"CUST-{i:05d}" for i in range(n_rows)],
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "InternetService": internet_service,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": churn,
        }
    )

def main() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_churn_data()
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Generated dataset at {RAW_DATA_PATH}, shape={df.shape}")

if __name__ == "__main__":
    main()
