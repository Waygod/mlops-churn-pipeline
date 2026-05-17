from src.generate_data import generate_churn_data
from src.preprocess import preprocess_data

def test_preprocessing_removes_null_values():
    df = generate_churn_data(n_rows=100, random_state=1)
    df.loc[0, "MonthlyCharges"] = None
    cleaned = preprocess_data(df)
    assert cleaned.isnull().sum().sum() == 0

def test_preprocessing_keeps_required_target():
    df = generate_churn_data(n_rows=100, random_state=2)
    cleaned = preprocess_data(df)
    assert "Churn" in cleaned.columns
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
