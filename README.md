# Automated Customer Churn Prediction MLOps Pipeline

A complete MLOps assessment artefact using GitHub Actions, Flask, Docker, automated testing, Google VM deployment, and scheduled continuous training.

## Pipeline Stages

1. Data acquisition and preprocessing
2. Model training and testing
3. Model deployment
4. Continuous integration
5. Continuous delivery
6. Continuous training

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

python src/generate_data.py
python src/preprocess.py
python src/train.py
python src/evaluate.py
python app/app.py
```

Open:

```text
http://127.0.0.1:5000/health
```

## Test Prediction

```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"gender\":\"Female\",\"SeniorCitizen\":0,\"Partner\":\"Yes\",\"Dependents\":\"No\",\"tenure\":12,\"PhoneService\":\"Yes\",\"InternetService\":\"Fiber optic\",\"Contract\":\"Month-to-month\",\"PaperlessBilling\":\"Yes\",\"PaymentMethod\":\"Electronic check\",\"MonthlyCharges\":85.5,\"TotalCharges\":1026.0}"
```

## Run Tests

```bash
pytest
```

## Docker

```bash
docker build -t churn-api .
docker run -p 5000:5000 churn-api
```

## GitHub Secrets Needed for Deployment

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
GCP_VM_HOST
GCP_VM_USER
GCP_VM_SSH_KEY
```

## Branching Strategy

```text
feature/* -> develop -> main
```

- `main`: production branch, triggers deployment
- `develop`: integration branch
- `feature/*`: new development
- `hotfix/*`: urgent production fixes
