# Automated Customer Churn Prediction MLOps Pipeline

## Project Overview
This project implements an end-to-end MLOps pipeline for customer churn prediction. It uses a synthetic telecom-style churn dataset, trains a `RandomForestClassifier`, serves predictions through a Flask REST API, containerizes the application using Docker, and deploys the API to a local Kind Kubernetes cluster. GitHub Actions workflows automate testing, model training/evaluation, Docker build validation, and scheduled retraining.

GitHub repository: https://github.com/Waygod/mlops-churn-pipeline

## Objectives
- Build a complete machine learning pipeline from data generation to deployment.
- Automate testing, training, evaluation, and Docker validation using GitHub Actions.
- Serve the trained model through a Flask API.
- Containerize the API using Docker.
- Deploy the containerized Flask API to a Kind Kubernetes cluster.
- Demonstrate CI, CD-ready deployment validation, and CT concepts.

## Technology Stack
| Component | Technology |
|---|---|
| Programming language | Python |
| ML model | RandomForestClassifier |
| API framework | Flask |
| Production server | Gunicorn |
| Containerization | Docker |
| Orchestration | Kind Kubernetes |
| Automation | GitHub Actions |
| Testing | Pytest |
| Version control | Git/GitHub |

## Project Structure
```text
mlops-churn-pipeline/
├── app/
│   ├── app.py
│   ├── model.pkl
│   └── metrics.json
├── data/
│   └── generated dataset files
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── src/
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── unit test files
├── .github/workflows/
│   └── CI/CD/CT workflow files
├── Dockerfile
├── requirements.txt
└── README.md
```

## System Architecture
```text
Synthetic Data Generation
        ↓
Data Preprocessing
        ↓
Model Training - RandomForestClassifier
        ↓
Model Evaluation - accuracy, F1, ROC-AUC
        ↓
Model Artifact - model.pkl
        ↓
Flask REST API - /health and /predict
        ↓
Docker Container
        ↓
Kind Kubernetes Deployment
        ↓
GitHub Actions CI/CD/CT Workflows
```

## Pipeline Stages

### 1. Data Acquisition and Preprocessing
Synthetic customer churn data is generated using `src/generate_data.py`. The preprocessing stage prepares categorical and numerical features for model training.

### 2. Model Training and Testing
The model is trained using `src/train.py`. The chosen model is `RandomForestClassifier`, which is suitable for tabular classification tasks and can handle nonlinear relationships between customer attributes and churn behaviour.

### 3. Model Evaluation
The model is evaluated using `src/evaluate.py`. The key metrics are:

| Metric | Approximate Value |
|---|---:|
| Accuracy | 0.7675 |
| F1-score | 0.4973 |
| ROC-AUC | 0.746 |

The F1-score is important because churn prediction can involve class imbalance, where accuracy alone may not fully represent positive-class prediction quality.

### 4. Flask API Deployment
The trained model is served through a Flask API.

Available endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API overview |
| `/health` | GET | Health check |
| `/predict` | POST | Returns churn prediction |

### 5. Docker Containerization
The Flask API is packaged into a Docker image. This ensures the application runs consistently across different environments.

### 6. Kubernetes Deployment
The Kubernetes deployment was tested using Kind on a local Windows machine with Docker Desktop. After loading the Docker image into the Kind cluster, the Flask API pod reached `Running` status and the `/predict` endpoint returned a churn prediction through port-forwarding.
The Docker image is deployed to a Kind Kubernetes cluster using:

- `k8s/deployment.yaml`
- `k8s/service.yaml`

The Kubernetes Deployment manages the Flask API pod. The Kubernetes Service exposes the application using NodePort and port-forwarding.

### 7. Continuous Integration
During development, the CI workflow initially failed because the `src` package was not detected during pytest execution in the GitHub runner. This was fixed by setting `PYTHONPATH: .` in the workflow environment.
The CI workflow runs automatically on push and pull request. It performs:

- dependency installation
- unit testing with pytest
- data generation
- model training
- model evaluation
- Docker image build validation

### 8. Continuous Delivery
The CD workflow validates deployment readiness by building the Docker image and preparing the application for container-based deployment. The final deployment target used in this project is a Kind Kubernetes cluster.

### 9. Continuous Training
The CT workflow supports periodic or manually triggered retraining. This demonstrates how the model can be refreshed as data changes.

## Branching Strategy
The project uses a simple branching strategy suitable for an individual MLOps project:

```text
feature/* → develop → main
```

- `feature/*`: used for isolated changes such as API updates, tests, or workflow edits.
- `develop`: used for integration testing before production-ready merge.
- `main`: stable branch used for final submission and deployment.

Pull requests should be used before merging into `main`, and GitHub Actions should pass before accepting changes.

## How to Run Locally

### 1. Create and activate virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Run ML pipeline
```powershell
python src/generate_data.py
python src/train.py
python src/evaluate.py
```

### 4. Run Flask API locally
```powershell
python app/app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Docker Deployment

### Build image
```powershell
docker build -t churn-api .
```

### Run container
```powershell
docker run -p 5000:5000 churn-api
```

## Kubernetes Deployment with Kind

### Create cluster
```powershell
kind create cluster --name churn-cluster
```

### Build Docker image
```powershell
docker build -t churn-api:1.0 .
```

### Load image into Kind
```powershell
kind load docker-image churn-api:1.0 --name churn-cluster
```

### Apply Kubernetes manifests
```powershell
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

### Check deployment
```powershell
kubectl get pods
kubectl get deployments
kubectl get services
```

Expected output should show:

```text
churn-api pod: Running
churn-api deployment: 1/1 available
churn-api-service: NodePort 80:30008/TCP
```

### Port forward service
```powershell
kubectl port-forward service/churn-api-service 5000:80
```

## API Usage

### Prediction request
```powershell
$body = @{
    gender = "Male"
    Partner = "Yes"
    Dependents = "No"
    PhoneService = "Yes"
    InternetService = "Fiber optic"
    Contract = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    SeniorCitizen = 0
    tenure = 12
    MonthlyCharges = 70.5
    TotalCharges = 846
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:5000/predict" `
    -Method POST `
    -Body $body `
    -ContentType "application/json"
```

Example output:

```text
churn_probability: 0.8044
model_version: 20260517121353
prediction: 1
prediction_label: Churn
```

## Unit Testing
Run:

```powershell
pytest -q
```

The project includes at least two unit tests to validate pipeline/API behaviour.

## GitHub Actions Workflows
The repository includes GitHub Actions workflows for:

- CI: testing, training, evaluation, and Docker build validation
- CD: deployment readiness/container validation
- CT: scheduled or manual retraining

The successful green workflow run provides evidence that the pipeline is reproducible in a clean GitHub Actions runner.

## Future Improvements
- Add MLflow model registry and experiment tracking.
- Push Docker image to Docker Hub or GitHub Container Registry.
- Add automated Kubernetes deployment through a self-hosted runner.
- Add monitoring for prediction drift and data drift.
- Use real customer churn data instead of synthetic data.
