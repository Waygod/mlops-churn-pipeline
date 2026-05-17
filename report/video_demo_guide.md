# 10–15 Minute Video Demonstration Guide

Do not read this word-for-word. Use it as a speaking structure.

## 1. Introduction — 1 minute

Explain:
- This is a customer churn prediction MLOps pipeline.
- The goal is automated ML lifecycle management.
- The pipeline uses GitHub Actions, Flask, Docker, Google VM deployment, and continuous training.

## 2. Repository Structure — 2 minutes

Show:
- `src/` for data generation, preprocessing, training, evaluation, retraining.
- `app/` for Flask API.
- `tests/` for Pytest tests.
- `.github/workflows/` for CI, CD, and CT workflows.
- `Dockerfile` for containerization.

## 3. Model Use Case — 1 minute

Explain:
- Use case: telecom churn prediction.
- Input: customer attributes such as contract, tenure, monthly charges, and payment method.
- Output: churn/no churn and churn probability.
- Model: Random Forest classifier.

## 4. Flask API — 2 minutes

Show:
- `app/app.py`
- `/health`
- `/predict`

Run:

```bash
python app/app.py
```

Then test:

```bash
bash scripts/test_api_curl.sh
```

## 5. Docker Deployment — 2 minutes

Show:

```bash
docker build -t churn-api .
docker run -p 5000:5000 churn-api
```

Explain that Docker packages the model, API, dependencies, and runtime environment into one deployable container.

## 6. GitHub Actions CI — 2 minutes

Open GitHub Actions tab.

Explain CI workflow:
- install dependencies
- generate data
- preprocess
- train
- evaluate
- run tests
- build Docker image

## 7. Continuous Deployment — 2 minutes

Show `deploy.yml`.

Explain:
- triggered on push to main
- builds Docker image
- pushes image to DockerHub
- SSHs into Google VM
- pulls latest image
- restarts container

## 8. Continuous Training — 1–2 minutes

Show `retrain.yml`.

Explain:
- scheduled weekly
- can be triggered manually
- retrains model
- evaluates metrics
- uploads model artefacts

## 9. Branching Strategy — 1 minute

Explain:
- `feature/*` for development
- `develop` for integration
- `main` for production
- only `main` deploys

## 10. Closing — 30 seconds

Say:
The final system demonstrates a full MLOps lifecycle: preprocessing, training, testing, Dockerization, CI, CD, and CT.
