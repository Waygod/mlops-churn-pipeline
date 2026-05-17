# Automated Customer Churn Prediction MLOps Pipeline Using GitHub Actions and Docker

## 1. Introduction

Machine learning systems require more than model development. In production-oriented environments, models must be trained, tested, versioned, deployed, and updated in a controlled and reproducible manner. This project designs and implements an MLOps pipeline for a customer churn prediction use case. The system uses GitHub Actions to automate continuous integration, continuous delivery, and continuous training. The trained model is exposed through a Flask API and deployed as a Docker container on a Google Compute Engine virtual machine.

The motivation for this approach is that machine learning systems can accumulate technical debt when data, code, model artefacts, and deployment processes are managed manually. Sculley et al. (2015) argue that ML systems create hidden technical debt due to dependency entanglement, unstable data dependencies, and feedback loops. Therefore, this project focuses on reproducibility and automation rather than only model accuracy.

## 2. Use Case and Approach Taken

The selected use case is telecom customer churn prediction. The model predicts whether a customer is likely to leave the service based on features such as tenure, contract type, monthly charges, payment method, internet service type, and billing preferences. This use case was selected because churn prediction is a practical binary classification problem with clear business value. It also supports a compact but complete MLOps pipeline because the data can be preprocessed, the model can be trained quickly, and predictions can be served through a simple API.

A Random Forest classifier was selected because it performs well on structured/tabular data, is robust to non-linear relationships, and does not require GPU infrastructure. This is appropriate because the assessment focuses on MLOps lifecycle design rather than deep learning complexity. Flask was used to expose the model through a `/predict` endpoint. Docker was used to package the application so that the same runtime environment can be used locally, in CI, and on the deployment server.

The project follows the MLOps principles described by Kreuzberger, Kühl, and Hirschl (2023), who define MLOps as the combination of machine learning, DevOps, and data engineering practices for automating and operationalising ML systems. Google Cloud's MLOps architecture guidance also identifies CI, CD, and CT as core automation layers for predictive ML systems.

## 3. Branching Strategy

A simple GitFlow-style branching strategy is used:

| Branch | Purpose |
|---|---|
| `main` | Production-ready code. Deployment workflow runs from this branch. |
| `develop` | Integration branch for tested features before release. |
| `feature/*` | Used for new features such as preprocessing changes, API updates, or workflow improvements. |
| `hotfix/*` | Used for urgent production fixes. |

The development flow is:

```text
feature/* → develop → main
```

New code is developed in feature branches and merged into `develop` after testing. Once the pipeline is stable, `develop` is merged into `main`. Only commits to `main` trigger the automatic deployment workflow. This reduces the risk of deploying unstable code.

## 4. MLOps Pipeline Stages

### 4.1 Data Acquisition and Preprocessing

The data stage generates or acquires a Telco-style customer churn dataset and stores it under `data/raw/`. The preprocessing script validates required columns, removes invalid records, handles missing values, and writes a clean dataset to `data/processed/`.

### 4.2 Model Training and Testing

The training stage uses a scikit-learn pipeline containing preprocessing transformations and a Random Forest classifier. Categorical features are encoded with one-hot encoding, and numeric features are scaled. The trained pipeline is saved as `app/model.pkl`, allowing the Flask application to use the same preprocessing logic during inference.

Model evaluation records accuracy, F1-score, and ROC-AUC in `app/metrics.json`. The evaluation script can fail the pipeline if model quality falls below a defined threshold. This prevents weak models from being accepted into deployment.

### 4.3 Continuous Integration

The CI workflow runs on pushes and pull requests. It installs dependencies, generates data, preprocesses data, trains the model, evaluates it, runs unit tests, and builds the Docker image. This ensures that code changes do not break the model pipeline or API before they are merged.

### 4.4 Continuous Delivery

The CD workflow runs when code is pushed to `main`. It builds the Docker image, pushes it to DockerHub, connects to the Google VM using SSH, pulls the latest image, stops the old container, and starts a new container. This provides automated deployment of the Flask API.

### 4.5 Continuous Training

The CT workflow runs weekly and can also be triggered manually. It reruns the data generation/acquisition, preprocessing, model training, model evaluation, and testing stages. This demonstrates how the model can be updated over time as new data becomes available.

## 5. Deployment Architecture

The deployment architecture uses a Dockerized Flask API hosted on a Google Compute Engine VM.

```text
GitHub Repository
    ↓
GitHub Actions
    ↓
Docker Image Build
    ↓
DockerHub Registry
    ↓
Google Compute Engine VM
    ↓
Running Flask API Container
```

This architecture was selected because it is simpler and more reliable for a small assessment project than Kubernetes, while still satisfying the requirement for automatic deployment through a Google VM. The container exposes the API on port 5000 internally and maps it to port 80 on the VM.

## 6. Testing

The pipeline includes automated tests using Pytest. The first test validates the preprocessing stage by checking that missing values are removed and that the target variable remains valid. The second test validates the Flask API by sending a sample prediction request and checking that the API returns a valid prediction and churn probability.

## 7. Conclusion

This project implements a complete MLOps pipeline for customer churn prediction. It includes data preprocessing, model training, automated testing, Docker containerization, CI/CD workflows, automatic deployment to a Google VM, and scheduled continuous training. The approach prioritises reproducibility, automation, and deployment reliability, which are central concerns in practical ML systems.

Future improvements could include MLflow experiment tracking, model registry integration, drift monitoring, Kubernetes deployment, and automated rollback if a newly trained model performs worse than the production model.

## References

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J. and Dennison, D. (2015). Hidden Technical Debt in Machine Learning Systems. Advances in Neural Information Processing Systems.

Kreuzberger, D., Kühl, N. and Hirschl, S. (2023). Machine Learning Operations (MLOps): Overview, Definition, and Architecture. IEEE Access, 11, pp.31866–31879.

Google Cloud. (2024). MLOps: Continuous delivery and automation pipelines in machine learning. Google Cloud Architecture Center.
