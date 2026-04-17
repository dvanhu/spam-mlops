# spam-mlops

![CI/CD](https://github.com/dvanhu/spam-mlops/actions/workflows/ci.yml/badge.svg)
![Docker Hub](https://img.shields.io/docker/pulls/dvanhu/spam-mlops?logo=docker&label=Docker%20Pulls)
![Docker Image Size](https://img.shields.io/docker/image-size/dvanhu/spam-mlops/latest?logo=docker)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-tracked-945DD6?logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracked-0194E2?logo=mlflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade machine learning pipeline for email spam classification. This project demonstrates an end-to-end MLOps workflow, covering data versioning, experiment tracking, model training, API serving, containerization, and automated CI/CD — built with industry-standard tooling and designed for reproducibility.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Running the API Locally](#running-the-api-locally)
- [Running with Docker](#running-with-docker)
- [CI/CD Pipeline](#cicd-pipeline)
- [Future Improvements](#future-improvements)

---

## Project Overview

`spam-mlops` is a spam classification system built around a Naive Bayes NLP pipeline (CountVectorizer + MultinomialNB). The project goes beyond model training to address the operational concerns that matter in production: reproducible pipelines, versioned data, tracked experiments, containerized serving, and automated deployment.

The classifier is exposed as a REST API via FastAPI and packaged as a Docker image. A GitHub Actions workflow automatically builds and pushes updated images to Docker Hub on every push to `main`.

---

## Architecture

The system follows a linear MLOps flow from raw data to deployed API:

```
Raw Data — spam.csv (DVC-tracked)
        │
        ▼
  DVC Pipeline (dvc.yaml)
  ┌─────────────────────┐
  │  Training Stage     │  ← CountVectorizer + MultinomialNB fit and serialized
  └─────────────────────┘
        │
        ▼
  Model Artifacts
  ├── models/model.pkl
  └── models/vectorizer.pkl
        │
        ▼
  FastAPI Inference API (api/)
  ├── app.py        ← Route definitions and model loading
  ├── schema.py     ← Pydantic request/response models
  └── utils.py      ← Preprocessing and prediction helpers
        │
        ▼
  Docker Image
        │
        ▼
  GitHub Actions CI/CD → Docker Hub
```

**Data layer:** The raw dataset (`spam.csv`) is tracked by DVC and excluded from Git. Only the `.dvc` pointer file and `dvc.lock` snapshot are committed, ensuring any collaborator can reproduce the exact dataset by running `dvc pull`.

**Training layer:** The DVC pipeline (`dvc.yaml`) defines the training stage with explicit input/output dependencies. Running `dvc repro` executes only the stages whose dependencies have changed, making the pipeline incremental and auditable.

**Artifact layer:** Training produces two serialized artifacts — `model.pkl` (the fitted classifier) and `vectorizer.pkl` (the fitted CountVectorizer). Separating these allows the vectorizer to be reused independently during inference without re-fitting.

**Serving layer:** The FastAPI application loads both artifacts at startup. Inference logic is isolated in `utils.py`, keeping route definitions in `app.py` clean. Input/output contracts are enforced via Pydantic schemas in `schema.py`.

**Deployment layer:** A `Dockerfile` packages the API and all runtime dependencies into a portable image. GitHub Actions builds and pushes this image to Docker Hub on every push to `main`.

---

## Tech Stack

| Tool | Role | Why |
|---|---|---|
| **Python 3.12** | Runtime | Latest stable release; improved performance and typing support |
| **Scikit-learn** | ML model | Battle-tested NLP primitives; CountVectorizer + MultinomialNB is a strong, interpretable baseline for text classification |
| **FastAPI** | API serving | Async-native, low-overhead, and automatically generates OpenAPI documentation |
| **DVC** | Data & pipeline versioning | Separates large binary artifacts from Git history; enables reproducible pipeline execution with stage-level caching |
| **MLflow** | Experiment tracking | Provides a structured store for run parameters, metrics, and artifacts across training iterations |
| **Docker** | Containerization | Eliminates environment drift between development and production; produces a self-contained, portable runtime |
| **GitHub Actions** | CI/CD | Native to GitHub; no external CI server required; triggers directly on repository events |
| **Docker Hub** | Image registry | Widely supported pull target; accessible to downstream deployment environments without additional infrastructure |

---

## Project Structure

```
spam-mlops/
├── .dvc/
│   ├── .gitignore
│   └── config                   # DVC remote configuration
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI/CD pipeline
├── api/
│   ├── app.py                   # FastAPI application, route definitions, model loading
│   ├── schema.py                # Pydantic request/response models
│   └── utils.py                 # Text preprocessing and prediction logic
├── data/
│   ├── .gitignore               # Excludes raw CSV from Git
│   └── spam.csv.dvc             # DVC pointer to the tracked dataset
├── models/
│   ├── .gitignore               # Excludes serialized artifacts from Git
│   ├── model.pkl                # Trained MultinomialNB classifier
│   └── vectorizer.pkl           # Fitted CountVectorizer
├── src/
│   └── train.py                 # Training script: fits pipeline, serializes artifacts, logs to MLflow
├── .dockerignore
├── .dvcignore
├── .gitignore
├── Dockerfile                   # Container image definition for the FastAPI application
├── dvc.lock                     # Cryptographic snapshot of pipeline state (committed to Git)
├── dvc.yaml                     # Pipeline stage definitions and dependency graph
├── requirements.txt
└── trigger.txt                  # Used to force CI pipeline re-runs without code changes
```

> `data/` and `models/` are excluded from Git via `.gitignore`. Their contents are managed entirely by DVC. The `models/` directory contains artifacts that are either pulled via `dvc pull` or generated by running `dvc repro`.

---

## Setup Instructions

### Prerequisites

- Python 3.12
- Docker (for containerized runs)
- DVC (`pip install dvc`)

### 1. Clone the repository

```bash
git clone https://github.com/dvanhu/spam-mlops.git
cd spam-mlops
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull data and reproduce the pipeline

```bash
dvc pull        # Fetch tracked data and cached artifacts from the configured DVC remote
dvc repro       # Re-run any pipeline stages whose inputs have changed
```

After `dvc repro` completes, `models/model.pkl` and `models/vectorizer.pkl` will be present and ready for serving.

---

## Running the API Locally

With the model artifacts in place, start the development server:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Classify input text as spam or ham |
| `GET` | `/docs` | Auto-generated OpenAPI documentation (Swagger UI) |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Congratulations! You have won a free prize. Click here to claim."}'
```

### Example response

```json
{
  "label": "spam",
  "confidence": 0.9741
}
```

---

## Running with Docker

### Build the image locally

```bash
docker build -t spam-mlops:latest .
```

### Run the container

```bash
docker run -p 8000:8000 spam-mlops:latest
```

The API will be accessible at `http://localhost:8000`.

### Pull the pre-built image from Docker Hub

```bash
docker pull dvanhu/spam-mlops:latest
docker run -p 8000:8000 dvanhu/spam-mlops:latest
```

> The Docker Hub image is updated automatically by the CI/CD pipeline on every push to `main`.

---

## CI/CD Pipeline

The pipeline is defined in `.github/workflows/ci.yml` and triggers on every push to the `main` branch.

### Pipeline stages

```
Push to main
     │
     ▼
Checkout repository
     │
     ▼
Set up Python 3.12 + install dependencies
     │
     ▼
Docker build
     │
     ▼
Authenticate to Docker Hub
     │
     ▼
Push image: dvanhu/spam-mlops:latest
            dvanhu/spam-mlops:<git-sha>
```

Both `latest` and commit-SHA tags are pushed. This allows deployment environments to either track the most recent image or pin to a specific, immutable build.

### Required repository secrets

Configure these under **Settings → Secrets and variables → Actions**:

| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub account username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (not the account password) |

> `trigger.txt` provides a lightweight mechanism to force a CI run — edit and push the file when a rebuild is needed without any code changes (e.g., to pick up a base image update).

---

## Future Improvements

The current implementation represents a functional, deployable baseline. The following enhancements would move it closer to a fully production-ready system:

**DVC remote storage**
Migrate the DVC remote to a cloud backend (S3, GCS, or Azure Blob) so that datasets and model artifacts can be shared across environments and pulled directly by CI runners without manual transfer.

**MLflow Model Registry**
Integrate the MLflow Model Registry to formalize the model promotion workflow from `Staging` to `Production`. This enables gated deployments where only validated model versions are served, with a full audit trail of who promoted what and when.

**Automated evaluation gate in CI**
Extend the GitHub Actions pipeline to run `dvc repro` and evaluate the model against a held-out test set as part of the build. Fail the pipeline if accuracy or F1 drops below a defined threshold, preventing regressions from reaching Docker Hub.

**Kubernetes deployment**
Replace direct `docker run` usage with a Kubernetes deployment manifest. This enables horizontal scaling, rolling updates, and liveness/readiness probes for the inference API.

**Input drift monitoring**
Integrate a tool such as Evidently or WhyLogs to monitor incoming prediction requests for distribution shift relative to the training corpus. Surface alerts or trigger automated retraining when drift exceeds a configured threshold.

**API authentication**
Add API key or OAuth2 authentication to the FastAPI application to restrict access in non-development deployments.
