# spam-mlops

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
Raw Data (DVC-tracked)
        │
        ▼
  DVC Pipeline (dvc.yaml)
  ┌─────────────────────┐
  │  1. Preprocessing   │  ← CountVectorizer fit on training set
  │  2. Training        │  ← MultinomialNB trained and serialized
  │  3. Evaluation      │  ← Metrics logged to MLflow
  └─────────────────────┘
        │
        ▼
  Model Artifact (models/)
        │
        ▼
  FastAPI Inference API (api/)
        │
        ▼
  Docker Image
        │
        ▼
  GitHub Actions CI/CD → Docker Hub
```

**Data layer:** Raw datasets are tracked by DVC and excluded from Git. Only the `dvc.lock` file (a cryptographic snapshot of pipeline state) is committed, ensuring any collaborator can reproduce the exact dataset and model by running `dvc pull`.

**Training layer:** The DVC pipeline (`dvc.yaml`) defines the stages: preprocessing, training, and evaluation. Running `dvc repro` executes only the stages whose dependencies have changed, making the pipeline incremental and efficient.

**Experiment tracking:** MLflow records parameters, metrics, and artifact paths for every training run. This provides a persistent audit trail and enables comparison across experiments without manually managing log files.

**Serving layer:** The trained model artifact is loaded at startup by the FastAPI application, which exposes a `/predict` endpoint accepting text input and returning a spam/ham classification with confidence.

**Deployment layer:** A `Dockerfile` packages the API and all runtime dependencies into a portable image. GitHub Actions builds this image on every push to `main` and pushes it to Docker Hub, keeping the registry image current without manual intervention.

---

## Tech Stack

| Tool | Role | Why |
|---|---|---|
| **Python 3.12** | Runtime | Latest stable release; improved performance and typing support |
| **Scikit-learn** | ML model | Battle-tested NLP primitives; CountVectorizer + MultinomialNB is a strong baseline for text classification |
| **FastAPI** | API serving | Async-native, automatically generates OpenAPI docs, and has low latency overhead compared to Flask |
| **DVC** | Data & pipeline versioning | Separates large binary artifacts from Git history; enables reproducible pipeline execution with stage caching |
| **MLflow** | Experiment tracking | Provides a structured store for run parameters, metrics, and artifacts; integrates with the model registry for promotion workflows |
| **Docker** | Containerization | Eliminates environment drift between development and production; produces a self-contained, portable runtime |
| **GitHub Actions** | CI/CD | Native to GitHub; no external CI server required; tightly integrated with the repository event model |
| **Docker Hub** | Image registry | Widely supported pull target for deployment environments; free tier sufficient for open-source projects |

---

## Project Structure

```
spam-mlops/
├── api/
│   ├── main.py              # FastAPI application and route definitions
│   └── schemas.py           # Pydantic request/response models
├── src/
│   ├── preprocess.py        # Text cleaning and feature extraction
│   ├── train.py             # Model training and artifact serialization
│   └── evaluate.py          # Metric computation and MLflow logging
├── data/
│   └── .gitignore           # Raw data excluded from Git; managed by DVC
├── models/
│   └── .gitignore           # Serialized model artifacts; populated by DVC or training run
├── .github/
│   └── workflows/
│       └── ci.yml           # GitHub Actions CI/CD pipeline definition
├── dvc.yaml                 # Pipeline stage definitions and dependency graph
├── dvc.lock                 # Cryptographic snapshot of pipeline state (committed to Git)
├── Dockerfile               # Multi-stage image build for the FastAPI application
├── requirements.txt         # Python dependencies
└── README.md
```

> `data/` and `models/` directories contain `.gitignore` files that exclude binary content from version control. Their contents are managed entirely by DVC.

---

## Setup Instructions

### Prerequisites

- Python 3.12
- Docker (for containerized runs)
- DVC (`pip install dvc`)
- A configured DVC remote (local or cloud) if pulling data from scratch

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
dvc pull          # Fetch tracked data and cached artifacts from the DVC remote
dvc repro         # Re-run any pipeline stages whose inputs have changed
```

If no remote is configured, training data must be placed manually in `data/` and the pipeline run with:

```bash
dvc repro --no-cache
```

### 5. Start the MLflow tracking server (optional)

```bash
mlflow ui
```

Navigate to `http://localhost:5000` to view logged experiments and metrics.

---

## Running the API Locally

After completing setup and ensuring the model artifact exists in `models/`:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
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

> The Docker Hub image is updated automatically by the CI/CD pipeline on every merge to `main`.

---

## CI/CD Pipeline

The pipeline is defined in `.github/workflows/ci.yml` and triggers on every push to the `main` branch.

### Pipeline stages

```
Push to main
     │
     ▼
Checkout code
     │
     ▼
Set up Python 3.12
     │
     ▼
Install dependencies
     │
     ▼
Run tests (if present)
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

### Secrets required

The following secrets must be set in the repository's **Settings → Secrets and variables → Actions**:

| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub account username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (not account password) |

Both `latest` and commit-SHA tags are pushed, allowing deployment environments to pin to a specific build or always track the most recent image.

---

## Future Improvements

The current implementation represents a functional baseline. The following enhancements would move it closer to a fully production-ready system:

**DVC remote storage**
Configure a cloud-backed DVC remote (S3, GCS, or Azure Blob) so that datasets and model artifacts can be shared across environments and CI runners without manual file transfers.

**MLflow Model Registry**
Integrate the MLflow Model Registry to formalize the promotion workflow from `Staging` to `Production`. This enables gated deployments where only validated model versions are served.

**Automated model evaluation gate in CI**
Extend the GitHub Actions pipeline to run `dvc repro` and evaluate the model against a held-out test set. Fail the build if accuracy or F1 falls below a defined threshold, preventing regressions from being deployed.

**Kubernetes deployment**
Replace direct `docker run` usage with a Kubernetes deployment manifest. This enables horizontal scaling, rolling updates, and liveness/readiness probes for the inference API.

**Input drift monitoring**
Integrate a tool such as Evidently or WhyLogs to monitor incoming prediction requests for distribution shift relative to the training data. Trigger retraining pipelines automatically when drift exceeds a configured threshold.

**API authentication**
Add API key or OAuth2 authentication to the FastAPI application to restrict access in non-development environments.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
