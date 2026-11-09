# Sentiment Analysis MLOps Pipeline

An end-to-end MLOps pipeline for sentiment classification using fine-tuned **DistilBERT**, **FastAPI**, **MLflow** (experiment tracking & model registry with champion alias), **Docker**, and **GitHub Actions** CI/CD with automated image publishing to **GitHub Container Registry (GHCR)**.

---

## 📦 Container Registry (GHCR)

The production Docker image is automatically built, tested, and published to GitHub Container Registry:

```bash
docker pull ghcr.io/vedansht2211/sentiment-analysis:latest
```

*Specific tagged version:*
```bash
docker pull ghcr.io/vedansht2211/sentiment-analysis:sha-96fee25
```

---

## 📊 Dataset & Training Results

### Dataset Source
The training and test sets are sourced from Kaggle:
- **Kaggle Dataset:** [Sentiment Analysis Dataset (by abhi8923shriv)](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset)
- **Task:** 3-class tweet sentiment classification (`negative`, `neutral`, `positive`).

### Benchmark & Validation Results

| Metric / Parameter | Value |
| :--- | :--- |
| **Dataset** | [Kaggle Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset) (`train.csv` / `test.csv`) |
| **Model** | `distilbert-base-uncased` |
| **Epochs** | 2 epochs (688 steps) |
| **Validation Accuracy** | **80.02%** |
| **Validation F1 Score (Weighted)** | **0.8006** |
| **Test Accuracy** | **79.48%** |
| **Test F1 Score** | **0.7953** |

> **Quality Gate:** `train.py` automatically validates the model against a minimum accuracy threshold (`MIN_TEST_ACCURACY = 0.70`). Only models meeting this gate are saved and promoted to the MLflow Model Registry.

---

## 🏗️ Repository Structure

```text
├── .github/workflows/
│   └── ci.yml               # Automated Pytest suite + Docker build and GHCR push on main
├── models/sentiment_bert/   # Saved DistilBERT model weights and tokenizer files
├── tests/
│   └── test_api.py          # Fast API unit tests (uses mock model, runs without MLflow)
├── Dockerfile               # Container definition for the FastAPI service
├── .dockerignore            # Excludes datasets, local DBs, and virtualenvs from image
├── main.py                  # Production FastAPI service with MLflow champion resolver
├── train.py                 # Primary: Fine-tunes DistilBERT, validates, and auto-registers model to MLflow
├── register.py              # Optional/Colab: Registers external/Colab weights into MLflow under @champion
├── requirements.txt         # Core production dependencies
└── mlflow.db                # SQLite backend for MLflow tracking and registry
```

---

## ⚡ Step-by-Step Setup & Execution

Follow these steps in the exact order below.

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torchvision
```

> **Note:** `torchvision` is required by `mlflow.transformers` during model logging and registration.

---

### 2. Start the MLflow Tracking Server

Run MLflow from the project root with the explicit SQLite backend store and host permissions:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 --allowed-hosts "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,host.docker.internal,host.docker.internal:5000"
```

- Accessible locally at: `http://localhost:5000`
- `--allowed-hosts` ensures Docker containers can query MLflow via `host.docker.internal:5000`.

---

### 3. Register the Model (Assign `@champion`)

The API requires a registered model aliased as `@champion` in MLflow to start up. Choose the workflow that matches your environment:

#### Primary Workflow: Local End-to-End Training (Auto-registers `@champion`)
```bash
python train.py
```
- Trains DistilBERT locally.
- Verifies the accuracy quality gate (`>= 70%`).
- Saves weights to `models/sentiment_bert/`.
- Logs run parameters, metrics, and artifacts to MLflow.
- **Automatically registers** the model in MLflow and assigns the `@champion` alias.
- **Note:** In this flow, `register.py` is **NOT necessary**.

#### Alternative Workflow: Remote / Colab GPU Training (Manual Registration)
Use this if you train on Google Colab or an external GPU without access to your local MLflow server:
1. Run your training script on Google Colab (`report_to="none"`, no MLflow dependency).
2. Download the generated model files (`model.safetensors`, `config.json`, `tokenizer.json`, etc.).
3. Replace the contents of your local `models/sentiment_bert/` folder with these new files.
4. Run `register.py` locally:
   ```bash
   python register.py
   ```
   This registers the imported weights to your local MLflow server and assigns the `@champion` alias.

---

### 4. Run Automated Tests

The unit tests mock the pipeline, requiring neither a running MLflow server nor GPU/model weights:

```bash
python -m pytest -v
```

All 4 test cases validate the `/health` endpoint, prediction contracts, input sanitization, and Pydantic length validations.

---

### 5. Run the Production API Container

Launch the container from GitHub Container Registry (GHCR) pointing to your host MLflow server. The container dynamically pulls the approved model from MLflow via the `@champion` alias upon startup:

```bash
docker run -d --name sentiment-api -p 8000:8000 -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 ghcr.io/vedansht2211/sentiment-analysis:latest
```

---

### 6. Test the API Endpoints

#### Check Service Health
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_name": "sentiment-classifier",
  "model_version": "1"
}
```

#### Run a Sentiment Prediction
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"text\": \"This product exceeded all my expectations! Absolutely fantastic.\"}"
```
**Response:**
```json
{
  "class_name": "positive",
  "confidence": 0.9854
}
```

