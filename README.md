```markdown
# Iris Flower Classification - Production ML Project

A complete end-to-end Machine Learning project for classifying Iris flowers, built following engineering best practices to take models from notebook experiments to production deployment.

---

## Project Architecture

```text
iris_classification_project/
├── .gitignore               # Ignore virtual environment, cache, and log files
├── README.md                # Project documentation and usage guide
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker image definition for the application
├── docker-compose.yml       # Easy container orchestration
├── run.py                   # Local training and experimentation entry point
├── run_app.py               # Flask server entry point
├── logs.log                 # Application event logs
│
├── data/                    # Dataset storage (iris.csv)
│
├── src/
│   ├── conf/                # Project configuration (JSON-driven)
│   │   ├── design.json      # Hyperparameters, CV folds, paths
│   │   └── config.py        # get_default() for dynamic config loading
│   │
│   ├── utils/               # Helper utilities
│   │   └── logger.py        # Professional logging system
│   │
│   ├── main/                # Core Machine Learning logic
│   │   ├── data_loader.py   # Data loading and preparation
│   │   ├── preprocessing.py # Stratified split and feature processing
│   │   ├── train.py         # GridSearchCV hyperparameter tuning, training, and model persistence
│   │   └── predict.py       # Model loading, predictions, and probability extraction
│   │
│   ├── resources/           # Trained model storage (model.joblib)
│   │
│   └── template_app/        # Flask-based REST API
│       └── app.py           # Endpoints for receiving data and returning predictions
│
└── tests/                   # Unit tests
    ├── test_data.py         # Data quality and split tests
    ├── test_model.py        # Model accuracy and prediction integrity tests
    └── test_app.py          # REST API endpoint tests
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate the environment
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train and Evaluate the Model Locally

```bash
python run.py
```

This command will:

1. Load the Iris dataset
2. Perform a stratified train/test split
3. Train the model using `GridSearchCV` with cross-validation
4. Compute evaluation metrics (Accuracy & Classification Report)
5. Save the trained model to `src/resources/model.joblib`
6. Run a quick sample prediction test

### 3. Run the Test Suite

```bash
pytest
```

### 4. Start the Flask API Server

```bash
python run_app.py
```

The server will be available at: `http://localhost:5000`

---

## API Endpoints

### 1. Health Check

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Prediction Endpoint

```http
POST /predict
Content-Type: application/json
```

**Input format (list of feature vectors):**

```json
{
  "features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.0, 5.2, 2.3]
  ]
}
```

**Input format (single sample as dictionary):**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Example Response:**

```json
{
  "status": "success",
  "predictions": [
    {
      "sample_index": 0,
      "prediction_id": 0,
      "prediction_name": "setosa",
      "confidence": 0.985,
      "probabilities": {
        "setosa": 0.985,
        "versicolor": 0.014,
        "virginica": 0.001
      }
    }
  ]
}
```

---

## Docker Deployment

```bash
# Build the image and start the container
docker compose up --build
```

The API will be accessible at `http://localhost:5000` once the container is running.

---

## Features

- **JSON-driven configuration** for hyperparameters and paths
- **Stratified train/test splitting** to preserve class distribution
- **GridSearchCV** with cross-validation for robust hyperparameter tuning
- **Professional logging** system
- **Model persistence** with joblib
- **Flask REST API** with health and prediction endpoints
- **Unit tests** covering data, model, and API layers
- **Docker support** for reproducible deployment

---

## License

This project is intended for educational and demonstration purposes.
```
