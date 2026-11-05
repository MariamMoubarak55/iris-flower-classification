```markdown
# Iris Flower Classification

**Production-Ready Machine Learning Project**

A complete end-to-end ML pipeline for classifying Iris flowers.  
Built with engineering best practices to move models from experimentation to production deployment.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Train the Model](#1-train-the-model)
  - [Run Tests](#2-run-tests)
  - [Start the API Server](#3-start-the-api-server)
- [API Reference](#api-reference)
  - [Health Check](#health-check)
  - [Predict](#predict)
- [Docker Deployment](#docker-deployment)
- [License](#license)

---

## Overview

This project demonstrates a clean and modular approach to building a production-grade machine learning system. It covers the full lifecycle:

1. Data loading and stratified splitting  
2. Hyperparameter tuning with `GridSearchCV`  
3. Model training and persistence  
4. REST API serving with Flask  
5. Unit testing and Docker packaging  

The Iris dataset is used as a classic multi-class classification problem (Setosa, Versicolor, Virginica).

---

## Features

| Feature                        | Description                                      |
|--------------------------------|--------------------------------------------------|
| JSON-driven configuration      | Hyperparameters and paths managed via `design.json` |
| Stratified train/test split    | Preserves class distribution                     |
| GridSearchCV + Cross-Validation| Robust hyperparameter tuning                     |
| Professional logging           | Centralized logging system                       |
| Model persistence              | Saved with `joblib`                              |
| Flask REST API                 | Clean prediction endpoints                       |
| Unit tests                     | Coverage for data, model, and API layers         |
| Docker support                 | Reproducible deployment                          |

---

## Project Structure

```text
iris_classification_project/
│
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.py                      # Training entry point
├── run_app.py                  # Flask server entry point
├── logs.log
│
├── data/                       # Dataset (iris.csv)
│
├── src/
│   ├── conf/
│   │   ├── design.json         # Hyperparameters, CV folds, paths
│   │   └── config.py           # Dynamic configuration loader
│   │
│   ├── utils/
│   │   └── logger.py           # Logging utility
│   │
│   ├── main/
│   │   ├── data_loader.py      # Load and prepare data
│   │   ├── preprocessing.py    # Stratified split & feature handling
│   │   ├── train.py            # Training + GridSearchCV + save model
│   │   └── predict.py          # Load model and generate predictions
│   │
│   ├── resources/
│   │   └── model.joblib        # Trained model
│   │
│   └── template_app/
│       └── app.py              # Flask application
│
└── tests/
    ├── test_data.py
    ├── test_model.py
    └── test_app.py
```

---

## Installation

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

**Windows (PowerShell)**
```bash
.\venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Model

```bash
python run.py
```

**What this command does:**

1. Loads the Iris dataset  
2. Performs stratified train/test split  
3. Runs `GridSearchCV` with cross-validation  
4. Evaluates the model (Accuracy + Classification Report)  
5. Saves the model to `src/resources/model.joblib`  
6. Performs a quick sample prediction  

### 2. Run Tests

```bash
pytest
```

### 3. Start the API Server

```bash
python run_app.py
```

Server will be available at: **http://localhost:5000**

---

## API Reference

### Health Check

```http
GET /health
```

**Response**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### Predict

```http
POST /predict
Content-Type: application/json
```

#### Option 1 — List of feature vectors

```json
{
  "features": [
    [5.1, 3.5, 1.4, 0.2],
    [6.7, 3.0, 5.2, 2.3]
  ]
}
```

#### Option 2 — Single sample (dictionary)

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

#### Example Response

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

Build and run the application using Docker Compose:

```bash
docker compose up --build
```

Once the container is running, the API will be available at:

**http://localhost:5000**

---

## License

This project is intended for educational and demonstration purposes.
```

### التحسينات اللي اتعملت:

- أضفت **Table of Contents** للتنقل السهل
- قسّمت المحتوى لأقسام واضحة ومنظمة
- استخدمت جدول للـ Features
- حسّنت تنسيق الـ Project Structure
- رتّبت الـ API بشكل أوضح
- خليت التنسيق متناسق وسهل القراءة
