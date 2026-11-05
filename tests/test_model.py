"""Unit tests for model training, evaluation, and inference."""

import pytest
from pathlib import Path
from src.main.data_loader import load_data
from src.main.preprocessing import split_data
from src.main.train import train_model, evaluate_model, save_model
from src.main.predict import predict, load_trained_model


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory):
    """Fixture that trains a model and saves it to a temp directory."""
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, num_folds=3, scoring="accuracy")
    
    temp_dir = tmp_path_factory.mktemp("models")
    temp_model_path = temp_dir / "model.joblib"
    save_model(model, output_path=str(temp_model_path))

    return {
        "model": model,
        "model_path": str(temp_model_path),
        "X_test": X_test,
        "y_test": y_test
    }


def test_model_training_and_evaluation(trained_artifacts):
    """Test that model trains and achieves acceptable accuracy."""
    model = trained_artifacts["model"]
    X_test = trained_artifacts["X_test"]
    y_test = trained_artifacts["y_test"]

    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["accuracy"] >= 0.85, "Accuracy should be at least 85%"


def test_model_persistence_and_prediction(trained_artifacts):
    """Test serialization and deserialization with predictions."""
    loaded = load_trained_model(trained_artifacts["model_path"])
    sample = [[5.1, 3.5, 1.4, 0.2]]  # Setosa typical values

    preds = predict(sample, model=loaded)
    assert len(preds) == 1
    assert preds[0]["prediction_name"] == "setosa"
    assert preds[0]["confidence"] > 0.80
