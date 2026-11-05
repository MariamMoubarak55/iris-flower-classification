"""Model training and evaluation module.

Implements Pipeline, GridSearchCV with config-driven folds and scoring, and model serialization
using joblib (Slide 4, 8, & 19).
"""

from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from src.utils.logger import get_logger
from src.conf.config import get_default, NUM_FOLDS, SCORING
from src.main.data_loader import load_data
from src.main.preprocessing import split_data

logger = get_logger(__name__)


def build_pipeline() -> Pipeline:
    """Build a scikit-learn Pipeline with scaler and classifier.

    Returns:
        Pipeline: Sklearn Pipeline.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=200, random_state=42))
    ])


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    num_folds: Optional[int] = None,
    scoring: Optional[str] = None
) -> Pipeline:
    """Train classifier using GridSearchCV based on design.json parameters (Slide 19).

    Args:
        X_train: Training feature set.
        y_train: Training labels.
        num_folds: Number of cross validation folds (defaults to NUM_FOLDS from config).
        scoring: Scoring metric (defaults to SCORING from config).

    Returns:
        Pipeline: Best fitted pipeline estimator.
    """
    if num_folds is None:
        num_folds = NUM_FOLDS
    if scoring is None:
        scoring = SCORING

    pipeline = build_pipeline()

    param_grid: Dict[str, Any] = {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs", "saga"]
    }

    logger.info(
        f"Starting GridSearchCV with {num_folds} folds and scoring='{scoring}'"
    )
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=num_folds,
        scoring=scoring,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best cross-validation {scoring}: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluate trained model on unseen test dataset (Slide 8).

    Args:
        model: Trained pipeline model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict[str, Any]: Evaluation metrics dictionary.
    """
    logger.info("Evaluating model on test dataset...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(f"Test Accuracy: {acc * 100:.2f}%")
    return {
        "accuracy": acc,
        "classification_report": report
    }


def save_model(model: Pipeline, output_path: Optional[str] = None) -> Path:
    """Serialize and save trained model using joblib (Slide 4).

    Args:
        model: Trained pipeline.
        output_path: Target file path.

    Returns:
        Path: Path to saved model file.
    """
    config = get_default()
    if output_path is None:
        output_path = config.get("paths", {}).get("model_path", "src/resources/model.joblib")

    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, file_path)
    logger.info(f"Model successfully serialized and saved to: {file_path.resolve()}")
    return file_path


def run_training_pipeline() -> Dict[str, Any]:
    """Execute complete end-to-end training pipeline.

    Returns:
        Dict[str, Any]: Results containing trained model path and evaluation metrics.
    """
    logger.info("===== Starting ML Training Pipeline =====")
    
    # 1. Load data
    df = load_data()

    # 2. Preprocess & split
    X_train, X_test, y_train, y_test = split_data(df)

    # 3. Train with Cross-Validation
    model = train_model(X_train, y_train)

    # 4. Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # 5. Serialize
    saved_path = save_model(model)

    logger.info("===== Training Pipeline Completed Successfully =====")
    return {
        "model_path": str(saved_path),
        "metrics": metrics
    }


if __name__ == "__main__":
    run_training_pipeline()
