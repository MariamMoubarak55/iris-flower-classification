"""Model inference and prediction module.

Loads serialized model and serves predictions on new feature samples (Slide 3 & 4).
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger
from src.conf.config import get_default
from src.main.preprocessing import FEATURE_COLUMNS

logger = get_logger(__name__)

CLASS_MAPPING = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}


def load_trained_model(model_path: Optional[str] = None) -> Pipeline:
    """Load serialized model object from disk (Slide 4).

    Args:
        model_path: Path to serialized model file.

    Returns:
        Pipeline: Loaded scikit-learn pipeline.
    """
    config = get_default()
    if model_path is None:
        model_path = config.get("paths", {}).get("model_path", "src/resources/model.joblib")

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path.resolve()}. Please train the model first by running train.py."
        )

    logger.info(f"Loading trained model from {path.resolve()}")
    model: Pipeline = joblib.load(path)
    return model


def predict(
    features: Union[List[List[float]], pd.DataFrame, np.ndarray],
    model: Optional[Pipeline] = None,
    model_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Predict flower species for given feature inputs.

    Args:
        features: Input feature matrix (each row: [sepal_length, sepal_width, petal_length, petal_width]).
        model: Pre-loaded model pipeline (optional).
        model_path: Path to model file if model is not pre-loaded.

    Returns:
        List[Dict[str, Any]]: List of predictions with class ID, name, and probabilities.
    """
    if model is None:
        model = load_trained_model(model_path)

    if not isinstance(features, pd.DataFrame):
        df_features = pd.DataFrame(features, columns=FEATURE_COLUMNS)
    else:
        df_features = features[FEATURE_COLUMNS]

    logger.info(f"Generating predictions for {len(df_features)} sample(s)")
    raw_preds = model.predict(df_features)
    probabilities = model.predict_proba(df_features)

    results: List[Dict[str, Any]] = []
    for idx, (pred, probs) in enumerate(zip(raw_preds, probabilities)):
        pred_int = int(pred)
        results.append({
            "sample_index": idx,
            "prediction_id": pred_int,
            "prediction_name": CLASS_MAPPING.get(pred_int, f"Class_{pred_int}"),
            "confidence": float(np.max(probs)),
            "probabilities": {
                CLASS_MAPPING.get(i, f"Class_{i}"): float(prob)
                for i, prob in enumerate(probs)
            }
        })

    return results
