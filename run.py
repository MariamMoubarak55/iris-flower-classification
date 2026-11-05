"""Local entry point for running the ML pipeline (Slide 9).

Executes data loading, preprocessing, model training, evaluation, serialization,
and tests a sample prediction.
"""

import sys
from pathlib import Path

# Ensure root project directory is in python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.main.train import run_training_pipeline
from src.main.predict import predict

logger = get_logger("run_local")


def main():
    logger.info("Starting local pipeline execution...")

    # 1. Run training and evaluation
    results = run_training_pipeline()
    logger.info(f"Training finished. Accuracy: {results['metrics']['accuracy'] * 100:.2f}%")

    # 2. Run sample inference
    sample_features = [
        [5.1, 3.5, 1.4, 0.2],  # setosa sample
        [6.0, 2.7, 5.1, 1.6],  # versicolor/virginica sample
    ]
    logger.info("Testing sample local prediction...")
    predictions = predict(sample_features)
    for p in predictions:
        logger.info(f"Sample {p['sample_index']}: {p['prediction_name']} (Confidence: {p['confidence'] * 100:.2f}%)")

    logger.info("Local run completed successfully!")


if __name__ == "__main__":
    main()
