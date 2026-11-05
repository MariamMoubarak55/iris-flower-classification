"""Data loader module for Iris dataset.

Handles downloading/loading the Iris dataset with proper logging and typing (Slide 10 & 18).
"""

from pathlib import Path
from typing import Optional
import pandas as pd
from sklearn.datasets import load_iris

from src.utils.logger import get_logger
from src.conf.config import get_default

logger = get_logger(__name__)


def load_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load Iris dataset from CSV or create it using sklearn datasets.

    Args:
        csv_path: Optional path to CSV file. If None or not found,
                  falls back to sklearn's load_iris and saves it to data_path.

    Returns:
        pd.DataFrame: DataFrame containing Iris features and target 'species'.
    """
    config = get_default()
    if csv_path is None:
        csv_path = config.get("paths", {}).get("data_path", "data/iris.csv")

    path = Path(csv_path)

    if path.exists():
        logger.info(f"Loading Iris dataset from file: {path.resolve()}")
        df = pd.read_csv(path)
    else:
        logger.info("Dataset CSV not found locally. Fetching Iris from sklearn.datasets...")
        raw_iris = load_iris(as_frame=True)
        df = raw_iris.frame.copy()
        
        # Standardize column names
        df.columns = [
            c.replace(" (cm)", "").replace(" ", "_") for c in df.columns
        ]
        if "target" in df.columns:
            df.rename(columns={"target": "species"}, inplace=True)
        
        # Map target numbers to species names for readability
        species_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
        df["species_name"] = df["species"].map(species_names)

        # Ensure directory exists and save CSV for reproducibility
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Saved Iris dataset locally to: {path.resolve()}")

    logger.info(f"Dataset successfully loaded with shape: {df.shape}")
    return df
