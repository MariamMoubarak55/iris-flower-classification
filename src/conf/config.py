"""Configuration management module.

Loads parameters and settings from design.json (as specified in Slide 19).
"""

from pathlib import Path
import json
from typing import Any, Dict


CONFIG_FILE_PATH = Path(__file__).resolve().parent / "design.json"


def get_default() -> Dict[str, Any]:
    """Load and return default project configurations from design.json.

    Returns:
        Dict[str, Any]: Configuration dictionary containing parameters for
        dataset, training, paths, and server.
    """
    if not CONFIG_FILE_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found at: {CONFIG_FILE_PATH}")
    
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)
    
    return config


# Quick access to key configuration blocks
_CONFIG = get_default()
NUM_FOLDS: int = _CONFIG.get("training", {}).get("num_folds", 5)
SCORING: str = _CONFIG.get("training", {}).get("scoring", "accuracy")
TEST_SIZE: float = _CONFIG.get("dataset", {}).get("test_size", 0.2)
RANDOM_STATE: int = _CONFIG.get("dataset", {}).get("random_state", 42)
MODEL_PATH: str = _CONFIG.get("paths", {}).get("model_path", "src/resources/model.joblib")
LOG_FILE: str = _CONFIG.get("paths", {}).get("log_file", "logs.log")
