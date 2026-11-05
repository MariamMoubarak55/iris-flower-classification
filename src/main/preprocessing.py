"""Data preprocessing and feature preparation module.

Provides typed and documented functions for splitting and preprocessing (Slide 10 & 18).
"""

from typing import Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.conf.config import get_default

logger = get_logger(__name__)

FEATURE_COLUMNS: List[str] = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]


def split_data(
    df: pd.DataFrame,
    target_column: str = "species",
    test_size: Optional[float] = None,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split dataset into stratified train and test subsets (Slide 10 & 18).

    Args:
        df: Input DataFrame containing features and target.
        target_column: Name of target column (default: 'species').
        test_size: Proportion of the dataset to include in test split.
        random_state: Random state seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            X_train, X_test, y_train, y_test
    """
    config = get_default()
    if test_size is None:
        test_size = config.get("dataset", {}).get("test_size", 0.2)
    if random_state is None:
        random_state = config.get("dataset", {}).get("random_state", 42)

    logger.info("Splitting train and test sets")

    # Select only the feature columns
    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    X = df[available_features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info(
        f"Split complete. Train size: {len(X_train)} samples, Test size: {len(X_test)} samples."
    )
    return X_train, X_test, y_train, y_test


def sample_stratified_split(
    df: pd.DataFrame,
    n_samples: int = 5,
    group_col: str = "species"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Alternative stratified sampling function as demonstrated in Slide 10.

    Args:
        df: Input DataFrame.
        n_samples: Number of samples per class for test set.
        group_col: Target column to stratify by.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
    """
    logger.info(f"Extracting stratified sample of {n_samples} per class in '{group_col}'")
    n = min(n_samples, df[group_col].value_counts().min())
    
    test_df = df.groupby(group_col, group_keys=False).sample(n=n, random_state=42)
    train_df = df.loc[~df.index.isin(test_df.index)]

    return train_df, test_df
