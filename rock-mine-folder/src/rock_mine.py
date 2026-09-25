"""Rock-vs-mine classification pipeline using logistic regression."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, RANDOM_STATE, TARGET_COLUMN, TRAIN_TEST_SPLIT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the sonar dataset into a DataFrame.

    Args:
        data_path: Relative or absolute path to the CSV file.

    Returns:
        DataFrame with the target in the final column.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading sonar dataset from %s", path)
    return pd.read_csv(path, header=None)


def prepare_features(df: pd.DataFrame, target_column: int = TARGET_COLUMN) -> tuple[pd.DataFrame, pd.Series]:
    """Extract the feature matrix and target series from the sonar data.

    Args:
        df: DataFrame containing numeric sonar values.
        target_column: Index of the target label column.

    Returns:
        Tuple of feature DataFrame and target series.
    """
    if target_column >= df.shape[1]:
        raise ValueError(f"Target column index {target_column} is out of range for dataset width {df.shape[1]}.")

    features = df.drop(columns=target_column, axis=1)
    target = df[target_column]
    return features, target


def train_model(features: pd.DataFrame, target: pd.Series) -> LogisticRegression:
    """Train a logistic regression model on the sonar dataset.

    Args:
        features: Model input data.
        target: Target labels for rock vs mine.

    Returns:
        Trained logistic regression classifier.
    """
    x_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=TRAIN_TEST_SPLIT,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    model = LogisticRegression()
    logger.info("Training logistic regression model.")
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute training and test accuracy for the model.

    Args:
        model: Trained classifier.
        x_test: Test feature matrix.
        y_test: Ground truth labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    test_predictions = model.predict(x_test)
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
    }
    logger.info("Test accuracy: %.4f", metrics["test_accuracy"])
    return metrics


def predict_single_sample(model: LogisticRegression, values: list[float]) -> str:
    """Predict whether a sonar sample corresponds to a rock or a mine.

    Args:
        model: Trained classifier.
        values: Feature values from a single sonar sample.

    Returns:
        Prediction label: either 'R' or 'M'.
    """
    array = np.asarray(values, dtype=float).reshape(1, -1)
    prediction = model.predict(array)
    return str(prediction[0])


def main() -> None:
    """Run the rock-vs-mine classification workflow."""
    df = load_dataset(DATA_PATH)
    features, target = prepare_features(df)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TRAIN_TEST_SPLIT,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    model = train_model(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)

    sample_input = [0.0317, 0.0956, 0.1321, 0.1408, 0.1674, 0.1710, 0.0731, 0.1401, 0.2083, 0.3513, 0.1786, 0.0658, 0.0513, 0.3752, 0.5419, 0.5440, 0.5150, 0.4262, 0.2024, 0.4233, 0.7723, 0.9735, 0.9390, 0.5559, 0.5268, 0.6826, 0.5713, 0.5429, 0.2177, 0.2149, 0.5811, 0.6323, 0.2965, 0.1873, 0.2969, 0.5163, 0.6153, 0.4283, 0.5479, 0.6133, 0.5017, 0.2377, 0.1957, 0.1749, 0.1304, 0.0597, 0.1124, 0.1047, 0.0507, 0.0159, 0.0195, 0.0201, 0.0248, 0.0131, 0.0070, 0.0138, 0.0092, 0.0143, 0.0036, 0.0103]
    prediction = predict_single_sample(model, sample_input)

    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
