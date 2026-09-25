"""Diabetes prediction pipeline using a linear SVM classifier."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm

from .config import DATA_PATH, RANDOM_STATE, TARGET_COLUMN, TRAIN_TEST_SPLIT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the diabetes dataset from disk.

    Args:
        data_path: Relative or absolute path to the dataset CSV file.

    Returns:
        A pandas DataFrame containing clinical measurement columns and the target.

    Raises:
        FileNotFoundError: If the dataset path does not exist.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    logger.info("Loading dataset from %s", path)
    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> tuple[np.ndarray, pd.Series]:
    """Separate explanatory variables from the outcome variable.

    Args:
        df: Input data frame containing the target column.
        target_column: Column name representing the diabetes outcome.

    Returns:
        A tuple of feature matrix and target series.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    features = df.drop(columns=target_column, axis=1)
    target = df[target_column]
    return features.to_numpy(dtype=float), target


def train_and_evaluate_model(features: np.ndarray, target: pd.Series) -> tuple[svm.SVC, dict[str, float]]:
    """Train a linear SVM and return evaluation metrics.

    Args:
        features: Standardized feature matrix.
        target: Target labels.

    Returns:
        A tuple containing the trained model and evaluation metrics.
    """
    logger.info("Splitting dataset into train and test sets.")
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TRAIN_TEST_SPLIT,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = svm.SVC(kernel="linear")
    logger.info("Training SVM classifier.")
    model.fit(x_train_scaled, y_train)

    train_predictions = model.predict(x_train_scaled)
    test_predictions = model.predict(x_test_scaled)

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
    }

    logger.info("Training accuracy: %.4f", metrics["train_accuracy"])
    logger.info("Test accuracy: %.4f", metrics["test_accuracy"])
    return model, metrics


def predict_single_instance(model: svm.SVC, scaler: StandardScaler, values: list[float]) -> int:
    """Predict a single patient status from a feature list.

    Args:
        model: Trained SVM model.
        scaler: Fitted scaler used for preprocessing.
        values: A list of feature values in the same order as the dataset columns.

    Returns:
        Prediction label: 0 or 1.

    Raises:
        ValueError: If the supplied feature count does not match the model input dimension.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input values must be a 1D array-like object.")

    scaled = scaler.transform(arr.reshape(1, -1))
    prediction = model.predict(scaled)
    return int(prediction[0])


def main() -> None:
    """Run the diabetes prediction workflow."""
    df = load_dataset(DATA_PATH)
    features, target = prepare_features(df)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    model, metrics = train_and_evaluate_model(scaled_features, target)

    sample_input = [6, 148, 72, 35, 0, 33.6, 0.627, 50]
    prediction = predict_single_instance(model, scaler, sample_input)
    logger.info("Prediction for sample input: %s", prediction)

    print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Sample prediction: {prediction}")


if __name__ == "__main__":
    main()
