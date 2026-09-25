"""House price regression pipeline using XGBoost."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from .config import RANDOM_STATE, TRAIN_TEST_SPLIT

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_dataset() -> pd.DataFrame:
    """Fetch the California housing dataset and convert it into a DataFrame.

    Returns:
        DataFrame containing feature columns and a target column named `price`.
    """
    logger.info("Loading California Housing dataset.")
    dataset = fetch_california_housing()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["price"] = dataset.target
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target variable.

    Args:
        df: Dataset containing a `price` column.

    Returns:
        Feature matrix and price target series.
    """
    if "price" not in df.columns:
        raise ValueError("The dataset must contain a 'price' column.")

    features = df.drop(columns=["price"], axis=1)
    target = df["price"]
    return features, target


def train_model(features: pd.DataFrame, target: pd.Series) -> XGBRegressor:
    """Train an XGBoost regressor.

    Args:
        features: Feature matrix for model training.
        target: Continuous regression target.

    Returns:
        Trained XGBoost regressor.
    """
    x_train, _, y_train, _ = train_test_split(
        features,
        target,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE,
    )

    model = XGBRegressor()
    logger.info("Training XGBoost regressor.")
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: XGBRegressor, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Evaluate the regression model using R² and MAE.

    Args:
        model: Trained regressor.
        x_test: Test feature matrix.
        y_test: Actual test target values.

    Returns:
        Dictionary with evaluation metrics.
    """
    predictions = model.predict(x_test)
    metrics_dict = {
        "r2_score": float(metrics.r2_score(y_test, predictions)),
        "mae": float(metrics.mean_absolute_error(y_test, predictions)),
    }
    logger.info("R² score: %.4f", metrics_dict["r2_score"])
    logger.info("MAE: %.4f", metrics_dict["mae"])
    return metrics_dict


def plot_correlation(df: pd.DataFrame) -> None:
    """Display a correlation heatmap for the dataset.

    Args:
        df: DataFrame used for exploratory analysis.
    """
    correlation = df.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={"size": 8}, cmap="pink")
    plt.title("Feature correlation heatmap")
    plt.show()


def predict_new_house(model: XGBRegressor, values: list[float]) -> float:
    """Predict the price of a house from a feature vector.

    Args:
        model: Trained regressor.
        values: Feature values for a single house.

    Returns:
        Predicted house price.
    """
    array = np.asarray([values], dtype=float)
    prediction = model.predict(array)
    return float(prediction[0])


def main() -> None:
    """Run the full California housing workflow."""
    df = load_dataset()
    plot_correlation(df)
    features, target = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE,
    )
    model = train_model(x_train, y_train)
    metrics_dict = evaluate_model(model, x_test, y_test)

    sample_input = [4.1, 0.4, 1.2, 0.5, 4.2, 3.2, 2.1, 2.2]
    prediction = predict_new_house(model, sample_input)

    print(f"R² score: {metrics_dict['r2_score']:.4f}")
    print(f"MAE: {metrics_dict['mae']:.4f}")
    print(f"Predicted price for new house: ${prediction * 100000:.2f}")


if __name__ == "__main__":
    main()
