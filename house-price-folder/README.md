# House Price Prediction

## Problem Statement

This project predicts the sale price of a house based on features such as median income, house age, average rooms, and other neighborhood indicators from the California Housing dataset.

## Dataset

The original script used the built-in California Housing dataset from `sklearn.datasets.fetch_california_housing()`. The resulting data is transformed into a pandas DataFrame and used for regression modeling.

## Workflow

1. Load the California Housing dataset.
2. Convert the data to a DataFrame with feature names.
3. Add the target column (`price`).
4. Explore statistical summaries and correlation patterns.
5. Split the data into training and test subsets.
6. Train an XGBoost regressor.
7. Evaluate with `R²` and `Mean Absolute Error`.
8. Visualize actual vs. predicted values.

## Model

Algorithm: XGBoost Regressor

- Training method: supervised regression
- Evaluation metrics: `R²` and `MAE`
- Split strategy: 80/20 random split

## Evaluation

The original script reports results using:

- `R²` score for training and test predictions
- `Mean Absolute Error` for training and test predictions

These are used as a baseline for the educational regression example and should be refreshed when run in a local environment.

## Sample Prediction

The model supports a single-house input array and returns a monetary estimate. The original example scaled the result to represent approximate dollar value.

## Project Structure

```text
house-price-folder/
├── src/
│   ├── config.py
│   └── house_price.py
├── tests/
│   └── test_house_price.py
├── README.md
└── notebooks/
```
