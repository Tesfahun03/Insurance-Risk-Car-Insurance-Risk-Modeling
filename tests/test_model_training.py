import pytest
import pandas as pd
import numpy as np
from src.model_training import DataSplitter, ModelTrainer, ModelEvaluator
from sklearn.linear_model import LinearRegression


@pytest.fixture
def sample_data():
    """Fixture to provide sample data for modeling."""
    X = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    y = pd.Series([10, 20, 30])
    return X, y


def test_split_data(sample_data):
    X, y = sample_data
    splitter = DataSplitter(X, y)
    X_train, X_test, y_train, y_test = splitter.split_data()
    assert len(X_train) == 2  # 80% of 3 rows
    assert len(X_test) == 1   # 20% of 3 rows


def test_linear_regression(sample_data):
    X, y = sample_data
    trainer = ModelTrainer(X, y)
    model = trainer.linear_regression()
    assert isinstance(model, LinearRegression)


def test_evaluate_model(sample_data):
    X, y = sample_data
    trainer = ModelTrainer(X, y)
    model = trainer.linear_regression()
    evaluator = ModelEvaluator()
    mae, mse, r2, y_pred = evaluator.evaluate_model(model, X, y)
    assert isinstance(mae, float)
    assert isinstance(mse, float)
    assert isinstance(r2, float)
    assert len(y_pred) == len(y)
