import numpy as np
from src.evaluation import calculate_mae, calculate_rmse, calculate_mape

def test_metrics():
    actual = np.array([1, 2, 3])
    predicted = np.array([1, 3, 2])
    assert calculate_mae(actual, predicted) == 2 / 3
    assert round(calculate_rmse(actual, predicted), 6) == round(np.sqrt(2 / 3), 6)
    assert calculate_mape(actual, predicted) > 0

def test_mape_zero_values():
    actual = np.array([0, 0])
    predicted = np.array([1, 2])
    assert np.isnan(calculate_mape(actual, predicted))
