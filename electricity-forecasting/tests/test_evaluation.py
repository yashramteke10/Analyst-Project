from src.evaluation import calculate_metrics


def test_metrics():
    metrics = calculate_metrics([1, 2, 3], [1, 2, 4])
    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert metrics["mape"] is not None
