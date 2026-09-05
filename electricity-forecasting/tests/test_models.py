import numpy as np
from src.models.random_forest import train_random_forest, predict_random_forest

def test_random_forest(tmp_path):
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    path = tmp_path / "rf.pkl"
    model = train_random_forest(X, y, save_path=path, n_estimators=10)
    pred = predict_random_forest(model, X[:3])
    assert len(pred) == 3
