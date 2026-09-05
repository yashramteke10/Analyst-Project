import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def calculate_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def compare_models(actuals, predictions):
    rows = []
    for model_name, y_pred in predictions.items():
        y_true = actuals[model_name]
        rows.append({
            "Model": model_name,
            "MAE": calculate_mae(y_true, y_pred),
            "RMSE": calculate_rmse(y_true, y_pred),
            "MAPE": calculate_mape(y_true, y_pred),
        })
    return pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)

def select_best_model(metrics):
    return metrics.sort_values(["RMSE", "MAE"]).iloc[0]["Model"]
