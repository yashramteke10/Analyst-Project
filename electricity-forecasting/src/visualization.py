import matplotlib.pyplot as plt
import pandas as pd

def plot_historical(df):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["datetime"], df["consumption"], linewidth=1.2)
    ax.set_title("Historical Electricity Consumption")
    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Consumption (kWh)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig

def plot_actual_vs_predicted(
    rf_times, actual, rf_pred, lstm_times=None, lstm_pred=None
):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rf_times, actual, label="Actual", linewidth=2)
    ax.plot(rf_times, rf_pred, label="Random Forest")
    if lstm_times is not None and lstm_pred is not None:
        ax.plot(lstm_times, lstm_pred, label="LSTM")
    ax.set_title("Actual vs Predicted Electricity Consumption")
    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Consumption (kWh)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig

def plot_forecast(history_times, history_values, forecast_times, forecast_values):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(history_times, history_values, label="Historical")
    ax.plot(forecast_times, forecast_values, marker="o", label="Forecast")
    ax.set_title("Future Electricity Consumption Forecast")
    ax.set_xlabel("Date/Time")
    ax.set_ylabel("Consumption (kWh)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig

def plot_model_comparison(metrics):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width / 2 for i in x], metrics["MAE"], width, label="MAE")
    ax.bar([i + width / 2 for i in x], metrics["RMSE"], width, label="RMSE")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics["Model"])
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig
