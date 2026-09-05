import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_pipeline import process_dataframe, save_processed_data
from src.evaluation import compare_models
from src.visualization import (
    plot_historical,
    plot_actual_vs_predicted,
    plot_forecast,
    plot_model_comparison,
)
from src.db import (
    init_db,
    create_dataset,
    create_model_run,
    save_evaluation,
    save_forecasts,
)
from src.models.random_forest import train_random_forest, predict_random_forest
from src.models.lstm_model import (
    prepare_lstm_data,
    train_lstm,
    predict_lstm,
    forecast_next,
)

st.set_page_config(
    page_title="Electricity Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
)

init_db()

st.title("⚡ Electricity Consumption Forecasting")
st.caption("Using AI and Machine Learning — Random Forest + LSTM")

with st.sidebar:
    st.header("Forecasting Controls")
    uploaded_file = st.file_uploader(
        "Upload electricity CSV",
        type=["csv"],
        help="Required: datetime/date/time and consumption/demand column.",
    )
    test_percent = st.slider("Test set (%)", 15, 30, 20, 5)
    lookback = st.slider("LSTM lookback steps", 12, 168, 24, 12)
    epochs = st.slider("LSTM epochs", 1, 30, 10)
    run = st.button("🚀 Train & Forecast", type="primary", use_container_width=True)

if uploaded_file is None:
    st.info("Upload your historical electricity CSV to begin.")
    st.markdown("""
### Minimum input format

| datetime | consumption_kwh |
|---|---:|
| 2025-01-01 00:00:00 | 4.32 |
| 2025-01-01 01:00:00 | 4.05 |
| 2025-01-01 02:00:00 | 3.88 |

Optional columns such as temperature, humidity and holiday indicators can also
be kept in the source dataset.
""")
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read the CSV file: {exc}")
    st.stop()

# Preserve original upload in data/raw/
raw_dir = ROOT / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)
(raw_dir / uploaded_file.name).write_bytes(uploaded_file.getvalue())

st.subheader("1. Dataset Preview")
st.dataframe(raw_df.head(10), use_container_width=True)
st.write(f"**Rows:** {len(raw_df):,} &nbsp;&nbsp; **Columns:** {len(raw_df.columns)}")

try:
    cleaned_df, report = process_dataframe(raw_df)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Preprocessing Summary", expanded=True):
    cols = st.columns(5)
    cols[0].metric("Original rows", report["original_rows"])
    cols[1].metric("Duplicates removed", report["duplicates_removed"])
    cols[2].metric("Missing values filled", report["missing_values_filled"])
    cols[3].metric("Invalid datetime", report["invalid_datetime_rows"])
    cols[4].metric("Final rows", report["final_rows"])

processed_path = save_processed_data(cleaned_df, ROOT / "data" / "processed")

st.subheader("2. Exploratory Data Analysis")
m = st.columns(4)
m[0].metric("Mean", f"{cleaned_df['consumption'].mean():.3f}")
m[1].metric("Minimum", f"{cleaned_df['consumption'].min():.3f}")
m[2].metric("Maximum", f"{cleaned_df['consumption'].max():.3f}")
m[3].metric("Std. deviation", f"{cleaned_df['consumption'].std():.3f}")

st.pyplot(plot_historical(cleaned_df), use_container_width=True)

daily = cleaned_df.set_index("datetime")["consumption"].resample("D").mean()
weekly = cleaned_df.set_index("datetime")["consumption"].resample("W").mean()
monthly = cleaned_df.set_index("datetime")["consumption"].resample("ME").mean()

c1, c2, c3 = st.columns(3)
with c1:
    st.write("Daily average")
    st.line_chart(daily)
with c2:
    st.write("Weekly average")
    st.line_chart(weekly)
with c3:
    st.write("Monthly average")
    st.line_chart(monthly)

if not run:
    st.warning("Click **Train & Forecast** to train the Random Forest and LSTM models.")
    st.stop()

if len(cleaned_df) < max(100, lookback + 30):
    st.error(
        f"Not enough records. Please use at least {max(100, lookback + 30)} rows "
        "for the selected lookback."
    )
    st.stop()

dataset_id = create_dataset(
    file_name=uploaded_file.name,
    row_count=len(cleaned_df),
    start_date=cleaned_df["datetime"].min(),
    end_date=cleaned_df["datetime"].max(),
)

with st.status("Running forecasting pipeline...", expanded=True) as status:
    from src.data_pipeline import create_features, chronological_split

    st.write("Creating calendar, lag and rolling features...")
    feature_df = create_features(cleaned_df)

    feature_columns = [
        "hour", "day_of_week", "day_of_month", "month", "day_of_year",
        "is_weekend", "lag_1", "lag_2", "lag_3", "lag_6",
        "lag_12", "lag_24", "lag_168",
        "rolling_mean_3", "rolling_mean_6",
        "rolling_mean_12", "rolling_mean_24", "rolling_mean_168",
    ]
    feature_columns = [c for c in feature_columns if c in feature_df.columns]

    rf_df = feature_df.dropna(subset=feature_columns + ["consumption"]).copy()
    train_df, test_df = chronological_split(
        rf_df, test_size=test_percent / 100
    )

    st.write("Training Random Forest baseline...")
    rf_model = train_random_forest(
        train_df[feature_columns],
        train_df["consumption"],
        save_path=ROOT / "models_store" / "random_forest_model.pkl",
    )
    rf_pred = predict_random_forest(rf_model, test_df[feature_columns])

    rf_run_id = create_model_run(
        dataset_id,
        "RandomForest",
        train_df["datetime"].min(),
        train_df["datetime"].max(),
    )

    st.write("Preparing LSTM sliding-window sequences...")
    X_train, y_train, X_test, y_test, scaler, test_times = prepare_lstm_data(
        cleaned_df, lookback=lookback, test_size=test_percent / 100
    )

    st.write("Training LSTM...")
    lstm_model = train_lstm(
        X_train,
        y_train,
        epochs=epochs,
        save_path=ROOT / "models_store" / "lstm_model.h5",
    )

    lstm_pred = predict_lstm(lstm_model, X_test, scaler)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    lstm_run_id = create_model_run(
        dataset_id,
        "LSTM",
        cleaned_df["datetime"].iloc[0],
        cleaned_df["datetime"].iloc[int(len(cleaned_df) * (1 - test_percent / 100)) - 1],
    )

    status.update(label="Training completed successfully", state="complete")

actuals = {
    "RandomForest": test_df["consumption"].to_numpy(),
    "LSTM": y_test_actual,
}
predictions = {
    "RandomForest": rf_pred,
    "LSTM": lstm_pred,
}

metrics = compare_models(actuals, predictions)
winner = metrics.sort_values(["RMSE", "MAE"]).iloc[0]["Model"]

rf_metrics = metrics[metrics["Model"] == "RandomForest"].iloc[0]
lstm_metrics = metrics[metrics["Model"] == "LSTM"].iloc[0]

save_evaluation(
    rf_run_id,
    rf_metrics["MAE"],
    rf_metrics["RMSE"],
    rf_metrics["MAPE"],
    winner == "RandomForest",
)
save_evaluation(
    lstm_run_id,
    lstm_metrics["MAE"],
    lstm_metrics["RMSE"],
    lstm_metrics["MAPE"],
    winner == "LSTM",
)

save_forecasts(
    rf_run_id,
    test_df["datetime"].tolist(),
    actuals["RandomForest"],
    predictions["RandomForest"],
)
save_forecasts(
    lstm_run_id,
    test_times.tolist(),
    actuals["LSTM"],
    predictions["LSTM"],
)

st.subheader("3. Model Evaluation")
st.dataframe(metrics, use_container_width=True)
st.success(f"🏆 Best model by RMSE: **{winner}**")
st.pyplot(plot_model_comparison(metrics), use_container_width=True)

st.subheader("4. Actual vs Predicted")
st.pyplot(
    plot_actual_vs_predicted(
        test_df["datetime"],
        actuals["RandomForest"],
        rf_pred,
        test_times,
        lstm_pred,
    ),
    use_container_width=True,
)

st.subheader("5. Next-Step Forecast")
try:
    next_value = forecast_next(
        lstm_model,
        cleaned_df["consumption"].to_numpy(),
        scaler,
        lookback,
    )
    if len(cleaned_df) >= 2:
        step = cleaned_df["datetime"].iloc[-1] - cleaned_df["datetime"].iloc[-2]
        next_timestamp = cleaned_df["datetime"].iloc[-1] + step
    else:
        next_timestamp = None

    c1, c2 = st.columns(2)
    c1.metric("Predicted next consumption", f"{next_value:.3f} kWh")
    c2.metric(
        "Next timestamp",
        str(next_timestamp) if next_timestamp is not None else "N/A",
    )

    future_times = pd.date_range(
        start=next_timestamp,
        periods=1,
        freq=step,
    ) if next_timestamp is not None else pd.DatetimeIndex([])

    if len(future_times):
        st.pyplot(
            plot_forecast(
                cleaned_df["datetime"].tail(min(168, len(cleaned_df))),
                cleaned_df["consumption"].tail(min(168, len(cleaned_df))),
                future_times,
                [next_value],
            ),
            use_container_width=True,
        )
except Exception as exc:
    st.warning(f"Next-step forecast unavailable: {exc}")

st.subheader("6. Downloads")
st.download_button(
    "Download processed dataset",
    cleaned_df.to_csv(index=False).encode("utf-8"),
    "processed_electricity_data.csv",
    "text/csv",
)

st.caption(
    f"Processed data saved to: {processed_path.name} | "
    "MVP: Random Forest + LSTM + Streamlit + SQLite"
)
