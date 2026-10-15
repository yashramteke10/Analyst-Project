from pathlib import Path
import pandas as pd
import streamlit as st
from src.data_pipeline import load_and_clean_data, save_processed_data, TARGET_COL
from src.models.random_forest import train_random_forest
from src.evaluation import calculate_metrics
from src.visualization import plot_actual_vs_predicted
from src.db import initialize_db, save_run

st.set_page_config(page_title="Electricity Consumption Forecasting", page_icon="⚡", layout="wide")
st.title("⚡ Electricity Consumption Forecasting")
st.write("Forecast hourly electricity consumption using machine learning.")
initialize_db()
DEFAULT_CSV = Path("data/raw/electricity_consumption_data.csv")

uploaded = st.file_uploader("Upload an electricity CSV (optional)", type=["csv"])
try:
    if uploaded is not None:
        raw_df = load_and_clean_data(uploaded)
        file_name = uploaded.name
    elif DEFAULT_CSV.exists():
        raw_df = load_and_clean_data(DEFAULT_CSV)
        file_name = DEFAULT_CSV.name
    else:
        st.info("Upload a CSV with 'datetime' and 'consumption_kWh' columns to begin.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read dataset: {exc}")
    st.stop()

st.success(f"Loaded {len(raw_df):,} clean rows.")
col1, col2, col3 = st.columns(3)
col1.metric("Records", f"{len(raw_df):,}")
col2.metric("Start", str(raw_df.index.min().date()))
col3.metric("End", str(raw_df.index.max().date()))
with st.expander("Preview dataset"):
    st.dataframe(raw_df.reset_index().head(20), use_container_width=True)

st.subheader("Consumption over time")
st.line_chart(raw_df[TARGET_COL])

st.subheader("Train Random Forest")
with st.sidebar:
    st.header("Model settings")
    n_estimators = st.slider("Number of trees", 50, 300, 150, 50)
    min_leaf = st.slider("Minimum samples per leaf", 1, 10, 2)
    run_model = st.button("Train and evaluate model", type="primary")

if run_model:
    with st.spinner("Training Random Forest using a chronological 80/20 split..."):
        try:
            save_processed_data(raw_df)
            result = train_random_forest(raw_df, n_estimators=n_estimators, min_samples_leaf=min_leaf)
            metrics = calculate_metrics(result["y_test"], result["predictions"])
            save_run(file_name, raw_df, "Random Forest", metrics, result["y_test"], result["predictions"], result["test_index"])
            st.session_state["rf_result"] = result
            st.session_state["rf_metrics"] = metrics
            st.success("Model training and evaluation completed.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

if "rf_result" in st.session_state:
    result = st.session_state["rf_result"]
    metrics = st.session_state["rf_metrics"]
    a, b, c = st.columns(3)
    a.metric("MAE", f"{metrics['mae']:.4f} kWh")
    b.metric("RMSE", f"{metrics['rmse']:.4f} kWh")
    c.metric("MAPE", f"{metrics['mape']:.2f}%" if metrics["mape"] is not None else "N/A")
    st.pyplot(plot_actual_vs_predicted(result["y_test"], result["predictions"], result["test_index"]))
    st.caption("The test set is the final 20% of the timeline; records are not randomly shuffled.")
    out = pd.DataFrame({"datetime": result["test_index"], "actual_kWh": result["y_test"].values,
                        "predicted_kWh": result["predictions"]})
    st.download_button("Download test predictions CSV", out.to_csv(index=False), "forecast_predictions.csv", "text/csv")

st.subheader("Optional LSTM model")
st.write("LSTM is available if TensorFlow is installed. Random Forest is the default MVP model.")
if st.button("Train LSTM (optional)"):
    try:
        from src.models.lstm_model import train_lstm
        with st.spinner("Training LSTM; this may take a few minutes..."):
            result_lstm = train_lstm(raw_df)
            metrics_lstm = calculate_metrics(result_lstm["y_test"], result_lstm["predictions"])
        st.write({"MAE": metrics_lstm["mae"], "RMSE": metrics_lstm["rmse"], "MAPE": metrics_lstm["mape"]})
        st.pyplot(plot_actual_vs_predicted(result_lstm["y_test"], result_lstm["predictions"], result_lstm["test_index"], "LSTM: Actual vs Predicted"))
    except Exception as exc:
        st.error(str(exc))
