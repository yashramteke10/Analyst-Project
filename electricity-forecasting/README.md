# Electricity Consumption Forecasting Using AI and Machine Learning

A student-friendly machine learning project that analyzes hourly electricity consumption and forecasts future usage using a Streamlit dashboard.

## Project Overview

Electricity demand changes by hour, day, and season. This project uses historical electricity-consumption data and machine learning to learn usage patterns, produce predictions, and compare model performance.

### Objectives

- Load and inspect electricity consumption data from a CSV file.
- Clean and prepare timestamped data for modelling.
- Create time-based and lag features for forecasting.
- Train and evaluate forecasting models.
- Visualize actual consumption against model predictions.
- Store project metadata and results locally where supported by the application.

## Dataset

The project dataset is stored at:

```text
data/raw/electricity_consumption_data.csv
```

The supplied dataset contains hourly observations for 2024. The main target column is `consumption_kWh`.

| Column | Description |
|---|---|
| `datetime` | Timestamp for the observation |
| `hour` | Hour of the day |
| `day_of_week` | Day of the week |
| `month` | Month number |
| `is_weekend` | Indicates whether the date is a weekend |
| `is_holiday` | Indicates whether the date is marked as a holiday |
| `temperature_C` | Temperature in degrees Celsius |
| `humidity_pct` | Relative humidity percentage |
| `consumption_kWh` | Electricity consumption in kilowatt-hours; forecasting target |

Keep the column names consistent with the dataset or update the data-processing code to match your own CSV.

## Technology Stack

- **Python 3.10 or newer** — main programming language
- **Pandas and NumPy** — data loading and preparation
- **scikit-learn** — Random Forest modelling and evaluation utilities
- **TensorFlow/Keras** — LSTM model support, if included and installed
- **Streamlit** — interactive web dashboard
- **Matplotlib / Streamlit charts** — visualization
- **SQLite** — local metadata/results storage, where implemented

## Project Structure

```text
electricity-forecasting/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   │   └── electricity_consumption_data.csv
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── db.py
│   └── models/
│       ├── __init__.py
│       ├── random_forest.py
│       ├── lstm_model.py
│       └── transformer_model.py
├── models_store/
├── database/
└── docs/
```

Some folders or optional modules may be present only if they were included in your project version.

## Installation and Setup

### 1. Install Python

Install Python 3.10 or newer. During installation on Windows, enable **Add Python to PATH** if that option is shown.

Check your installation:

```bash
python --version
```

### 2. Open the project

Extract the project ZIP file, then open the `electricity-forecasting` folder in VS Code.

### 3. Create a virtual environment

Run these commands in the VS Code terminal from the project root.

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the project has optional LSTM functionality and TensorFlow is not in `requirements.txt`, install a TensorFlow version compatible with your Python version:

```bash
pip install tensorflow
```

### 5. Start the application

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, commonly `http://localhost:8501`. Open that URL in your browser if it does not open automatically.

## How to Use the Application

1. Start the Streamlit application.
2. Confirm that the electricity dataset is loaded, or upload a CSV if the dashboard provides an upload option.
3. Review the dataset summary and consumption chart.
4. Choose the available model and its parameters.
5. Start model training and evaluation.
6. Review the evaluation metrics and actual-versus-predicted chart.
7. Compare models if the dashboard provides model comparison.

The exact buttons and options depend on the current implementation in `app.py`.

## Data Processing Workflow

The data pipeline is designed around timestamped electricity-consumption data. Typical processing steps include:

1. Read the CSV file.
2. Check that the timestamp and consumption columns exist.
3. Parse timestamps and sort observations chronologically.
4. Check for missing values and duplicate timestamps.
5. Prepare time-based features such as hour, day of week, and month.
6. Create lag and rolling features when required by the model.
7. Split data chronologically into training and testing portions.

For time-series forecasting, do not randomly shuffle observations before splitting the data. Keeping the test period later than the training period better reflects forecasting on future data.

## Models

### Random Forest Regressor

Random Forest is a tree-based regression model that can learn relationships between consumption and engineered features. It is a useful baseline because it is relatively straightforward to train and compare.

### LSTM (Long Short-Term Memory)

LSTM is a recurrent neural network designed to learn patterns in sequences. It can use a sequence of past observations to predict a future consumption value. TensorFlow/Keras must be installed if the LSTM module is enabled.

### Transformer (Optional)

A Transformer model can be explored as a future extension. It is not required for the minimum viable project unless the corresponding module and dashboard integration have been completed.

## Evaluation Metrics

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted consumption. Lower is better.
- **RMSE (Root Mean Squared Error):** Gives larger errors more weight. Lower is better.
- **MAPE (Mean Absolute Percentage Error):** Expresses error as a percentage. It can be misleading or undefined when actual values are zero or close to zero, so handle those values carefully.

Use the same chronological test period when comparing models.

## Troubleshooting

### `streamlit` is not recognized

Make sure the virtual environment is activated and dependencies are installed. You can also run:

```bash
python -m streamlit run app.py
```

### Module import errors

Run commands from the main project folder, where `app.py` and `requirements.txt` are located. Activate the project's virtual environment and reinstall the requirements if needed.

### CSV or column error

Check that the CSV is located at `data/raw/electricity_consumption_data.csv` and that its columns match those expected by the data pipeline, especially `datetime` and `consumption_kWh`.

### TensorFlow installation or import error

LSTM support is optional. Check your Python version and install a TensorFlow version that supports it, or use the Random Forest model if that is sufficient for your current project version.

### Port already in use

Run Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

## Limitations

- Forecast quality depends on the amount, quality, and representativeness of the historical data.
- The supplied dataset is a sample dataset and should not be treated as a real electricity-grid operational forecast.
- Results from a test split do not guarantee future performance.
- Weather, holiday, and other external features can improve a model only when they are accurate and available for the forecast period.
- Transformer modelling, automated multi-step forecasts, explainability, and local LLM explanations are optional extensions unless explicitly implemented.

## Future Enhancements

- Add multi-step forecasting for the next 24 hours or week.
- Integrate real weather forecasts as external features.
- Add a Transformer-based forecasting model.
- Add anomaly detection for unusual consumption.
- Use SHAP or similar techniques to explain model predictions.
- Add automated tests and containerized deployment.

## Academic Project

**Project title:** Electricity Consumption Forecasting Using AI and Machine Learning

This project is intended for educational use and demonstrates a basic end-to-end time-series machine learning workflow.
