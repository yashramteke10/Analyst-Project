# Electricity Consumption Forecasting Using AI and Machine Learning

A beginner-friendly Streamlit project using the supplied hourly electricity dataset. Random Forest is the default MVP model; LSTM is optional and requires TensorFlow.

## Dataset

The supplied CSV is stored at `data/raw/electricity_consumption_data.csv`. It contains `datetime`, `consumption_kWh`, calendar features, temperature, humidity and holiday information. The app also accepts another CSV with at least `datetime` and `consumption_kWh`.

## Run on Windows

Open this folder in VS Code, then open **Terminal → New Terminal** and run:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt terminal and run `.venv\Scripts\activate.bat`, or run `.venv\Scripts\python.exe -m streamlit run app.py` after installing packages.

## Run on macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How to use

1. The app loads the included dataset automatically, or upload another CSV.
2. Inspect the rows and consumption chart.
3. Use the sidebar settings and click **Train and evaluate model**.
4. Review MAE, RMSE, MAPE and the actual-vs-predicted chart.
5. Download test predictions as a CSV.
6. The database is created automatically at `database/forecasting.db`.

## Project structure

- `app.py`: Streamlit user interface and workflow
- `src/data_pipeline.py`: validation, cleaning, time features and lag features
- `src/models/random_forest.py`: Random Forest model with chronological test split
- `src/models/lstm_model.py`: optional TensorFlow LSTM
- `src/models/transformer_model.py`: placeholder for post-MVP work
- `src/evaluation.py`: MAE, RMSE and guarded MAPE
- `src/visualization.py`: actual-vs-predicted chart
- `src/db.py`: SQLite tables and run persistence
- `data/raw/`: original dataset
- `data/processed/`: processed CSV output
- `models_store/`: saved model artifacts can be added here

## Notes

- Data is synthetic/sample-style and covers 2024; forecasts demonstrate model workflow and are not operational utility forecasts.
- A chronological split is used to avoid random shuffling of time-series observations.
- Transformer, FastAPI and local LLM explanation are optional extensions, not required to run the MVP.
