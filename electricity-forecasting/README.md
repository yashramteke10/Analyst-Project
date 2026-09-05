# Electricity Consumption Forecasting Using AI and Machine Learning

## MVP

The MVP implements:

1. CSV upload
2. Data cleaning
3. Missing-value handling
4. Duplicate removal
5. Datetime conversion and chronological sorting
6. Feature engineering
7. Chronological train/test split
8. Random Forest baseline
9. LSTM forecasting
10. MAE/RMSE/MAPE evaluation
11. Best-model selection
12. Historical and actual-vs-predicted visualization
13. Next-step forecast
14. Streamlit dashboard
15. SQLite metadata, run, evaluation and forecast storage

## Directory

```text
electricity-forecasting/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
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
│   ├── random_forest_model.pkl
│   ├── lstm_model.h5
│   └── transformer_model.h5
├── database/
│   └── forecasting.db
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── routes/
│       ├── __init__.py
│       ├── upload.py
│       ├── train.py
│       └── predict.py
├── llm/
│   └── explain.py
├── notebooks/
├── tests/
└── docs/
```

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Run Streamlit:

```bash
streamlit run app.py
```

Optional API:

```bash
uvicorn api.main:app --reload
```

## Dataset

Minimum required fields are a Date/Time field and electricity
consumption/demand. The application recognizes common names including:

- datetime
- date/time
- timestamp
- date
- consumption
- consumption_kwh
- power
- demand
- load
- energy

Hourly data is recommended.

## Important time-series rule

The project uses chronological splitting. It does not randomly shuffle
records before testing, because future observations must not leak into training.

## Optional components

The Transformer, FastAPI, notebooks, tests and Ollama LLM assistant are kept
separate from the MVP. Build and validate Random Forest + LSTM + Streamlit
first.
