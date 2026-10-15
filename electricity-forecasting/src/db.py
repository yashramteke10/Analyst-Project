import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path("database/forecasting.db")


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_db():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
          dataset_id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT NOT NULL,
          upload_time TEXT NOT NULL, row_count INTEGER, start_date TEXT, end_date TEXT
        );
        CREATE TABLE IF NOT EXISTS model_runs (
          run_id INTEGER PRIMARY KEY AUTOINCREMENT, dataset_id INTEGER,
          model_name TEXT NOT NULL, train_start TEXT, train_end TEXT, created_at TEXT NOT NULL,
          FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
        );
        CREATE TABLE IF NOT EXISTS evaluation_results (
          result_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER,
          mae REAL, rmse REAL, mape REAL, is_best_model INTEGER DEFAULT 0,
          FOREIGN KEY(run_id) REFERENCES model_runs(run_id)
        );
        CREATE TABLE IF NOT EXISTS forecasts (
          forecast_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER,
          timestamp TEXT, actual_value REAL, predicted_value REAL,
          FOREIGN KEY(run_id) REFERENCES model_runs(run_id)
        );
        """)


def save_run(file_name, df, model_name, metrics, actual, predicted, index):
    with get_connection() as conn:
        cur = conn.execute("INSERT INTO datasets(file_name, upload_time, row_count, start_date, end_date) VALUES(?,?,?,?,?)",
            (file_name, datetime.now().isoformat(timespec="seconds"), len(df), str(df.index.min()), str(df.index.max())))
        dataset_id = cur.lastrowid
        cur = conn.execute("INSERT INTO model_runs(dataset_id, model_name, train_start, train_end, created_at) VALUES(?,?,?,?,?)",
            (dataset_id, model_name, str(df.index.min()), str(df.index.max()), datetime.now().isoformat(timespec="seconds")))
        run_id = cur.lastrowid
        conn.execute("INSERT INTO evaluation_results(run_id,mae,rmse,mape,is_best_model) VALUES(?,?,?,?,1)",
                     (run_id, metrics["mae"], metrics["rmse"], metrics["mape"]))
        conn.executemany("INSERT INTO forecasts(run_id,timestamp,actual_value,predicted_value) VALUES(?,?,?,?)",
            [(run_id, str(t), float(a), float(p)) for t, a, p in zip(index, actual, predicted)])
