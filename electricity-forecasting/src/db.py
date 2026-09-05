import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("database/forecasting.db")

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL,
            upload_time DATETIME NOT NULL,
            row_count INTEGER NOT NULL,
            start_date DATETIME,
            end_date DATETIME
        );

        CREATE TABLE IF NOT EXISTS model_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            train_start DATETIME,
            train_end DATETIME,
            created_at DATETIME NOT NULL,
            FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id)
        );

        CREATE TABLE IF NOT EXISTS evaluation_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            mae REAL,
            rmse REAL,
            mape REAL,
            is_best_model BOOLEAN DEFAULT 0,
            FOREIGN KEY(run_id) REFERENCES model_runs(run_id)
        );

        CREATE TABLE IF NOT EXISTS forecasts (
            forecast_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            timestamp DATETIME NOT NULL,
            actual_value REAL,
            predicted_value REAL,
            FOREIGN KEY(run_id) REFERENCES model_runs(run_id)
        );
        """)

def create_dataset(file_name, row_count, start_date, end_date):
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO datasets
            (file_name, upload_time, row_count, start_date, end_date)
            VALUES (?, ?, ?, ?, ?)
        """, (
            file_name,
            datetime.now().isoformat(timespec="seconds"),
            int(row_count),
            str(start_date),
            str(end_date),
        ))
        return cur.lastrowid

def create_model_run(dataset_id, model_name, train_start, train_end):
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO model_runs
            (dataset_id, model_name, train_start, train_end, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            dataset_id,
            model_name,
            str(train_start),
            str(train_end),
            datetime.now().isoformat(timespec="seconds"),
        ))
        return cur.lastrowid

def save_evaluation(run_id, mae, rmse, mape, is_best_model):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO evaluation_results
            (run_id, mae, rmse, mape, is_best_model)
            VALUES (?, ?, ?, ?, ?)
        """, (
            run_id, float(mae), float(rmse),
            None if mape != mape else float(mape),
            int(is_best_model),
        ))

def save_forecasts(run_id, timestamps, actual_values, predicted_values):
    with get_connection() as conn:
        conn.executemany("""
            INSERT INTO forecasts
            (run_id, timestamp, actual_value, predicted_value)
            VALUES (?, ?, ?, ?)
        """, [
            (
                run_id,
                str(ts),
                float(actual) if actual == actual else None,
                float(pred),
            )
            for ts, actual, pred in zip(
                timestamps, actual_values, predicted_values
            )
        ])

def get_latest_runs(limit=20):
    with get_connection() as conn:
        return conn.execute("""
            SELECT mr.run_id, d.file_name, mr.model_name,
                   er.mae, er.rmse, er.mape, er.is_best_model,
                   mr.created_at
            FROM model_runs mr
            JOIN datasets d ON d.dataset_id = mr.dataset_id
            LEFT JOIN evaluation_results er ON er.run_id = mr.run_id
            ORDER BY mr.created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
