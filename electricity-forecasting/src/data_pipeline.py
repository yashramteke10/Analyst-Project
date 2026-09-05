from pathlib import Path
import numpy as np
import pandas as pd

DATETIME_ALIASES = [
    "datetime", "date/time", "date_time", "timestamp", "date", "time"
]
CONSUMPTION_ALIASES = [
    "consumption", "consumption_kwh", "electricity consumption",
    "energy", "energy_kwh", "power", "demand", "load", "usage"
]

def _find_column(columns, aliases):
    normalized = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    for column in columns:
        low = str(column).strip().lower()
        if any(alias in low for alias in aliases):
            return column
    return None

def process_dataframe(df):
    if df is None or df.empty:
        raise ValueError("The uploaded CSV is empty.")

    out = df.copy()
    original_rows = len(out)

    datetime_col = _find_column(out.columns, DATETIME_ALIASES)
    consumption_col = _find_column(out.columns, CONSUMPTION_ALIASES)

    if datetime_col is None:
        raise ValueError(
            "Date/Time column not found. Use datetime, Date/Time, timestamp, date, etc."
        )
    if consumption_col is None:
        raise ValueError(
            "Consumption column not found. Use consumption_kwh, consumption, "
            "power, demand, load, energy, etc."
        )

    out = out.rename(columns={
        datetime_col: "datetime",
        consumption_col: "consumption",
    })

    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["consumption"] = pd.to_numeric(out["consumption"], errors="coerce")

    invalid_datetime_rows = int(out["datetime"].isna().sum())
    out = out.dropna(subset=["datetime"]).copy()

    before_duplicates = len(out)
    out = out.drop_duplicates(subset=["datetime"], keep="last")
    duplicates_removed = before_duplicates - len(out)

    missing_values = int(out["consumption"].isna().sum())
    if missing_values:
        out["consumption"] = (
            out["consumption"]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    out = out.dropna(subset=["consumption"]).copy()
    out["consumption"] = out["consumption"].clip(lower=0)
    out = out.sort_values("datetime").reset_index(drop=True)

    if out["consumption"].isna().any():
        raise ValueError("Missing consumption values remain after preprocessing.")

    report = {
        "original_rows": original_rows,
        "duplicates_removed": duplicates_removed,
        "missing_values_filled": missing_values,
        "invalid_datetime_rows": invalid_datetime_rows,
        "final_rows": len(out),
    }

    return out[["datetime", "consumption"]], report

def create_features(df):
    out = df.copy()
    dt = out["datetime"]

    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["day_of_month"] = dt.dt.day
    out["month"] = dt.dt.month
    out["day_of_year"] = dt.dt.dayofyear
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    for lag in [1, 2, 3, 6, 12, 24, 168]:
        if len(out) > lag:
            out[f"lag_{lag}"] = out["consumption"].shift(lag)

    for window in [3, 6, 12, 24, 168]:
        if len(out) >= window:
            out[f"rolling_mean_{window}"] = (
                out["consumption"].shift(1).rolling(window).mean()
            )

    return out

def chronological_split(df, test_size=0.2):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    split_index = int(len(df) * (1 - test_size))
    if split_index <= 0 or split_index >= len(df):
        raise ValueError("Invalid chronological train/test split.")
    return (
        df.iloc[:split_index].copy(),
        df.iloc[split_index:].copy(),
    )

def save_processed_data(df, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "processed_electricity_data.csv"
    df.to_csv(path, index=False)
    return path
