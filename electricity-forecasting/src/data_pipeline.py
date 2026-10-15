from pathlib import Path
import pandas as pd

TIME_COL = "datetime"
TARGET_COL = "consumption_kWh"


def load_and_clean_data(source):
    """Load a CSV path or file-like object and return clean hourly data."""
    df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]
    required = {TIME_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "CSV must contain columns 'datetime' and 'consumption_kWh'. "
            f"Missing: {', '.join(sorted(missing))}."
        )
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, TARGET_COL])
    df = df.sort_values(TIME_COL).drop_duplicates(subset=[TIME_COL], keep="last")
    df = df.set_index(TIME_COL)
    df = df[~df.index.duplicated(keep="last")]
    # Derive time features from datetime, so the app also works with simpler CSVs.
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["month"] = df.index.month
    df["day_of_year"] = df.index.dayofyear
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    # Keep optional numeric columns such as weather and holiday indicators.
    for col in df.columns:
        if col not in {TARGET_COL}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.interpolate(method="time", limit_direction="both").ffill().bfill()
    return df


def make_features(df):
    """Create leakage-safe features: lags and rolling means use past values."""
    data = df.copy()
    for lag in (1, 24, 168):
        data[f"lag_{lag}"] = data[TARGET_COL].shift(lag)
    data["rolling_mean_24"] = data[TARGET_COL].shift(1).rolling(24, min_periods=1).mean()
    data["rolling_mean_168"] = data[TARGET_COL].shift(1).rolling(168, min_periods=1).mean()
    data = data.dropna(subset=[f"lag_{lag}" for lag in (1, 24, 168)])
    return data


def save_processed_data(df, path="data/processed/cleaned_data.csv"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, index_label=TIME_COL)
    return path
