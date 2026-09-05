import pandas as pd
from src.data_pipeline import process_dataframe, create_features, chronological_split

def test_process_dataframe():
    df = pd.DataFrame({
        "datetime": ["2025-01-01 01:00", "2025-01-01 00:00", "2025-01-01 00:00"],
        "consumption_kwh": [5.0, 4.0, 4.0],
    })
    cleaned, report = process_dataframe(df)
    assert list(cleaned.columns) == ["datetime", "consumption"]
    assert len(cleaned) == 2
    assert cleaned["datetime"].is_monotonic_increasing
    assert report["duplicates_removed"] == 1

def test_feature_engineering():
    dates = pd.date_range("2025-01-01", periods=200, freq="h")
    df = pd.DataFrame({"datetime": dates, "consumption": range(200)})
    features = create_features(df)
    assert "hour" in features.columns
    assert "lag_24" in features.columns
    assert "rolling_mean_24" in features.columns

def test_chronological_split():
    dates = pd.date_range("2025-01-01", periods=100, freq="h")
    df = pd.DataFrame({"datetime": dates, "consumption": range(100)})
    train, test = chronological_split(df, 0.2)
    assert len(train) == 80
    assert train["datetime"].max() < test["datetime"].min()
