from src.data_pipeline import load_and_clean_data, make_features


def test_load_and_clean_data():
    import pandas as pd
    df = pd.DataFrame({"datetime": ["2024-01-01 01:00", "2024-01-01 00:00"],
                       "consumption_kWh": [2.0, 1.0]})
    clean = load_and_clean_data(df.to_csv(index=False).encode())
    assert clean.index.is_monotonic_increasing
    assert len(clean) == 2


def test_make_features():
    import pandas as pd
    from src.data_pipeline import TARGET_COL
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    df = pd.DataFrame({TARGET_COL: range(200)}, index=idx)
    featured = make_features(df)
    assert "lag_24" in featured.columns
    assert len(featured) > 0
