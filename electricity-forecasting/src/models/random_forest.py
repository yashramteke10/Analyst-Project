from sklearn.ensemble import RandomForestRegressor
from src.data_pipeline import TARGET_COL, make_features


def train_random_forest(df, n_estimators=150, max_depth=None, min_samples_leaf=2):
    featured = make_features(df)
    feature_cols = [c for c in featured.columns if c != TARGET_COL]
    split = int(len(featured) * 0.8)
    if split < 1 or len(featured) - split < 1:
        raise ValueError("Not enough rows to create a chronological train/test split.")
    X_train, X_test = featured[feature_cols].iloc[:split], featured[feature_cols].iloc[split:]
    y_train, y_test = featured[TARGET_COL].iloc[:split], featured[TARGET_COL].iloc[split:]
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, random_state=42,
                                  n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"model": model, "feature_cols": feature_cols, "X_test": X_test,
            "y_test": y_test, "predictions": pred, "test_index": X_test.index,
            "train_rows": len(X_train), "test_rows": len(X_test)}
