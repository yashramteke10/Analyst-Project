def train_lstm(df, lookback=24, epochs=8, batch_size=32):
    """Optional TensorFlow LSTM. TensorFlow must be installed separately if unavailable."""
    try:
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as exc:
        raise ImportError("LSTM needs TensorFlow. Install it with: pip install tensorflow") from exc

    from src.data_pipeline import TARGET_COL
    values = df[[TARGET_COL]].astype(float).values
    split_at = int(len(values) * 0.8)
    scaler = MinMaxScaler()
    scaler.fit(values[:split_at])
    scaled = scaler.transform(values)
    X, y, positions = [], [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
        positions.append(i)
    X = np.asarray(X).reshape(-1, lookback, 1)
    y = np.asarray(y)
    positions = np.asarray(positions)
    train_mask = positions < split_at
    test_mask = positions >= split_at
    model = Sequential([LSTM(32, input_shape=(lookback, 1)), Dropout(0.2), Dense(16, activation="relu"), Dense(1)])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(X[train_mask], y[train_mask], validation_split=0.1, epochs=epochs,
              batch_size=batch_size, shuffle=False, verbose=0,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
    pred_scaled = model.predict(X[test_mask], verbose=0)
    predictions = scaler.inverse_transform(pred_scaled).ravel()
    actual = scaler.inverse_transform(y[test_mask].reshape(-1, 1)).ravel()
    return {"model": model, "predictions": predictions, "y_test": actual,
            "test_index": df.index[positions[test_mask]], "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()), "scaler": scaler}
