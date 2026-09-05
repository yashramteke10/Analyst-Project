from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(df, lookback=24, test_size=0.2):
    values = df["consumption"].to_numpy(dtype=float).reshape(-1, 1)
    times = df["datetime"].reset_index(drop=True)

    split_point = int(len(values) * (1 - test_size))
    if split_point <= lookback:
        raise ValueError("Training portion is smaller than the LSTM lookback.")
    if len(values) - split_point < 1:
        raise ValueError("Test set contains no records.")

    scaler = MinMaxScaler()
    scaler.fit(values[:split_point])
    scaled = scaler.transform(values).astype("float32")

    X, y, timestamps = [], [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, 0])
        timestamps.append(times.iloc[i])

    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="float32")
    timestamps = np.asarray(timestamps)

    target_indices = np.arange(lookback, len(scaled))
    train_mask = target_indices < split_point
    test_mask = ~train_mask

    return (
        X[train_mask],
        y[train_mask],
        X[test_mask],
        y[test_mask],
        scaler,
        timestamps[test_mask],
    )

def build_lstm(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model

def train_lstm(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    save_path="models_store/lstm_model.h5",
):
    tf.keras.utils.set_random_seed(42)
    model = build_lstm((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        callbacks=callbacks,
        verbose=0,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    return model

def predict_lstm(model, X_test, scaler):
    scaled_prediction = model.predict(X_test, verbose=0).ravel()
    return scaler.inverse_transform(
        scaled_prediction.reshape(-1, 1)
    ).ravel()

def forecast_next(model, last_values, scaler, lookback):
    values = np.asarray(last_values, dtype=float).reshape(-1, 1)
    if len(values) < lookback:
        raise ValueError("Not enough historical values for next-step forecasting.")

    scaled = scaler.transform(values)
    X = scaled[-lookback:].reshape(
        1, lookback, 1
    ).astype("float32")
    prediction_scaled = model.predict(X, verbose=0)[0, 0]
    return float(scaler.inverse_transform([[prediction_scaled]])[0, 0])

def load_lstm(path="models_store/lstm_model.h5"):
    return tf.keras.models.load_model(path)
