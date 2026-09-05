"""
Optional post-MVP Transformer model.

This module is intentionally separate from the MVP. It can be enabled after
the Random Forest + LSTM pipeline is validated.
"""

from pathlib import Path
import tensorflow as tf

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        attention = self.attention(inputs, inputs)
        attention = self.dropout1(attention, training=training)
        x = self.norm1(inputs + attention)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(x + ffn_output)

def build_transformer(input_shape, embed_dim=32, num_heads=2, ff_dim=64):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )
    return model

def train_transformer(
    X_train,
    y_train,
    epochs=10,
    save_path="models_store/transformer_model.h5",
):
    model = build_transformer(
        (X_train.shape[1], X_train.shape[2])
    )
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        validation_split=0.1,
        shuffle=False,
        verbose=0,
    )
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return model
