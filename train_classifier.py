"""Train classifier for sign language recognition with landmark augmentation."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DEFAULT_DATASET = Path("data_sequence.pickle")
DEFAULT_MODEL_PATH = Path("model.p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        dataset = pickle.load(f)
    data = np.asarray(dataset["data"], dtype=np.float32)
    labels = np.asarray(dataset["labels"], dtype=str)
    return data, labels


# 💡 Augmentation toán học cho landmarks Mediapipe
def augment_landmarks(sequence: np.ndarray) -> np.ndarray:
    """Thêm nhiễu ngẫu nhiên, scale và xoay nhẹ cho tọa độ landmarks."""
    # sequence shape: (frames, 42)
    coords = sequence.reshape(-1, 21, 2)

    # 1️⃣ Noise ngẫu nhiên
    noise = np.random.normal(0, 0.01, coords.shape)
    coords += noise

    # 2️⃣ Scale ngẫu nhiên (zoom in/out 0.9x–1.1x)
    scale = np.random.uniform(0.9, 1.1)
    coords *= scale

    # 3️⃣ Xoay nhẹ quanh gốc (ngẫu nhiên ±10 độ)
    angle = np.random.uniform(-0.1, 0.1)
    rotation_matrix = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    coords = np.matmul(coords, rotation_matrix)

    return coords.reshape(-1, 42).astype(np.float32)


def build_model(sequence_length: int, num_classes: int, learning_rate: float) -> tf.keras.Model:
    """Build a hybrid CNN-LSTM classifier for spatio-temporal landmarks."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, 42)),
            # Convolutional feature extractor (per-frame spatial cues)
            tf.keras.layers.Conv1D(128, kernel_size=3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Conv1D(256, kernel_size=3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Dropout(0.25),
            # Temporal modelling with stacked LSTM layers
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(128),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def _save_training_curves(history: tf.keras.callbacks.History) -> None:
    acc_path = f"accuracy_curve.png"
    loss_path = f"loss_curve.png"

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close()

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close()

    print(f"Saved training curves: {acc_path}, {loss_path}")


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    data, labels = load_dataset(args.dataset)

    feature_dim = data.shape[1]
    if feature_dim % 42 != 0:
        raise ValueError("Dataset feature dimension must be divisible by 42 (landmark coordinates)")
    sequence_length = feature_dim // 42
    data = data.reshape(-1, sequence_length, 42)

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        data, encoded_labels, test_size=args.test_size, shuffle=True, stratify=encoded_labels, random_state=42
    )

    # 🧩 Áp dụng augmentation toán học cho landmarks (tăng dữ liệu)
    augmented = np.array([augment_landmarks(x) for x in x_train])
    x_train = np.concatenate([x_train, augmented], axis=0)
    y_train = np.concatenate([y_train, y_train], axis=0)
    print(f"🧠 Augmented training data: {len(x_train)} samples total")

    num_classes = len(encoder.classes_)
    model = build_model(sequence_length, num_classes, args.learning_rate)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-5),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    _save_training_curves(history)

    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    score = accuracy_score(y_test, predictions)
    print(f"{score * 100:.2f}% of samples were classified correctly!")

    to_save = {
        "model_config": model.to_json(),
        "model_weights": model.get_weights(),
        "label_encoder": encoder,
        "sequence_length": int(sequence_length),
    }
    with args.model_path.open("wb") as f:
        pickle.dump(to_save, f)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
