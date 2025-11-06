"""Train classifier for sign language recognition."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DEFAULT_DATASET = Path("data_sequence.pickle")
DEFAULT_MODEL_PATH = Path("model.p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to dataset pickle produced by create_dataset.py.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to store the trained model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of samples reserved for evaluation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=35,
        help="Number of training epochs for the CNN classifier.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Mini-batch size used for gradient descent training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate for the optimiser.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        dataset = pickle.load(f)
    data = np.asarray(dataset["data"], dtype=np.float32)
    labels = np.asarray(dataset["labels"], dtype=str)
    return data, labels


def build_model(sequence_length: int, num_classes: int, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, 42)),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(256, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
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


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    data, labels = load_dataset(args.dataset)

    feature_dim = data.shape[1]
    if feature_dim % 42 != 0:
        raise ValueError(
            "Dataset feature dimension must be divisible by 42 (landmark coordinates)"
        )
    sequence_length = feature_dim // 42
    data = data.reshape(-1, sequence_length, 42)

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        encoded_labels,
        test_size=args.test_size,
        shuffle=True,
        stratify=encoded_labels,
        random_state=42,
    )

    num_classes = len(encoder.classes_)
    model = build_model(sequence_length, num_classes, args.learning_rate)

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-5
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True
        ),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

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
