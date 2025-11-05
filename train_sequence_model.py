"""Training script for sequence-based sign language classifier using CNN + LSTM."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, optimizers
import cv2
from glob import glob
from sklearn.model_selection import train_test_split


# Set global seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


@dataclass
class DatasetConfig:
    """Configuration describing the synthetic dataset layout."""

    num_samples: int = 240
    sequence_length: int = 16
    frame_height: int = 128
    frame_width: int = 128
    channels: int = 3
    num_classes: int = 20

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        return (self.frame_height, self.frame_width, self.channels)

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        return (self.sequence_length, *self.frame_shape)


@dataclass
class TrainingConfig:
    """Hyper-parameter bundle for the training pipeline."""

    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float | None = None

def prepare_combined_dataset(config: DatasetConfig, base_dir: str = "data") -> Tuple[np.ndarray, ...]:
    """Load both image (static) and video (sequence) data for sign language training."""
    image_dir = os.path.join(base_dir, "images")
    video_dir = os.path.join(base_dir, "videos")

    sequences = []
    labels = []

    label_names = sorted(os.listdir(video_dir))  # label folders shared by both sources
    num_labels = len(label_names)
    print(f"Detected {num_labels} classes: {label_names}")

    for label_index, label_name in enumerate(label_names):
        # ---- Load videos ----
        video_files = glob(os.path.join(video_dir, label_name, "*.mp4"))
        for video_path in video_files:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(total_frames // config.sequence_length, 1)

            for i in range(0, total_frames, step):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (config.frame_width, config.frame_height))
                frame = frame.astype("float32") / 255.0
                frames.append(frame)
                if len(frames) == config.sequence_length:
                    break

            cap.release()
            if len(frames) == config.sequence_length:
                sequences.append(frames)
                labels.append(label_index)

        # ---- Load individual images (treated as static sequences) ----
        image_files = glob(os.path.join(image_dir, label_name, "*.jpg"))
        for image_path in image_files:
            img = cv2.imread(image_path)
            if img is None:
                continue
            img = cv2.resize(img, (config.frame_width, config.frame_height))
            img = img.astype("float32") / 255.0
            # replicate 1 image to match sequence length
            frames = [img] * config.sequence_length
            sequences.append(frames)
            labels.append(label_index)

    X = np.array(sequences, dtype=np.float32)
    y = tf.keras.utils.to_categorical(labels, num_classes=config.num_classes)

    # Split 80/10/10
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp.argmax(1))

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_sequence_model(config: DatasetConfig, training: TrainingConfig) -> models.Model:
    """Construct the CNN + LSTM sequence classifier."""

    frame_input = layers.Input(shape=config.input_shape, name="sequence_frames")

    base_cnn = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=config.frame_shape,
    )
    base_cnn.trainable = True

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])
    x = layers.TimeDistributed(data_augmentation)(frame_input)

    x = layers.TimeDistributed(base_cnn, name="frame_feature_extractor")(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D(), name="feature_pooling")(x)

    x = layers.LSTM(256, return_sequences=False, name="temporal_modeling")(x)
    x = layers.Dropout(0.3, name="temporal_dropout")(x)
    x = layers.Dense(128, activation="relu", name="projection_dense")(x)
    x = layers.Dropout(0.3, name="projection_dropout")(x)

    outputs = layers.Dense(config.num_classes, activation="softmax", name="class_predictions")(x)

    model = models.Model(inputs=frame_input, outputs=outputs, name="sign_sequence_classifier")

    optimizer = optimizers.Adam(learning_rate=training.learning_rate)
    if training.weight_decay:
        optimizer = optimizers.AdamW(learning_rate=training.learning_rate, weight_decay=training.weight_decay)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def plot_history(history: tf.keras.callbacks.History, output_path: str = "training_history.png") -> None:
    """Plot and persist the training/validation accuracy and loss curves."""

    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["loss"], label="Train Loss")
    plt.plot(epochs, history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cross-Entropy Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["accuracy"], label="Train Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Training history plot saved to {os.path.abspath(output_path)}")



def main() -> None:
    dataset_config = DatasetConfig()
    training_config = TrainingConfig()

    x_train, y_train, x_val, y_val, x_test, y_test = prepare_combined_dataset(dataset_config)

    model = build_sequence_model(dataset_config, training_config)
    model.summary()

    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=1e-6),
        callbacks.ModelCheckpoint(
            filepath="best_sequence_classifier.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=training_config.batch_size,
        epochs=training_config.epochs,
        callbacks=callbacks_list,
        verbose=2,
    )

    plot_history(history)

    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=training_config.batch_size, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
