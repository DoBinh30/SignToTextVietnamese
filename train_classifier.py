"""Train classifier for sign language recognition."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    return parser.parse_args()


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        dataset = pickle.load(f)
    data = np.asarray(dataset["data"], dtype=np.float32)
    labels = np.asarray(dataset["labels"], dtype=str)
    return data, labels


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    data, labels = load_dataset(args.dataset)
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

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    print(f"{score * 100:.2f}% of samples were classified correctly!")

    feature_dim = data.shape[1]
    if feature_dim % 42 != 0:
        raise ValueError(
            "Dataset feature dimension must be divisible by 42 (landmark coordinates)"
        )
    sequence_length = feature_dim // 42
    to_save = {
        "model": model,
        "label_encoder": encoder,
        "sequence_length": int(sequence_length),
    }
    with args.model_path.open("wb") as f:
        pickle.dump(to_save, f)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()
