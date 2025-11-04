"""Dataset building utilities for static images and videos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .landmark_extractor import HandLandmarkExtractor


@dataclass
class DatasetSample:
    features: np.ndarray
    label: str


def _iter_media_files(base_dir: Path) -> Iterable[Tuple[str, Path]]:
    for label_dir in sorted(base_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.iterdir()):
            if path.is_file():
                yield label_dir.name, path


def _build_sequence(
    feature_list: List[np.ndarray],
    sequence_length: int,
    feature_dim: int,
) -> Optional[np.ndarray]:
    if not feature_list:
        return None
    features = np.stack(feature_list)
    if features.shape[0] >= sequence_length:
        indices = np.linspace(0, features.shape[0] - 1, sequence_length).astype(int)
        selected = features[indices]
    else:
        pad_count = sequence_length - features.shape[0]
        pad = np.repeat(features[-1:], pad_count, axis=0)
        selected = np.concatenate([features, pad], axis=0)
    return selected.reshape(sequence_length * feature_dim)


def _extract_from_image(
    extractor: HandLandmarkExtractor,
    image_path: Path,
    sequence_length: int,
    feature_dim: int,
) -> Optional[np.ndarray]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    result = extractor.extract(image)
    if result is None:
        return None
    vector = result.features
    tiled = np.tile(vector, sequence_length)
    return tiled.reshape(sequence_length * feature_dim)


def _extract_from_video(
    extractor: HandLandmarkExtractor,
    video_path: Path,
    sequence_length: int,
    feature_dim: int,
) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = extractor.extract(frame)
        if result is None:
            continue
        frames.append(np.asarray(result.features, dtype=np.float32))
    cap.release()
    return _build_sequence(frames, sequence_length, feature_dim)


def build_dataset(
    *,
    image_dir: Optional[Path],
    video_dir: Optional[Path],
    sequence_length: int,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.3,
    min_tracking_confidence: float = 0.5,
) -> Dict[str, Sequence[np.ndarray]]:
    """Build dataset features and labels from image and video directories."""

    samples: List[DatasetSample] = []

    feature_dim = 42
    if image_dir and image_dir.exists():
        with HandLandmarkExtractor(
            static_image_mode=True,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as extractor:
            for label, path in _iter_media_files(image_dir):
                features = _extract_from_image(extractor, path, sequence_length, feature_dim)
                if features is None:
                    continue
                samples.append(DatasetSample(features=features, label=label))

    if video_dir and video_dir.exists():
        with HandLandmarkExtractor(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ) as extractor:
            for label, path in _iter_media_files(video_dir):
                features = _extract_from_video(extractor, path, sequence_length, feature_dim)
                if features is None:
                    continue
                samples.append(DatasetSample(features=features, label=label))

    if not samples:
        raise ValueError("No samples were collected from the provided directories")

    data = np.stack([sample.features for sample in samples])
    labels = np.array([sample.label for sample in samples])
    return {"data": data, "labels": labels}
