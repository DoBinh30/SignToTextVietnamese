"""Hand landmark extraction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import cv2
import mediapipe as mp
import numpy as np


_FEATURE_SIZE = 42  # 21 landmarks * (x, y)


@dataclass
class LandmarkSequenceBuilder:
    """Utility to maintain a fixed-length sequence of landmarks."""

    sequence_length: int

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self._buffer: List[np.ndarray] = []

    @property
    def feature_size(self) -> int:
        return _FEATURE_SIZE

    def append(self, landmarks: Optional[Iterable[float]]) -> None:
        """Append a new set of landmarks to the buffer.

        Args:
            landmarks: Iterable with size ``feature_size``. ``None`` results in zeros.
        """

        if landmarks is None:
            vector = np.zeros(self.feature_size, dtype=np.float32)
        else:
            vector = np.asarray(landmarks, dtype=np.float32).reshape(-1)
            if vector.size != self.feature_size:
                raise ValueError(
                    f"Expected {self.feature_size} values, received {vector.size}."
                )
        self._buffer.append(vector)
        if len(self._buffer) > self.sequence_length:
            self._buffer.pop(0)

    def reset(self) -> None:
        """Clear all buffered landmarks."""

        self._buffer.clear()

    def is_ready(self) -> bool:
        """Return ``True`` when the buffer is full."""

        return len(self._buffer) == self.sequence_length

    def as_flattened(self) -> np.ndarray:
        """Return the buffer flattened into ``sequence_length * feature_size``."""

        if not self.is_ready():
            raise ValueError("Sequence buffer is not full")
        return np.concatenate(self._buffer, axis=0)


@dataclass
class LandmarkResult:
    features: np.ndarray
    hand_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList


class HandLandmarkExtractor:
    """Extracts normalized hand landmarks using MediaPipe."""

    def __init__(
        self,
        *,
        static_image_mode: bool,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.3,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _normalize_landmarks(hand_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> List[float]:
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]
        min_x, min_y = min(x_coords), min(y_coords)
        data = []
        for lm in hand_landmarks.landmark:
            data.extend([lm.x - min_x, lm.y - min_y])
        return data

    def extract(self, frame: np.ndarray) -> Optional[LandmarkResult]:
        """Return normalized landmarks from a BGR frame.

        Args:
            frame: BGR image read by OpenCV.
        Returns:
            :class:`LandmarkResult` or ``None`` if detection fails.
        """

        if frame is None:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return None
        hand_landmarks = results.multi_hand_landmarks[0]
        features = np.asarray(self._normalize_landmarks(hand_landmarks), dtype=np.float32)
        return LandmarkResult(features=features, hand_landmarks=hand_landmarks)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
