"""Realtime sign language inference supporting dynamic gestures.

The script renders a refined UI suitable for product usage:

* A dedicated text canvas shows the confirmed transcription and Vietnamese
  language suggestions.
* Frame-level predictions are stabilised to avoid false positives while keeping
  gesture recognition responsive.
* Special gesture labels enable users to delete characters, clear the entire
  sentence, or accept the top suggestion hands-free.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import pickle
import textwrap
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from sign_language import HandLandmarkExtractor, LandmarkSequenceBuilder
from sign_language.vietnamese_suggester import VietnameseWordSuggester


DEFAULT_MODEL_PATH = Path("model.p")
WINDOW_NAME = "Sign Language Recognition"
TEXT_WINDOW_NAME = "Văn bản nhận diện"

# Gesture labels reserved for control actions. Update these constants to match
# the labels in your trained classifier before deployment.
DELETE_CHARACTER_LABEL = "DELETE_CHARACTER"
CLEAR_TEXT_LABEL = "CLEAR_TEXT"
ACCEPT_SUGGESTION_LABEL = "ACCEPT_SUGGESTION"

# Character level overrides for convenient mapping (e.g. a gesture producing a
# literal "SPACE" inserts an actual whitespace character).
CHARACTER_OVERRIDES = {"SPACE": " "}

# Prediction stabilisation parameters chosen to balance responsiveness and
# robustness when handling rapid gesture sequences from video input.
HISTORY_SIZE = 8
MIN_CONSENSUS = 4
MIN_CONFIDENCE = 0.6
POST_CONFIRM_COOLDOWN = 5


class PredictionStabiliser:
    """Aggregate frame-level predictions to suppress outliers."""

    def __init__(
        self,
        *,
        history_size: int = HISTORY_SIZE,
        min_consensus: int = MIN_CONSENSUS,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        self._history: Deque[Tuple[str, float]] = deque(maxlen=history_size)
        self._min_consensus = min_consensus
        self._min_confidence = min_confidence

    def reset(self) -> None:
        self._history.clear()

    def update(self, label: Optional[str], confidence: float) -> Optional[str]:
        if label is None:
            self.reset()
            return None

        self._history.append((label, confidence))
        if len(self._history) < self._min_consensus:
            return None

        counter = Counter(
            lbl for lbl, conf in self._history if conf >= self._min_confidence
        )
        if not counter:
            return None

        candidate, count = counter.most_common(1)[0]
        if count < self._min_consensus:
            return None
        return candidate


def build_text_canvas(
    recognized_text: str,
    suggestions: Sequence[str],
    *,
    width: int = 600,
    height: int = 400,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.rectangle(canvas, (10, 10), (width - 10, height - 10), (200, 200, 200), 2)

    cv2.putText(
        canvas,
        "Văn bản",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )

    wrapped_lines = textwrap.wrap(recognized_text, width=32)
    y_offset = 90
    for line in wrapped_lines[:8]:
        cv2.putText(
            canvas,
            line,
            (30, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        y_offset += 40

    cv2.putText(
        canvas,
        "Gợi ý",
        (30, height - 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (70, 70, 70),
        2,
        cv2.LINE_AA,
    )

    for idx, suggestion in enumerate(suggestions[:3], start=1):
        cv2.putText(
            canvas,
            f"{idx}. {suggestion}",
            (30, height - 120 + idx * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (50, 50, 50),
            2,
            cv2.LINE_AA,
        )

    return canvas


def normalise_label(label: str) -> str:
    return CHARACTER_OVERRIDES.get(label, label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model file.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to use.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum detection confidence for MediaPipe Hands.",
    )
    return parser.parse_args()


def load_model(path: Path):
    with path.open("rb") as f:
        model_dict = pickle.load(f)
    if "model" not in model_dict or "label_encoder" not in model_dict:
        raise ValueError("Model file does not contain required keys")
    sequence_length = model_dict.get("sequence_length")
    if sequence_length is None:
        raise ValueError("Model file missing sequence_length metadata")
    return model_dict["model"], model_dict["label_encoder"], sequence_length


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model, label_encoder, sequence_length = load_model(args.model_path)
    sequence_builder = LandmarkSequenceBuilder(sequence_length=sequence_length)
    stabiliser = PredictionStabiliser()
    suggester = VietnameseWordSuggester.from_default()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    predicted_character: Optional[str] = None
    confirmed_text: List[str] = []
    cooldown_frames = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(TEXT_WINDOW_NAME, cv2.WINDOW_NORMAL)

    with HandLandmarkExtractor(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.confidence,
        min_tracking_confidence=0.5,
    ) as extractor:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            frame = cv2.flip(frame, 1)
            result = extractor.extract(frame)
            sequence_builder.append(result.features if result else None)

            if result and result.hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    result.hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            if sequence_builder.is_ready():
                features = sequence_builder.as_flattened().reshape(1, -1)
                prediction = model.predict(features)
                predicted_character = label_encoder.inverse_transform(prediction)[0]

                confidence = 1.0
                if hasattr(model, "predict_proba"):
                    try:
                        probabilities = model.predict_proba(features)
                        confidence = float(np.max(probabilities))
                    except Exception:  # pragma: no cover - fallback to default
                        confidence = 1.0

                stable_label = stabiliser.update(predicted_character, confidence)

                if cooldown_frames > 0:
                    cooldown_frames -= 1
                    if stable_label is not None:
                        stabiliser.reset()
                    stable_label = None

                if stable_label is not None:
                    normalized = normalise_label(stable_label)

                    if stable_label == DELETE_CHARACTER_LABEL:
                        if confirmed_text:
                            confirmed_text.pop()
                    elif stable_label == CLEAR_TEXT_LABEL:
                        confirmed_text.clear()
                    elif stable_label == ACCEPT_SUGGESTION_LABEL:
                        composed = "".join(confirmed_text)
                        suggestion = suggester.best_suggestion(composed)
                        if suggestion:
                            confirmed_text = list(
                                suggester.apply_suggestion(composed, suggestion)
                            )
                    else:
                        confirmed_text.append(normalized)

                    cooldown_frames = POST_CONFIRM_COOLDOWN
                    stabiliser.reset()

            composed_text = "".join(confirmed_text)
            suggestions = suggester.suggest(composed_text)

            if predicted_character:
                cv2.putText(
                    frame,
                    predicted_character,
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 0),
                    6,
                )

            cv2.imshow(WINDOW_NAME, frame)
            text_canvas = build_text_canvas(composed_text, suggestions)
            cv2.imshow(TEXT_WINDOW_NAME, text_canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
