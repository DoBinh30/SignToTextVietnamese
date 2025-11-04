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
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from time import monotonic

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

# Gesture handling parameters tuned to reduce accidental activations while keeping
# dynamic gestures responsive.
STATIC_HOLD_DURATION = 0.3  # seconds
DYNAMIC_GESTURE_LABELS = {"J", "Z"}
NO_OUTPUT_LABEL = "NO_OUTPUT"

# Prediction stabilisation parameters chosen to balance responsiveness and
# robustness when handling rapid gesture sequences from video input.
HISTORY_SIZE = 8
MIN_CONSENSUS = 4
MIN_CONFIDENCE = 0.6
POST_CONFIRM_COOLDOWN = 5


FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)
_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _load_font(size: int) -> ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    for path in FONT_CANDIDATES:
        font_path = Path(path)
        if font_path.exists():
            try:
                font = ImageFont.truetype(str(font_path), size)
            except OSError:
                continue
            _FONT_CACHE[size] = font
            return font

    fallback = ImageFont.load_default()
    _FONT_CACHE[size] = fallback
    return fallback


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


def _wrap_text(text: str, *, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    if not text:
        return []

    words = text.split()
    if not words:
        return [text]

    lines: List[str] = []
    current_line: List[str] = []

    for word in words:
        test_line = " ".join(current_line + [word]).strip()
        if font.getlength(test_line) <= max_width:
            current_line.append(word)
            continue

        if current_line:
            lines.append(" ".join(current_line))
        current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def build_text_canvas(
    recognized_text: str,
    suggestions: Sequence[str],
    *,
    width: int = 720,
    height: int = 480,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(image)

    border_rect = (20, 20, width - 20, height - 20)
    draw.rounded_rectangle(border_rect, radius=20, fill=(255, 255, 255), outline=(200, 200, 200), width=3)

    title_font = _load_font(40)
    body_font = _load_font(28)
    suggestion_font = _load_font(30)

    draw.text((40, 50), "Văn bản", fill=(30, 30, 30), font=title_font)

    text_area_width = width - 80
    wrapped_lines = _wrap_text(recognized_text, font=body_font, max_width=text_area_width)
    y_offset = 120
    max_lines = 8
    line_spacing = int(body_font.size * 1.3)
    for line in wrapped_lines[:max_lines]:
        draw.text((40, y_offset), line, fill=(50, 50, 50), font=body_font)
        y_offset += line_spacing

    draw.text((40, height - 170), "Gợi ý", fill=(60, 60, 60), font=title_font)

    suggestion_spacing = int(suggestion_font.size * 1.4)
    for idx, suggestion in enumerate(suggestions[:3], start=1):
        draw.text(
            (40, height - 170 + idx * suggestion_spacing),
            f"{idx}. {suggestion}",
            fill=(80, 80, 80),
            font=suggestion_font,
        )

    canvas = np.array(image)[:, :, ::-1]
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
    active_label: Optional[str] = None
    active_label_since: Optional[float] = None

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

                if (
                    predicted_character in DYNAMIC_GESTURE_LABELS
                    and confidence >= MIN_CONFIDENCE
                ):
                    stable_label = predicted_character
                    stabiliser.reset()

                if cooldown_frames > 0:
                    cooldown_frames -= 1
                    if stable_label is not None:
                        stabiliser.reset()
                        active_label = None
                        active_label_since = None
                    stable_label = None

                if stable_label is not None:
                    now = monotonic()
                    if active_label != stable_label:
                        active_label = stable_label
                        active_label_since = now

                    hold_required = stable_label not in DYNAMIC_GESTURE_LABELS
                    hold_satisfied = (
                        not hold_required
                        or (
                            active_label_since is not None
                            and (now - active_label_since) >= STATIC_HOLD_DURATION
                        )
                    )

                    if hold_satisfied:
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
                        elif stable_label == NO_OUTPUT_LABEL:
                            pass
                        else:
                            confirmed_text.append(normalized)

                        cooldown_frames = POST_CONFIRM_COOLDOWN
                        stabiliser.reset()
                        active_label = None
                        active_label_since = None
                else:
                    active_label = None
                    active_label_since = None

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
