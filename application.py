"""Realtime sign language inference with a Tkinter transcription display.

The application offers a modernised user experience:

* A Tkinter window shows the recognised text and Vietnamese word suggestions
  using large, high-quality fonts that render Unicode characters accurately.
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
from time import monotonic
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import font as tkfont

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
STATIC_HOLD_DURATION = 0.2  # seconds
DYNAMIC_GESTURE_LABELS = {"j", "z"}
NO_OUTPUT_LABEL = "NO_OUTPUT"

# Prediction stabilisation parameters chosen to balance responsiveness and
# robustness when handling rapid gesture sequences from video input.
HISTORY_SIZE = 8
MIN_CONSENSUS = 4
MIN_CONFIDENCE = 0.6
POST_CONFIRM_COOLDOWN = 5


class TkinterTextDisplay:
    """Render recognised text and suggestions via Tkinter widgets."""

    def __init__(self, *, width: int = 720, height: int = 480) -> None:
        self._closed = False
        self._quit_requested = False
        self._root = tk.Tk()
        self._root.title(TEXT_WINDOW_NAME)
        self._root.configure(bg="#f2f2f2")
        self._root.geometry(f"{width}x{height}")
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root.bind("<KeyRelease-q>", self._on_quit_key)
        self._root.bind("<KeyRelease-Q>", self._on_quit_key)

        self._heading_font = self._create_font(size=28, weight="bold")
        self._body_font = self._create_font(size=20)
        self._suggestion_font = self._create_font(size=22)

        container = tk.Frame(self._root, bg="#ffffff", bd=0, highlightthickness=0)
        container.pack(fill="both", expand=True, padx=24, pady=24)

        text_section = tk.Frame(container, bg="#ffffff")
        text_section.pack(fill="both", expand=True)

        tk.Label(
            text_section,
            text="Văn bản",
            font=self._heading_font,
            bg="#ffffff",
            fg="#202020",
            anchor="w",
        ).pack(fill="x")

        self._text_var = tk.StringVar()
        self._text_label = tk.Label(
            text_section,
            textvariable=self._text_var,
            font=self._body_font,
            bg="#ffffff",
            fg="#303030",
            wraplength=width - 96,
            justify="left",
        )
        self._text_label.pack(fill="both", expand=True, pady=(12, 24))

        suggestion_section = tk.Frame(container, bg="#ffffff")
        suggestion_section.pack(fill="x", pady=(0, 12))

        tk.Label(
            suggestion_section,
            text="Gợi ý",
            font=self._heading_font,
            bg="#ffffff",
            fg="#202020",
            anchor="w",
        ).pack(fill="x")

        self._suggestion_vars: List[tk.StringVar] = [tk.StringVar() for _ in range(3)]
        for var in self._suggestion_vars:
            tk.Label(
                suggestion_section,
                textvariable=var,
                font=self._suggestion_font,
                bg="#ffffff",
                fg="#404040",
                anchor="w",
            ).pack(fill="x", pady=(6, 0))

    def _create_font(self, *, size: int, weight: str = "normal") -> tkfont.Font:
        preferred_families = [
            "Noto Sans",
            "DejaVu Sans",
            "Arial",
            "Helvetica",
        ]
        for family in preferred_families:
            try:
                return tkfont.Font(family=family, size=size, weight=weight)
            except tk.TclError:
                continue
        return tkfont.Font(size=size, weight=weight)

    def update(self, text: str, suggestions: Sequence[str]) -> None:
        self._text_var.set(text or "")
        for idx, var in enumerate(self._suggestion_vars):
            if idx < len(suggestions):
                var.set(f"{idx + 1}. {suggestions[idx]}")
            else:
                var.set("")

    def pump_events(self) -> bool:
        if self._closed or self._quit_requested:
            return False
        try:
            self._root.update_idletasks()
            self._root.update()
        except tk.TclError:
            self._closed = True
            return False
        return True

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._root.destroy()

    def _on_close(self) -> None:
        self._closed = True
        self._root.destroy()

    def _on_quit_key(self, _event: object) -> None:
        self._quit_requested = True
        self._on_close()


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
    last_written_label: Optional[str] = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    text_display = TkinterTextDisplay()

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
                    # and confidence >= MIN_CONFIDENCE
                ):
                    stable_label = predicted_character
                    stabiliser.reset()

                cooldown_blocked = False
                if cooldown_frames > 0:
                    cooldown_frames -= 1
                    if stable_label is not None:
                        if stable_label in DYNAMIC_GESTURE_LABELS:
                            cooldown_frames = 0
                        else:
                            stabiliser.reset()
                            active_label = None
                            active_label_since = None
                            cooldown_blocked = True
                    if cooldown_blocked:
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
                        should_commit = True
                        if stable_label not in {
                            DELETE_CHARACTER_LABEL,
                            CLEAR_TEXT_LABEL,
                            ACCEPT_SUGGESTION_LABEL,
                        } and stable_label == last_written_label:
                            should_commit = False

                        if should_commit:
                            normalized = normalise_label(stable_label)

                            if stable_label == DELETE_CHARACTER_LABEL:
                                if confirmed_text:
                                    confirmed_text.pop()
                                last_written_label = None
                            elif stable_label == CLEAR_TEXT_LABEL:
                                confirmed_text.clear()
                                last_written_label = None
                            elif stable_label == ACCEPT_SUGGESTION_LABEL:
                                composed = "".join(confirmed_text)
                                suggestion = suggester.best_suggestion(composed)
                                if suggestion:
                                    confirmed_text = list(
                                        suggester.apply_suggestion(composed, suggestion)
                                    )
                                last_written_label = None
                            elif stable_label == NO_OUTPUT_LABEL:
                                last_written_label = NO_OUTPUT_LABEL
                            else:
                                confirmed_text.append(normalized)
                                last_written_label = stable_label

                            if stable_label not in DYNAMIC_GESTURE_LABELS:
                                cooldown_frames = POST_CONFIRM_COOLDOWN
                            else:
                                cooldown_frames = 0
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
            text_display.update(composed_text, suggestions)
            if not text_display.pump_events() or cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    text_display.close()


if __name__ == "__main__":
    main()