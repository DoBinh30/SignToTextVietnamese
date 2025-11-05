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
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
from time import monotonic
from typing import Any, Deque, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import font as tkfont

from sign_language import HandLandmarkExtractor, LandmarkSequenceBuilder
from sign_language.vietnamese_suggester import VietnameseWordSuggester


DEFAULT_MODEL_PATH = Path("best_sequence_classifier.keras")
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
STATIC_HOLD_DURATION = 0.1  # seconds
DYNAMIC_GESTURE_LABELS = {"j", "z"}
NO_OUTPUT_LABEL = "NO_OUTPUT"

# Prediction stabilisation parameters chosen to balance responsiveness and
# robustness when handling rapid gesture sequences from video input.
HISTORY_SIZE = 8
MIN_CONSENSUS = 4
MIN_CONFIDENCE = 0.6
POST_CONFIRM_COOLDOWN = 3


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
        "--labels-path",
        type=Path,
        default=None,
        help=(
            "Optional path to class label metadata for sequence models. "
            "If omitted the application will look for a sibling file next to the model."
        ),
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


@dataclass
class LegacyModelBundle:
    model: Any
    label_encoder: Any
    sequence_length: int
    kind: str = field(init=False, default="landmark")


@dataclass
class SequenceModelBundle:
    model: Any
    class_names: Sequence[str]
    sequence_length: int
    frame_shape: Tuple[int, int, int]
    kind: str = field(init=False, default="sequence")


def _load_label_metadata(
    model_path: Path, labels_path: Optional[Path], num_classes: int
) -> List[str]:
    """Load ordered class names for sequence models."""

    candidates: List[Path] = []
    if labels_path is not None:
        candidates.append(labels_path)
    suffix = model_path.suffix
    if suffix:
        candidates.extend(
            [
                model_path.with_suffix(".labels.json"),
                model_path.with_suffix(".labels.txt"),
                model_path.with_suffix(".labels.npy"),
                model_path.with_suffix(".labels.pkl"),
                model_path.with_suffix(".labels.pickle"),
            ]
        )

    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".json":
            labels = json.loads(candidate.read_text(encoding="utf-8"))
        elif candidate.suffix == ".txt":
            labels = [line.strip() for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif candidate.suffix == ".npy":
            import numpy as np  # Local import to avoid hard dependency.

            labels = np.load(candidate, allow_pickle=True).tolist()
        elif candidate.suffix in {".pkl", ".pickle"}:
            with candidate.open("rb") as f:
                metadata = pickle.load(f)
            if hasattr(metadata, "classes_"):
                labels = list(metadata.classes_)
            elif isinstance(metadata, (list, tuple)):
                labels = list(metadata)
            else:
                raise ValueError(
                    f"Unsupported label metadata format in {candidate}."
                )
        else:
            continue

        if len(labels) != num_classes:
            raise ValueError(
                "Label metadata size does not match the model output dimension"
            )
        return [str(label) for label in labels]

    raise FileNotFoundError(
        "Unable to locate label metadata for the sequence model. "
        "Provide the path via --labels-path or place a labels file next to the model."
    )


def load_model(path: Path, labels_path: Optional[Path] = None):
    if path.suffix == ".p":
        with path.open("rb") as f:
            model_dict = pickle.load(f)
        if "model" not in model_dict or "label_encoder" not in model_dict:
            raise ValueError("Model file does not contain required keys")
        sequence_length = model_dict.get("sequence_length")
        if sequence_length is None:
            raise ValueError("Model file missing sequence_length metadata")
        return LegacyModelBundle(
            model=model_dict["model"],
            label_encoder=model_dict["label_encoder"],
            sequence_length=sequence_length,
        )

    if path.suffix in {".keras", ".h5"}:
        from tensorflow import keras  # Imported lazily to avoid startup cost.

        model = keras.models.load_model(path)
        input_shape = model.input_shape
        if not isinstance(input_shape, tuple) or len(input_shape) != 5:
            raise ValueError(
                "Expected sequence model input shape (batch, time, height, width, channels)"
            )
        _, sequence_length, height, width, channels = input_shape
        if None in (sequence_length, height, width, channels):
            raise ValueError("Model input shape must be fully defined")
        output_shape = model.output_shape
        if not isinstance(output_shape, tuple):
            raise ValueError("Unexpected model output shape")
        num_classes = output_shape[-1]
        if num_classes is None:
            raise ValueError("Model output dimension must be defined")
        class_names = _load_label_metadata(path, labels_path, num_classes)
        frame_shape = (int(height), int(width), int(channels))
        return SequenceModelBundle(
            model=model,
            class_names=class_names,
            sequence_length=int(sequence_length),
            frame_shape=frame_shape,
        )

    raise ValueError(f"Unsupported model format: {path.suffix}")


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model_bundle = load_model(args.model_path, args.labels_path)
    sequence_builder: Optional[LandmarkSequenceBuilder] = None
    frame_buffer: Optional[Deque[np.ndarray]] = None
    if isinstance(model_bundle, LegacyModelBundle):
        sequence_builder = LandmarkSequenceBuilder(
            sequence_length=model_bundle.sequence_length
        )
    elif isinstance(model_bundle, SequenceModelBundle):
        frame_buffer = deque(maxlen=model_bundle.sequence_length)
    else:
        raise RuntimeError("Unsupported model bundle returned by load_model")
    stabiliser = PredictionStabiliser(
        history_size=HISTORY_SIZE, min_consensus=MIN_CONSENSUS
    )
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
    def reset_stabiliser() -> None:
        stabiliser.reset()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    text_display = TkinterTextDisplay()

    with HandLandmarkExtractor(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.confidence,
        min_tracking_confidence=0.5,
    ) as extractor:
        hand_detected_recently = False
        frames_since_hand = 0
        hand_visible_frames = 0  # Đếm số frame tay đã ở trong khung

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            frame = cv2.flip(frame, 1)
            result = extractor.extract(frame)

            if isinstance(model_bundle, LegacyModelBundle):
                assert sequence_builder is not None  # for type checkers
                sequence_builder.append(result.features if result else None)
            elif isinstance(model_bundle, SequenceModelBundle):
                assert frame_buffer is not None
                if result and result.hand_landmarks:
                    height, width, channels = model_bundle.frame_shape
                    resized = cv2.resize(frame, (width, height))
                    if channels == 1:
                        processed = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)[
                            ..., np.newaxis
                        ]
                    elif channels == 3 and resized.ndim == 2:
                        processed = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                    else:
                        processed = resized
                    processed = processed.astype("float32") / 255.0
                    frame_buffer.append(processed)
                elif frame_buffer:
                    frame_buffer.clear()

            # Kiểm tra có bàn tay trong khung
            if result and result.hand_landmarks:
                frames_since_hand = 0
                hand_visible_frames += 1
                hand_detected_recently = True
                mp_drawing.draw_landmarks(
                    frame,
                    result.hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
            else:
                frames_since_hand += 1
                if frames_since_hand > 15:  # ~0.5 giây không thấy tay
                    hand_detected_recently = False
                    hand_visible_frames = 0  # reset khi mất tay
                    if frame_buffer is not None:
                        frame_buffer.clear()

            predicted_character = None
            confidence = 0.0
            stable_label: Optional[str] = None

            # Chỉ dự đoán khi có tay và tay đã ổn định ít nhất 10 frame (~0.3s)
            if hand_detected_recently and hand_visible_frames > 10:
                if isinstance(model_bundle, LegacyModelBundle):
                    assert sequence_builder is not None
                    if sequence_builder.is_ready():
                        features = sequence_builder.as_flattened().reshape(1, -1)
                        prediction = model_bundle.model.predict(features)
                        predicted_character = model_bundle.label_encoder.inverse_transform(
                            prediction
                        )[0]

                        confidence = 1.0
                        if hasattr(model_bundle.model, "predict_proba"):
                            try:
                                probabilities = model_bundle.model.predict_proba(features)
                                confidence = float(np.max(probabilities))
                            except Exception:
                                confidence = 1.0

                elif isinstance(model_bundle, SequenceModelBundle):
                    assert frame_buffer is not None
                    if len(frame_buffer) == frame_buffer.maxlen:
                        sequence_array = np.asarray(frame_buffer, dtype=np.float32)
                        sequence_array = sequence_array.reshape(
                            1,
                            model_bundle.sequence_length,
                            *model_bundle.frame_shape,
                        )
                        probabilities = model_bundle.model.predict(
                            sequence_array, verbose=0
                        )[0]
                        index = int(np.argmax(probabilities))
                        confidence = float(probabilities[index])
                        predicted_character = str(model_bundle.class_names[index])

                if predicted_character is not None:
                    stable_label = stabiliser.update(predicted_character, confidence)

                    if predicted_character in DYNAMIC_GESTURE_LABELS:
                        stable_label = predicted_character
                        reset_stabiliser()

                    cooldown_blocked = False
                    if cooldown_frames > 0:
                        cooldown_frames -= 1
                        if stable_label is not None:
                            if stable_label in DYNAMIC_GESTURE_LABELS:
                                cooldown_frames = 0
                            else:
                                reset_stabiliser()
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
                                and (now - active_label_since)
                                >= STATIC_HOLD_DURATION
                            )
                        )

                        if hold_satisfied:
                            should_commit = (
                                stable_label != last_written_label
                                or stable_label == NO_OUTPUT_LABEL
                            )

                            if should_commit:
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
                                            suggester.apply_suggestion(
                                                composed, suggestion
                                            )
                                        )
                                elif stable_label != NO_OUTPUT_LABEL:
                                    confirmed_text.append(normalized)

                                last_written_label = stable_label

                                if stable_label not in DYNAMIC_GESTURE_LABELS:
                                    cooldown_frames = POST_CONFIRM_COOLDOWN
                                else:
                                    cooldown_frames = 0

                                reset_stabiliser()
                                active_label = None
                                active_label_since = None
                    else:
                        active_label = None
                        active_label_since = None
                else:
                    active_label = None
                    active_label_since = None

            composed_text = "".join(confirmed_text)
            suggestions = suggester.suggest(composed_text)

            # Hiển thị chữ cái dự đoán tạm thời (nếu có)
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
