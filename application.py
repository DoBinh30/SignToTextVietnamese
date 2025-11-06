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
from typing import Any, Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tensorflow as tf
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
STATIC_HOLD_DURATION = 0.1  # seconds
DYNAMIC_GESTURE_LABELS = {"j", "z", "q", "w"}
NO_OUTPUT_LABEL = "NO_OUTPUT"

PUNCTUATION_DYNAMIC = {"SAC", "HUYEN", "HOI", "NGA"}
PUNCTUATION_STATIC = {"NANG"}
SPECIAL_CHARACTERS = {"^", "aw", "dd", "ow", "uw"}

# Prediction stabilisation parameters chosen to balance responsiveness and
# robustness when handling rapid gesture sequences from video input.
HISTORY_SIZE = 8
MIN_CONSENSUS = 4
DYNAMIC_MIN_CONFIDENCE = 0.75
MIN_CONFIDENCE = 0.75
STATIC_MIN_CONFIDENCE_GAP = 0.15
DYNAMIC_MIN_CONFIDENCE_GAP = 0.2
HAND_STABLE_FRAME_COUNT = 12
HAND_ABSENCE_TIMEOUT_FRAMES = 15
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

        if confidence < self._min_confidence:
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


def draw_hand_highlight(
    frame: np.ndarray,
    hand_landmarks: Any,
    *,
    color: Tuple[int, int, int] = (0, 200, 0),
    thickness: int = 2,
    padding: float = 0.12,
) -> None:
    """Draw a thin square highlight surrounding the detected hand."""

    height, width = frame.shape[:2]
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cx = (min_x + max_x) * 0.5 * width
    cy = (min_y + max_y) * 0.5 * height
    box_w = (max_x - min_x) * width
    box_h = (max_y - min_y) * height
    side = max(box_w, box_h) * (1.0 + padding)
    half = side / 2.0

    left = int(max(cx - half, 0))
    right = int(min(cx + half, width - 1))
    top = int(max(cy - half, 0))
    bottom = int(min(cy + half, height - 1))

    cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)


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
    if "label_encoder" not in model_dict:
        raise ValueError("Model file does not contain required keys")
    sequence_length = model_dict.get("sequence_length")
    if sequence_length is None:
        raise ValueError("Model file missing sequence_length metadata")
    model_config = model_dict.get("model_config")
    model_weights = model_dict.get("model_weights")
    if model_config is None or model_weights is None:
        raise ValueError("Model file missing neural network parameters")
    model = tf.keras.models.model_from_json(model_config)
    model.set_weights(model_weights)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model, model_dict["label_encoder"], sequence_length

# ===== Vietnamese IME helpers (Telex + Tone) =====

# Bảng nguyên âm cơ sở có/không mũ/dấu
_VOWEL_TONE_TABLE_LOWER = {
    "a": {"NONE":"a","SAC":"á","HUYEN":"à","HOI":"ả","NGA":"ã","NANG":"ạ"},
    "ă": {"NONE":"ă","SAC":"ắ","HUYEN":"ằ","HOI":"ẳ","NGA":"ẵ","NANG":"ặ"},
    "â": {"NONE":"â","SAC":"ấ","HUYEN":"ầ","HOI":"ẩ","NGA":"ẫ","NANG":"ậ"},
    "e": {"NONE":"e","SAC":"é","HUYEN":"è","HOI":"ẻ","NGA":"ẽ","NANG":"ẹ"},
    "ê": {"NONE":"ê","SAC":"ế","HUYEN":"ề","HOI":"ể","NGA":"ễ","NANG":"ệ"},
    "i": {"NONE":"i","SAC":"í","HUYEN":"ì","HOI":"ỉ","NGA":"ĩ","NANG":"ị"},
    "o": {"NONE":"o","SAC":"ó","HUYEN":"ò","HOI":"ỏ","NGA":"õ","NANG":"ọ"},
    "ô": {"NONE":"ô","SAC":"ố","HUYEN":"ồ","HOI":"ổ","NGA":"ỗ","NANG":"ộ"},
    "ơ": {"NONE":"ơ","SAC":"ớ","HUYEN":"ờ","HOI":"ở","NGA":"ỡ","NANG":"ợ"},
    "u": {"NONE":"u","SAC":"ú","HUYEN":"ù","HOI":"ủ","NGA":"ũ","NANG":"ụ"},
    "ư": {"NONE":"ư","SAC":"ứ","HUYEN":"ừ","HOI":"ử","NGA":"ữ","NANG":"ự"},
    "y": {"NONE":"y","SAC":"ý","HUYEN":"ỳ","HOI":"ỷ","NGA":"ỹ","NANG":"ỵ"},
}
_VOWEL_TONE_TABLE_UPPER = {
    "A": {"NONE":"A","SAC":"Á","HUYEN":"À","HOI":"Ả","NGA":"Ã","NANG":"Ạ"},
    "Ă": {"NONE":"Ă","SAC":"Ắ","HUYEN":"Ằ","HOI":"Ẳ","NGA":"Ẵ","NANG":"Ặ"},
    "Â": {"NONE":"Â","SAC":"Ấ","HUYEN":"Ầ","HOI":"Ẩ","NGA":"Ẫ","NANG":"Ậ"},
    "E": {"NONE":"E","SAC":"É","HUYEN":"È","HOI":"Ẻ","NGA":"Ẽ","NANG":"Ẹ"},
    "Ê": {"NONE":"Ê","SAC":"Ế","HUYEN":"Ề","HOI":"Ể","NGA":"Ễ","NANG":"Ệ"},
    "I": {"NONE":"I","SAC":"Í","HUYEN":"Ì","HOI":"Ỉ","NGA":"Ĩ","NANG":"Ị"},
    "O": {"NONE":"O","SAC":"Ó","HUYEN":"Ò","HOI":"Ỏ","NGA":"Õ","NANG":"Ọ"},
    "Ô": {"NONE":"Ô","SAC":"Ố","HUYEN":"Ồ","HOI":"Ổ","NGA":"Ỗ","NANG":"Ộ"},
    "Ơ": {"NONE":"Ơ","SAC":"Ớ","HUYEN":"Ờ","HOI":"Ở","NGA":"Ỡ","NANG":"Ợ"},
    "U": {"NONE":"U","SAC":"Ú","HUYEN":"Ù","HOI":"Ủ","NGA":"Ũ","NANG":"Ụ"},
    "Ư": {"NONE":"Ư","SAC":"Ứ","HUYEN":"Ừ","HOI":"Ử","NGA":"Ữ","NANG":"Ự"},
    "Y": {"NONE":"Y","SAC":"Ý","HUYEN":"Ỳ","HOI":"Ỷ","NGA":"Ỹ","NANG":"Ỵ"},
}

# Map ngược: ký tự có dấu -> (base_vowel, current_tone)
_INV_MAP = {}
for base, tones in {**_VOWEL_TONE_TABLE_LOWER, **_VOWEL_TONE_TABLE_UPPER}.items():
    for tone, ch in tones.items():
        _INV_MAP[ch] = (base, tone)

# Biến đổi ^ / aw / ow / uw / dd
_CIRC = {"a":"â","A":"Â","e":"ê","E":"Ê","o":"ô","O":"Ô"}
_BREVE = {"a":"ă","A":"Ă"}
_HORN  = {"o":"ơ","O":"Ơ","u":"ư","U":"Ư"}

def _last_word_bounds(chars: List[str]) -> tuple[int,int]:
    s = "".join(chars)
    if not s:
        return (0,0)
    i = len(s)-1
    while i>=0 and not s[i].isalpha():
        i -= 1
    if i<0: return (len(s), len(s))
    end = i+1
    while i>=0 and s[i].isalpha():
        i -= 1
    start = i+1
    return (start, end)

# Ưu tiên chọn nguyên âm để đặt dấu (gần với quy tắc gõ tiếng Việt phổ biến)
_VOWEL_PRIORITY = ["a","ă","â","e","ê","o","ô","ơ","u","ư","i","y",
                   "A","Ă","Â","E","Ê","O","Ô","Ơ","U","Ư","I","Y"]

def _apply_tone_to_char(ch: str, tone: str) -> str:
    # Nếu không phải nguyên âm có trong bảng -> giữ nguyên
    if ch not in _INV_MAP:
        # Có thể là nguyên âm không dấu (a/e/i/o/u/y...), map về base và tone NONE trước
        base = None
        if ch.lower() in ["a","e","i","o","u","y"]:
            # Chọn base có/không mũ/phụ thuộc vào chính ký tự
            base = ch
            # chuyển base về đúng bảng
            if ch in _VOWEL_TONE_TABLE_LOWER:
                return _VOWEL_TONE_TABLE_LOWER[ch][tone if tone!="NONE" else "NONE"]
            if ch in _VOWEL_TONE_TABLE_UPPER:
                return _VOWEL_TONE_TABLE_UPPER[ch][tone if tone!="NONE" else "NONE"]
            # nếu là a/e/i/o/u/y thường
            if ch.islower():
                return _VOWEL_TONE_TABLE_LOWER[ch][tone if tone!="NONE" else "NONE"]
            else:
                return _VOWEL_TONE_TABLE_UPPER[ch][tone if tone!="NONE" else "NONE"]
        return ch

    base, _old = _INV_MAP[ch]
    table = _VOWEL_TONE_TABLE_UPPER if base.isupper() else _VOWEL_TONE_TABLE_LOWER
    return table[base][tone if tone!="NONE" else "NONE"]

def _choose_vowel_index(word: str) -> int:
    # Trả về index trong word để đặt dấu; nếu không tìm được trả về -1
    idxs = [i for i,ch in enumerate(word) if ch in _INV_MAP or ch.lower() in ["a","e","i","o","u","y"] or ch in ["ă","â","ê","ô","ơ","ư","Ă","Â","Ê","Ô","Ơ","Ư"]]
    if not idxs:
        return -1
    # Ưu tiên theo bảng _VOWEL_PRIORITY
    best = None
    best_rank = 10**9
    for i in idxs:
        ch = word[i]
        # chuyển về base nếu đang là ký tự có dấu
        base = _INV_MAP[ch][0] if ch in _INV_MAP else ch
        # nếu là nguyên âm ASCII thường mà bảng không có (e.g. 'a'), map vào bảng lower/upper tương ứng
        if base not in _VOWEL_TONE_TABLE_LOWER and base not in _VOWEL_TONE_TABLE_UPPER:
            # chuyển 'a'..'y' bình thường sang đúng key
            if base.islower() and base in _VOWEL_TONE_TABLE_LOWER:
                pass
            elif base.isupper() and base in _VOWEL_TONE_TABLE_UPPER:
                pass
        rank = _VOWEL_PRIORITY.index(base) if base in _VOWEL_PRIORITY else 10**6
        if rank < best_rank:
            best_rank = rank
            best = i
    return best if best is not None else -1

def apply_punctuation(confirmed: List[str], tone: str) -> List[str]:
    """Áp dấu thanh vào nguyên âm mục tiêu của từ cuối; có thể đổi nhiều lần."""
    if not confirmed:
        return confirmed
    s = "".join(confirmed)
    L, R = _last_word_bounds(confirmed)
    if L == R:
        return confirmed
    word = s[L:R]
    i = _choose_vowel_index(word)
    if i < 0:
        return confirmed
    # thay nguyên âm ở vị trí i
    chars = list(s)
    chars[L+i] = _apply_tone_to_char(chars[L+i], tone)
    return chars

def apply_special_character(confirmed: List[str], token: str) -> List[str]:
    """Xử lý ^, aw, dd, ow, uw lên ký tự/cuối từ cuối."""
    s = "".join(confirmed)
    L, R = _last_word_bounds(confirmed)
    if token == "dd":
        # nếu ký tự cuối là d/D -> thay thành đ/Đ; nếu không, chèn 'đ'
        if confirmed and confirmed[-1] in ("d","D"):
            confirmed[-1] = "đ" if confirmed[-1]=="d" else "Đ"
        else:
            confirmed.append("đ")
        return confirmed

    if L == R:
        return confirmed
    word = list(s[L:R])

    # chọn vị trí nguyên âm mục tiêu gần nhất phía cuối
    # với ^: chỉ a/e/o; aw: chỉ a; ow: chỉ o; uw: chỉ u
    def _find_last(of_set: set[str]) -> int:
        for j in range(len(word)-1, -1, -1):
            if word[j] in of_set:
                return j
        return -1

    if token == "^":
        idx = _find_last(set(["a","A","e","E","o","O"]))
        if idx >= 0:
            ch = word[idx]
            if ch in _CIRC:
                word[idx] = _CIRC[ch]
    elif token == "aw":
        idx = _find_last(set(["a","A"]))
        if idx >= 0:
            word[idx] = _BREVE.get(word[idx], word[idx])
    elif token == "ow":
        idx = _find_last(set(["o","O"]))
        if idx >= 0:
            word[idx] = _HORN.get(word[idx], word[idx])
    elif token == "uw":
        idx = _find_last(set(["u","U"]))
        if idx >= 0:
            word[idx] = _HORN.get(word[idx], word[idx])

    # ghép lại vào confirmed
    new_s = "".join(s[:L] + "".join(word) + s[R:])
    return list(new_s)

DYNAMIC_LIKE = DYNAMIC_GESTURE_LABELS | PUNCTUATION_DYNAMIC

def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    model, label_encoder, sequence_length = load_model(args.model_path)
    sequence_builder = LandmarkSequenceBuilder(sequence_length=sequence_length)
    stabiliser = PredictionStabiliser(
        history_size=HISTORY_SIZE, min_consensus=MIN_CONSENSUS
    )
    suggester = VietnameseWordSuggester.from_default()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

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
        frames_since_hand = HAND_ABSENCE_TIMEOUT_FRAMES + 1
        hand_visible_frames = 0  # Đếm số frame tay đã ở trong khung

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            frame = cv2.flip(frame, 1)
            result = extractor.extract(frame)
            has_hand = result is not None and result.hand_landmarks is not None
            sequence_builder.append(result.features if has_hand else None)

            if has_hand:
                frames_since_hand = 0
                hand_visible_frames += 1
                hand_detected_recently = True
                draw_hand_highlight(frame, result.hand_landmarks)
            else:
                frames_since_hand += 1
                hand_visible_frames = 0
                if frames_since_hand > HAND_ABSENCE_TIMEOUT_FRAMES:
                    if hand_detected_recently:
                        reset_stabiliser()
                        sequence_builder.reset()
                    hand_detected_recently = False
                    active_label = None
                    active_label_since = None
                    last_written_label = None

            should_attempt_prediction = (
                has_hand
                and hand_detected_recently
                and hand_visible_frames >= HAND_STABLE_FRAME_COUNT
                and sequence_builder.is_ready()
            )

            if should_attempt_prediction:
                flat_sequence = sequence_builder.as_flattened()
                feature_tensor = flat_sequence.reshape(sequence_length, -1)
                feature_tensor = np.expand_dims(feature_tensor, axis=0)
                probabilities = model.predict(feature_tensor, verbose=0)[0]
                sorted_indices = np.argsort(probabilities)[::-1]
                top_index = int(sorted_indices[0])
                confidence = float(probabilities[top_index])
                second_confidence = (
                    float(probabilities[sorted_indices[1]])
                    if len(sorted_indices) > 1
                    else 0.0
                )
                confidence_gap = confidence - second_confidence
                predicted_character = label_encoder.inverse_transform([top_index])[0]

                is_dynamic_like = predicted_character in DYNAMIC_LIKE
                min_confidence = (
                    DYNAMIC_MIN_CONFIDENCE if is_dynamic_like else MIN_CONFIDENCE
                )
                min_gap = (
                    DYNAMIC_MIN_CONFIDENCE_GAP
                    if is_dynamic_like
                    else STATIC_MIN_CONFIDENCE_GAP
                )

                stable_label: Optional[str]
                if confidence >= min_confidence and confidence_gap >= min_gap:
                    if is_dynamic_like:
                        stable_label = predicted_character
                        reset_stabiliser()
                    else:
                        stable_label = stabiliser.update(predicted_character, confidence)
                else:
                    stable_label = None

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
                            and (now - active_label_since) >= STATIC_HOLD_DURATION
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
                                        suggester.apply_suggestion(composed, suggestion)
                                    )
                            elif stable_label != NO_OUTPUT_LABEL:
                                # 1) SPECIAL_CHARACTERS (telex hình thái)
                                if stable_label in SPECIAL_CHARACTERS:
                                    confirmed_text[:] = apply_special_character(confirmed_text, stable_label)

                                # 2) PUNCTUATION (dấu thanh)
                                elif stable_label in PUNCTUATION_DYNAMIC or stable_label in PUNCTUATION_STATIC:
                                    # Cho phép đổi dấu nhiều lần: áp thẳng tone mới vào nguyên âm mục tiêu
                                    confirmed_text[:] = apply_punctuation(confirmed_text, stable_label)

                                # 3) Ký tự alphabet/bình thường -> thêm mới
                                else:
                                    confirmed_text.append(normalized)

                            last_written_label = stable_label

                            if stable_label not in DYNAMIC_LIKE:
                                cooldown_frames = POST_CONFIRM_COOLDOWN
                            else:
                                cooldown_frames = 0

                            reset_stabiliser()
                            active_label = None
                            active_label_since = None
                else:
                    active_label = None
                    active_label_since = None

            composed_text = "".join(confirmed_text)
            suggestions = suggester.suggest(composed_text)

            cv2.imshow(WINDOW_NAME, frame)
            text_display.update(composed_text, suggestions)
            if not text_display.pump_events() or cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    text_display.close()



if __name__ == "__main__":
    main()