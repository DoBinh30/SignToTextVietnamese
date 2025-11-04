"""Realtime sign language inference supporting dynamic gestures."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
from sign_language import HandLandmarkExtractor, LandmarkSequenceBuilder


DEFAULT_MODEL_PATH = Path("model.p")


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

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    predicted_character: Optional[str] = None

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

            cv2.imshow("Sign Language Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
