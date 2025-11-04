"""Utility script to capture image samples for training."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


DATA_DIR = Path("data/images")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["8", "10", "11", "12", "13", "14"]
DATASET_SIZE = 300


def _ensure_camera(camera_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")
    return cap


def _read_frame(cap: cv2.VideoCapture) -> Tuple[bool, np.ndarray]:
    ret, frame = cap.read()
    if not ret or frame is None:
        return False, frame
    frame = cv2.flip(frame, 1)
    return True, frame


def main() -> None:
    cap = _ensure_camera(0)
    try:
        for label in LABELS:
            label_dir = DATA_DIR / label
            label_dir.mkdir(parents=True, exist_ok=True)

            print(f"Collecting data for class {label}")

            while True:
                ret, frame = _read_frame(cap)
                if not ret:
                    print("Warning: unable to read frame from camera. Retrying…")
                    continue
                cv2.putText(
                    frame,
                    'Ready? Press "Q" ! :)',
                    (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.imshow("frame", frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

            counter = 0
            while counter < DATASET_SIZE:
                ret, frame = _read_frame(cap)
                if not ret:
                    print("Warning: unable to read frame from camera. Skipping frame…")
                    continue
                cv2.imshow("frame", frame)
                cv2.waitKey(25)
                cv2.imwrite(str(label_dir / f"{counter}.jpg"), frame)
                counter += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
