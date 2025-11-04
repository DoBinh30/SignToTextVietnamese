"""Capture labelled video samples for gesture recognition datasets."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import cv2


WINDOW_NAME = "Video Collection"
DEFAULT_OUTPUT_DIR = Path("data/videos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels",
        nargs="+",
        help="List of labels to record. A sub-directory will be created for each label.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the labelled video folders will be stored.",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=10,
        help="Number of video samples to record for each label.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration in seconds for each recorded sample.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index to use for recording.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second for the recorded videos.",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=640,
        help="Frame width for the capture device.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=480,
        help="Frame height for the capture device.",
    )
    return parser.parse_args()


def _init_camera(args: argparse.Namespace) -> Tuple[cv2.VideoCapture, Tuple[int, int]]:
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.frame_height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or args.frame_width)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.frame_height)
    return cap, (width, height)


def _wait_for_trigger(cap: cv2.VideoCapture, message: str) -> bool:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: unable to read frame from camera while waiting for trigger.")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
            continue
        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            message,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Press SPACE to start, Q to quit",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return True
        if key == ord("q"):
            return False


def _record_sample(
    cap: cv2.VideoCapture,
    output_path: Path,
    duration: float,
    fps: int,
    frame_size: Tuple[int, int],
) -> bool:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        print(f"Error: Unable to open video writer for {output_path}")
        return False

    start_time = time.time()
    frames_recorded = 0
    try:
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Warning: unable to read frame while recording. Skipping frame…")
                continue
            frame = cv2.flip(frame, 1)
            writer.write(frame)
            frames_recorded += 1
            cv2.putText(
                frame,
                "Recording… Press Q to abort",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Recording aborted by user.")
                return False
    finally:
        writer.release()

    if frames_recorded == 0:
        print(f"Warning: no frames recorded for {output_path.name}. Removing file.")
        output_path.unlink(missing_ok=True)
        return False

    print(f"Saved video {output_path} ({frames_recorded} frames)")
    return True


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cap, frame_size = _init_camera(args)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for label in args.labels:
            label_dir = output_dir / label
            label_dir.mkdir(parents=True, exist_ok=True)
            print(f"Collecting videos for label '{label}'")

            for sample_idx in range(args.samples_per_label):
                sample_name = f"{sample_idx:03d}.mp4"
                sample_path = label_dir / sample_name
                message = f"Label: {label} | Sample {sample_idx + 1}/{args.samples_per_label}"
                if not _wait_for_trigger(cap, message):
                    print("Stopping collection by user request.")
                    return
                success = _record_sample(
                    cap=cap,
                    output_path=sample_path,
                    duration=args.duration,
                    fps=args.fps,
                    frame_size=frame_size,
                )
                if not success:
                    sample_path.unlink(missing_ok=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
