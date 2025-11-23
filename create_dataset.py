"""Create dataset from images and videos."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from sign_language import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/images"),
        help="Directory containing per-label subdirectories of images.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/videos"),
        help="Directory containing per-label subdirectories of videos.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="Number of frames to represent each sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_sequence.pickle"),
        help="Path to save the dataset pickle file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(
        image_dir=args.image_dir if args.image_dir.exists() else None,
        video_dir=args.video_dir if args.video_dir.exists() else None,
        sequence_length=args.sequence_length,
    )
    with args.output.open("wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset with {len(dataset['data'])} samples to {args.output}")


if __name__ == "__main__":
    main()