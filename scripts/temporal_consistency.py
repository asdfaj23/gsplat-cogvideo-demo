#!/usr/bin/env python3
"""Simple temporal consistency metrics for generated video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_image(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    return gx, gy


def evaluate(frames_dir: Path) -> dict[str, float | int]:
    frame_paths = sorted(
        p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )
    if len(frame_paths) < 2:
        raise ValueError("Need at least 2 frames to evaluate temporal consistency.")

    l1_values: list[float] = []
    grad_l1_values: list[float] = []

    prev = load_image(frame_paths[0])
    prev_gray = prev.mean(axis=2)
    prev_gx, prev_gy = gradients(prev_gray)

    for frame_path in frame_paths[1:]:
        cur = load_image(frame_path)
        if cur.shape != prev.shape:
            raise ValueError(
                f"Frame shape mismatch: {frame_path.name} has {cur.shape}, expected {prev.shape}."
            )

        cur_gray = cur.mean(axis=2)
        cur_gx, cur_gy = gradients(cur_gray)

        l1_values.append(float(np.mean(np.abs(cur - prev))))
        grad_l1_values.append(
            float(np.mean(np.abs(cur_gx - prev_gx)) + np.mean(np.abs(cur_gy - prev_gy)))
        )

        prev = cur
        prev_gx, prev_gy = cur_gx, cur_gy

    mean_l1 = float(np.mean(l1_values))
    mean_grad_l1 = float(np.mean(grad_l1_values))
    temporal_score = float(1.0 / (1.0 + mean_l1 + 0.5 * mean_grad_l1))

    return {
        "num_frames": len(frame_paths),
        "num_pairs": len(l1_values),
        "mean_l1": mean_l1,
        "mean_grad_l1": mean_grad_l1,
        "temporal_score": temporal_score,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_dir", type=Path, required=True, help="Directory containing video frames")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.frames_dir.exists() or not args.frames_dir.is_dir():
        raise ValueError(f"frames_dir does not exist or is not a directory: {args.frames_dir}")

    results = evaluate(args.frames_dir)
    text = json.dumps(results, ensure_ascii=False, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    print(text)


if __name__ == "__main__":
    main()
