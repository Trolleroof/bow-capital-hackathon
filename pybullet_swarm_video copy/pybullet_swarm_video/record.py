from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np


def tile_frames(
    frames: Sequence[np.ndarray],
    tile_columns: int | None = None,
    gap_px: int = 6,
    background_rgb: tuple[int, int, int] = (6, 8, 12),
) -> np.ndarray:
    if not frames:
        raise ValueError("expected at least one frame")

    frame_h, frame_w, channels = frames[0].shape
    if channels != 3:
        raise ValueError(f"expected RGB frames, got shape {frames[0].shape}")

    count = len(frames)
    cols = tile_columns or math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    border_px = max(1, min(2, gap_px))

    canvas_h = rows * frame_h + gap_px * (rows + 1)
    canvas_w = cols * frame_w + gap_px * (cols + 1)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(background_rgb, dtype=np.uint8)

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        top = gap_px + row * (frame_h + gap_px)
        left = gap_px + col * (frame_w + gap_px)
        canvas[top : top + frame_h, left : left + frame_w] = frame

        border_color = np.array(
            [
                (60 + idx * 35) % 255,
                (150 + idx * 20) % 255,
                (220 - idx * 25) % 255,
            ],
            dtype=np.uint8,
        )
        canvas[top - border_px : top, left : left + frame_w] = border_color
        canvas[
            top + frame_h : top + frame_h + border_px,
            left : left + frame_w,
        ] = border_color
        canvas[top : top + frame_h, left - border_px : left] = border_color
        canvas[
            top : top + frame_h,
            left + frame_w : left + frame_w + border_px,
        ] = border_color

    return canvas


class TiledVideoRecorder:
    def __init__(self, output_path: Path, fps: int) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.writer = imageio.get_writer(str(output_path), fps=fps)

    def append(self, frame: np.ndarray) -> None:
        self.writer.append_data(frame)

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> TiledVideoRecorder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MultiVideoRecorder:
    def __init__(self, output_dir: Path, fps: int, stem: str = "drone") -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.fps = fps
        self.stem = stem
        self._writers: list[imageio.Writer] = []

    def append(self, frames: Sequence[np.ndarray]) -> None:
        if not self._writers:
            self._writers = [
                imageio.get_writer(
                    str(self.output_dir / f"{self.stem}_{idx:02d}.mp4"),
                    fps=self.fps,
                )
                for idx in range(len(frames))
            ]
        if len(frames) != len(self._writers):
            raise ValueError(
                f"frame count changed from {len(self._writers)} to {len(frames)}"
            )
        for writer, frame in zip(self._writers, frames, strict=True):
            writer.append_data(frame)

    def close(self) -> None:
        for writer in self._writers:
            writer.close()
        self._writers.clear()

    def __enter__(self) -> MultiVideoRecorder:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
