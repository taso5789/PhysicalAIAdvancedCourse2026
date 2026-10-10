"""Color-based vision baseline. No learned model or simulator object positions."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    color: str
    center_x: float
    pixels: int
    distance: float


def detect_color(rgb, depth, color):
    channel = {"red": 0, "green": 1, "blue": 2}[color]
    values = rgb.astype(np.float32)
    selected = values[:, :, channel]
    other = np.max(np.delete(values, channel, axis=2), axis=2)
    mask = (selected > 65) & (selected > other * 1.6)
    rows, cols = np.nonzero(mask)
    if len(cols) < 30:
        return None
    distances = depth[mask]
    distances = distances[np.isfinite(distances) & (distances > 0)]
    if not len(distances):
        return None
    return Detection(color, float(cols.mean() / (rgb.shape[1] - 1) * 2 - 1),
                     len(cols), float(np.median(distances)))
