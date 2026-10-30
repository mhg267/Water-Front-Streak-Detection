"""Data passed between the pipeline steps."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class FrameData:
    """Steps 3–5 for one frame."""
    row: int
    frame_id: int
    mask_path: Path
    radius: np.ndarray          # r(θ) of the 681 rays (step 3)
    front_points: np.ndarray    # (x, y) of the front point on each ray (step 3)
    detected: np.ndarray        # True = real ray, False = interpolated ray (step 3)
    residuals: np.ndarray       # one residual curve per baseline window (step 5)


@dataclass
class Node:
    """A local peak of the bump or dent map (step 7)."""
    node_id: int
    row: int
    frame_id: int
    feature_type: str           # "protrusion" (bump) or "indentation" (dent)
    angle_index: int
    score_px: float
    dominant_scale: int
    x: int
    y: int
    radius_px: float


@dataclass
class TrackPoint:
    frame_id: int
    row: int
    feature_type: str
    x: int
    y: int
    angle_index: int
    angle_deg: float
    radius_px: float
    score_px: float
    dominant_scale: int


@dataclass
class Track:
    track_id: int
    feature_type: str
    points: list = field(default_factory=list)
    edge_mutual: list = field(default_factory=list)
    terminal_reason: str = "signal_ended"   # how it ended, or why it was removed

    @property
    def first(self):
        return self.points[0]

    @property
    def last(self):
        return self.points[-1]
