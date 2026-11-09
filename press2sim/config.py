"""Physical constants of the capture setup.

Every number here describes the real objects. To change one without editing code,
put the same keys in a JSON file and pass it with --config, for example
{"board": {"printed_width_mm": 259.4}, "indenter": {"cap_dot_mm": 8.7}}.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class Block:
    length_mm: float = 234.0   # along the filmed face (x)
    depth_mm: float = 139.5    # front to back (y)
    height_mm: float = 73.0    # base to top (z)
    mass_kg: float = 0.078


@dataclass
class DotGrid:
    cols: int = 15
    rows: int = 7
    pitch_mm: float = 10.0
    x0_mm: float = 47.0        # first column, measured from the block's left end
    z0_mm: float = 8.0         # bottom row, measured from the base
    dot_diameter_mm: float = 4.0
    measured_row_span_mm: float = 59.0   # ruler: bottom-row dot centre to top-row dot centre (0 = unknown)


@dataclass
class Board:
    """Printed checkerboard: 26 x 11 squares of 10 mm, 10 mm white margin, card stands on the table."""
    inner_cols: int = 25
    inner_rows: int = 10
    square_mm: float = 10.0
    margin_mm: float = 10.0
    printed_width_mm: float = 260.0    # measure the black-and-white area; enter the true size if the print scaled
    printed_height_mm: float = 110.0

    def corner_xz_mm(self):
        """(rows, cols, 2) inner-corner coordinates on the card, origin at the card's bottom-left corner.

        x runs to the right, z runs up. Row 0 is the lowest row of inner corners, 20 mm above the
        card's bottom edge for a true-size print. The card stands on the table, so z is height above it.
        """
        import numpy as np
        sx = self.printed_width_mm / ((self.inner_cols + 1) * self.square_mm)
        sz = self.printed_height_mm / ((self.inner_rows + 1) * self.square_mm)
        i = np.arange(self.inner_cols)
        j = np.arange(self.inner_rows)
        x = self.margin_mm + sx * self.square_mm * (i + 1)
        z = self.margin_mm + sz * self.square_mm * (j + 1)
        X, Z = np.meshgrid(x, z)
        return np.stack([X, Z], axis=-1)


@dataclass
class Indenter:
    cap_width_mm: float = 60.0
    cap_height_mm: float = 19.0
    cap_corner_radius_mm: float = 12.0
    body_width_mm: float = 61.0
    cap_dot_mm: float = 8.5      # cap dot centre above the cap's flat bottom face
    tape_dot_mm: float = 55.0    # tape dot centre above the cap's flat bottom face
    dot_diameter_mm: float = 4.0


@dataclass
class Tuning:
    rest_min_s: float = 0.25          # still period needed to accept the rest state
    rest_motion_px: float = 0.6       # max dot motion between sampled frames while at rest
    rest_sample_hz: float = 10.0      # how often to sample frames while searching for rest
    rest_average_s: float = 0.4       # rest positions are averaged over this long
    contrast_fraction: float = 0.45   # a dot pixel must be this dark relative to the rest-frame dot contrast
    min_area_ratio: float = 0.2       # tracked blob area vs its rest area, outside this range = invalid
    max_area_ratio: float = 3.5
    bottle_search_px: float = 40.0    # search radius for the two bottle dots
    contact_threshold_mm: float = 0.3
    release_gap_mm: float = 2.0
    bg_features: int = 40             # static background features for the camera-still check


@dataclass
class Config:
    block: Block = field(default_factory=Block)
    grid: DotGrid = field(default_factory=DotGrid)
    board: Board = field(default_factory=Board)
    indenter: Indenter = field(default_factory=Indenter)
    tuning: Tuning = field(default_factory=Tuning)

    def to_dict(self):
        return asdict(self)


def _update(obj, d):
    for f in fields(obj):
        if f.name not in d:
            continue
        cur = getattr(obj, f.name)
        if is_dataclass(cur):
            _update(cur, d[f.name])
        else:
            setattr(obj, f.name, type(cur)(d[f.name]))


def load_config(path: str | Path | None = None) -> Config:
    cfg = Config()
    if path:
        _update(cfg, json.loads(Path(path).read_text()))
    return cfg
