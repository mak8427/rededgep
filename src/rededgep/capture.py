"""High-level access to one MicaSense RedEdge-P capture.

A capture is the six images sharing a capture id::

    IMG_0000_1.tif  Blue
    IMG_0000_2.tif  Green
    IMG_0000_3.tif  Red
    IMG_0000_4.tif  NIR
    IMG_0000_5.tif  Red edge
    IMG_0000_6.tif  Panchro

The usual entry point is :meth:`Capture.from_directory` followed by
:meth:`Capture.reflectance`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import Calibration
from .chain import dn_to_reflectance
from .metadata import BANDS, SUFFIX_TO_BAND, ImageMeta, read_metadata, MetadataError
from .readers import read_dn

__all__ = ["Capture"]

#: Canonical order used for exported stacks.
STACK_ORDER = ("Blue", "Green", "Red", "Red edge", "NIR")


@dataclass
class Capture:
    """Six band images plus their metadata, for one capture id."""

    name: str
    dn: dict[str, np.ndarray]
    meta: dict[str, ImageMeta]

    @property
    def bands(self) -> tuple[str, ...]:
        return tuple(b for b in BANDS if b in self.dn)

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.dn.values())).shape

    @classmethod
    def from_directory(cls, directory: str | Path, name: str = "IMG_0000") -> "Capture":
        """Load a capture given the directory holding ``IMG_<id>_1..6.tif``."""
        directory = Path(directory)
        dn: dict[str, np.ndarray] = {}
        meta: dict[str, ImageMeta] = {}
        for suffix, band in SUFFIX_TO_BAND.items():
            path = directory / f"{name}_{suffix}.tif"
            if not path.is_file():
                continue
            dn[band] = read_dn(path)
            try:
                meta[band] = read_metadata(path)
            except MetadataError:
                continue
        if not dn:
            raise FileNotFoundError(f"no {name}_*.tif in {directory}")
        return cls(name=name, dn=dn, meta=meta)

    def reflectance(self, calibration: Calibration | None = None) -> dict[str, np.ndarray]:
        """Compute reflectance for every band.

        The vignetting is evaluated per pixel, so the result is a full-frame reflectance
        image per band.
        """
        calibration = calibration or Calibration.rededgep()
        out: dict[str, np.ndarray] = {}
        for band, dn in self.dn.items():
            meta = self.meta.get(band)
            if meta is None:
                continue
            sensor = calibration.sensor(band).scaled(width=meta.width, height=meta.height)
            rows, cols = dn.shape
            u, v = np.meshgrid(np.arange(cols, dtype=np.float64), np.arange(rows, dtype=np.float64))
            out[band] = dn_to_reflectance(dn, meta, sensor, u, v).astype(np.float32)
        return out

    def stack(self, calibration: Calibration | None = None) -> np.ndarray:
        """Reflectance as a ``(5, rows, cols)`` float32 array in canonical band order."""
        reflectance = self.reflectance(calibration)
        missing = [b for b in STACK_ORDER if b not in reflectance]
        if missing:
            raise ValueError(f"capture is missing bands: {missing}")
        return np.stack([reflectance[b] for b in STACK_ORDER])
