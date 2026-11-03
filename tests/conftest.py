"""Shared fixtures.

The suite is written for the standard library ``unittest`` runner so it needs no test
dependencies, but pytest can run it too.  Everything here is synthetic: no field data is
required, so the tests run anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rededgep.calibration import Calibration  # noqa: E402
from rededgep.geometry import CameraPose  # noqa: E402
from rededgep.metadata import ImageMeta  # noqa: E402


def make_meta(band: str = "Green", **overrides) -> ImageMeta:
    values = dict(
        path="synthetic.tif",
        band=band,
        iso=100.0,
        exposure_s=0.0015,
        black_level=3840.0,
        a1=1.7e-4,
        a2=0.0,
        a3=0.0,
        horizontal_irradiance=1.5,
        spectral_irradiance=1.52,
        solar_elevation_rad=1.0,
        solar_azimuth_rad=3.0,
        width=1456,
        height=1088,
    )
    values.update(overrides)
    return ImageMeta(**values)


def make_camera(lon: float = 9.9146, lat: float = 51.5628, alt: float = 207.0) -> CameraPose:
    """A synthetic nadir camera: ENU up maps to camera -Z, so depth is positive below."""
    return CameraPose(lon=lon, lat=lat, alt=alt, rotation=np.eye(3), label="synthetic")


def make_calibration() -> Calibration:
    return Calibration.rededgep()
