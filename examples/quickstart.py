#!/usr/bin/env python3
"""End-to-end demo of rededgep.

Runs without any data by generating a synthetic capture, and shows the real-world calls
commented alongside.

    python examples/quickstart.py
    python examples/quickstart.py /path/to/0004SET/000 IMG_0000
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rededgep as rp


def synthetic_capture() -> rp.Capture:
    """A fake capture: flat DN per band with realistic metadata."""
    shape = (1088, 1456)
    bands = ("Blue", "Green", "Red", "Red edge", "NIR")
    dn = {band: np.full(shape, 8000 + 2000 * i, dtype=np.uint16) for i, band in enumerate(bands)}
    meta = {
        band: rp.ImageMeta(
            path=f"synthetic_{band}.tif", band=band, iso=100.0, exposure_s=0.0015,
            black_level=3840.0, a1=1.7e-4, a2=0.0, a3=0.0,
            horizontal_irradiance=1.5, spectral_irradiance=1.52,
            solar_elevation_rad=1.0, solar_azimuth_rad=3.0,
            width=shape[1], height=shape[0],
        )
        for band in bands
    }
    return rp.Capture(name="synthetic", dn=dn, meta=meta)


def main(argv: list[str]) -> int:
    if len(argv) >= 2:
        directory = Path(argv[0])
        name = argv[1]
        print(f"loading {name} from {directory}")
        capture = rp.Capture.from_directory(directory, name)
    else:
        print("no data given — using a synthetic capture")
        capture = synthetic_capture()

    print(f"bands: {capture.bands}")

    calibration = rp.Calibration.rededgep()
    print(f"calibration: {calibration.source}")

    stack = capture.stack(calibration)
    print(f"stack: {stack.shape} {stack.dtype}")
    for index, band in enumerate(rp.STACK_ORDER):
        band_values = stack[index][np.isfinite(stack[index])]
        print(f"  {band:9s} median {np.median(band_values):.4f}")

    print(f"\nnative kernel available: {rp.native.available()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
