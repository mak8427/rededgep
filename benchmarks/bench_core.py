#!/usr/bin/env python3
"""Benchmarks for rededgep.

Three things are measured:

1. ``sample_reflectance`` — numpy backend against the fused C kernel, at several
   sample counts.  This is the operation most users hit in a loop.
2. The cost of the vignetting normalisation, cached versus recomputed.  Recomputing it
   per call is the single biggest performance trap in this pipeline.
3. A full-frame ``Capture.reflectance`` for reference.

Run with::

    python benchmarks/bench_core.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rededgep as rp  # noqa: E402
from rededgep import native  # noqa: E402
from rededgep.chain import vignetting  # noqa: E402

REPEATS = 5


def timed(function, repeats: int = REPEATS) -> float:
    """Best-of-N wall time in seconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def make_inputs(n: int):
    calibration = rp.Calibration.rededgep()
    sensor = calibration.sensor("Green")
    meta = rp.ImageMeta(
        path="bench", band="Green", iso=100.0, exposure_s=0.0015, black_level=3840.0,
        a1=1.7e-4, a2=0.0, a3=0.0, horizontal_irradiance=1.5, spectral_irradiance=1.52,
        solar_elevation_rad=1.0, solar_azimuth_rad=3.0, width=1456, height=1088,
    )
    camera = rp.CameraPose(lon=9.9146, lat=51.5628, alt=207.0, rotation=np.eye(3))
    rng = np.random.default_rng(0)
    dn = rng.integers(4000, 40000, size=(1088, 1456), dtype=np.uint16)
    lon = camera.lon + (rng.random(n) - 0.5) * 8e-5
    lat = camera.lat + (rng.random(n) - 0.5) * 6e-5
    height = np.full(n, camera.alt - 20.0)
    return dn, meta, sensor, camera, lon, lat, height


def bench_sampling() -> list[tuple[int, float, float]]:
    if not native.available():
        try:
            native.build()
        except Exception:
            pass
    rows = []
    print("sampling: numpy vs fused C kernel")
    print(f"  {'points':>10s} {'numpy (ms)':>12s} {'C (ms)':>10s} {'speedup':>9s}")
    for n in (10_000, 100_000, 1_000_000):
        dn, meta, sensor, camera, lon, lat, height = make_inputs(n)
        t_numpy = timed(lambda: rp.sample_reflectance(
            dn, meta, sensor, camera, lon, lat, height, backend="numpy"))
        if native.available():
            t_native = timed(lambda: rp.sample_reflectance(
                dn, meta, sensor, camera, lon, lat, height, backend="native"))
            print(f"  {n:10,d} {t_numpy*1e3:12.1f} {t_native*1e3:10.1f} {t_numpy/t_native:8.1f}x")
            rows.append((n, t_numpy, t_native))
        else:
            print(f"  {n:10,d} {t_numpy*1e3:12.1f} {'n/a':>10s} {'n/a':>9s}")
            rows.append((n, t_numpy, float("nan")))
    return rows


def bench_vignetting_cache() -> tuple[float, float]:
    calibration = rp.Calibration.rededgep()
    sensor = calibration.sensor("Green")
    height, width = sensor.height, sensor.width
    vv, uu = np.mgrid[0:height, 0:width]
    u = uu.astype(float)
    v = vv.astype(float)

    t_uncached = timed(lambda: float(vignetting(sensor.vignetting, u, v, width, height).mean()), repeats=3)
    t_cached = timed(lambda: calibration.sensor("Green").vignetting_mean(), repeats=1000)
    print("\nvignetting normalisation")
    print(f"  recompute full frame : {t_uncached*1e3:8.1f} ms")
    print(f"  cached               : {t_cached*1e6:8.1f} us")
    print(f"  speedup              : {t_uncached/t_cached:8.0f}x")
    return t_uncached, t_cached


def bench_full_frame() -> float:
    calibration = rp.Calibration.rededgep()
    shape = (1088, 1456)
    dn = {band: np.full(shape, 9000, dtype=np.uint16) for band in ("Blue", "Green", "Red", "Red edge", "NIR")}
    meta = {band: rp.ImageMeta(
        path="bench", band=band, iso=100.0, exposure_s=0.0015, black_level=3840.0,
        a1=1.7e-4, a2=0.0, a3=0.0, horizontal_irradiance=1.5, spectral_irradiance=1.52,
        solar_elevation_rad=1.0, solar_azimuth_rad=3.0, width=shape[1], height=shape[0],
    ) for band in dn}
    capture = rp.Capture(name="bench", dn=dn, meta=meta)
    t = timed(lambda: capture.reflectance(calibration), repeats=3)
    print("\nfull frame")
    print(f"  Capture.reflectance, 5 bands x {shape[0]}x{shape[1]}: {t*1e3:8.1f} ms")
    return t


def main() -> int:
    print(f"rededgep {rp.__version__}  |  C kernel: {'yes' if native.available() else 'no'}")
    print(f"numpy {np.__version__}\n")
    bench_sampling()
    bench_vignetting_cache()
    bench_full_frame()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
