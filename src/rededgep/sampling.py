"""Sample reflectance at arbitrary ground points.

This is the function to reach for when you do not want a full orthophoto, just the
reflectance at a set of coordinates (plot centres, field points, validation samples).

Two backends produce the same numbers: a numpy implementation and an optional C kernel
that fuses the per-point work into a single pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry, native
from .chain import dn_to_reflectance, reflectance_scale
from .metadata import ImageMeta

__all__ = ["bilinear", "sample_reflectance"]


def bilinear(array: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``array`` at ``(u, v)``; ``nan`` outside the image."""
    height, width = array.shape
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    # Replace invalid coordinates before flooring: casting NaN or -inf to int64 is
    # undefined and yields INT64_MIN, which then indexes out of bounds.  The mask
    # discards these values afterwards, but only after they have been clipped and cast.
    uu = np.where(ok, u, 0.0)
    vv = np.where(ok, v, 0.0)
    x0 = np.floor(uu).astype(np.int64)
    y0 = np.floor(vv).astype(np.int64)
    fx = uu - x0
    fy = vv - y0
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    values = (
        array[y0, x0] * (1 - fx) * (1 - fy)
        + array[y0, x1] * fx * (1 - fy)
        + array[y1, x0] * (1 - fx) * fy
        + array[y1, x1] * fx * fy
    )
    return np.where(ok, values, np.nan)


@dataclass(frozen=True)
class SampleResult:
    """Reflectance sampled at ground points."""

    reflectance: np.ndarray
    u: np.ndarray | None = None
    v: np.ndarray | None = None


def sample_reflectance(
    dn: np.ndarray,
    meta: ImageMeta,
    sensor,
    camera: geometry.CameraPose,
    lon,
    lat,
    height,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Reflectance at ground points ``(lon, lat, height)`` seen by ``camera``.

    Parameters
    ----------
    dn
        Raw digital numbers for this band.
    meta
        Metadata for this image.
    sensor
        Calibration for this band.
    camera
        Camera pose from :func:`rededgep.geometry.load_cameras_txt`.
    lon, lat, height
        Ground coordinates (degrees, degrees, metres above the CRS datum).
    backend
        ``"auto"`` (default), ``"numpy"`` or ``"native"``.

    Returns
    -------
    numpy.ndarray
        Reflectance, ``nan`` where the point does not land on the image.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)

    use_native = backend == "native" or (backend == "auto" and native.available())
    if use_native:
        # The kernel does the geodetic -> ECEF -> camera conversion itself, so the only
        # per-point Python work left is flattening the input arrays.
        out = native.call_kernel_geodetic(
            dn, lon, lat, height,
            camera.lon, camera.lat, camera.alt, camera.rotation,
            sensor.focal_px, sensor.cx, sensor.cy,
            (sensor.k1, sensor.k2, sensor.k3, sensor.p1, sensor.p2),
            sensor.vignetting, sensor.vignetting_mean(), meta.black_level,
            reflectance_scale(meta), meta.horizontal_irradiance,
        )
        return out.reshape(lon.shape)

    if lat.size != lon.size or height.size != lon.size:
        raise ValueError("lon, lat and height must have the same size")
    u, v = geometry.project(camera, sensor, lon, lat, height)
    values = bilinear(np.asarray(dn), u, v)
    return dn_to_reflectance(values, meta, sensor, u, v)
