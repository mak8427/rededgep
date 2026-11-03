"""Sample reflectance at arbitrary ground points.

This is the function to reach for when you do not want a full orthophoto, just the
reflectance at a set of coordinates (plot centres, field points, validation samples).

Two backends produce the same numbers: a numpy implementation and an optional C kernel
that fuses the per-point work into a single pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry, native
from .chain import dn_to_reflectance, reflectance_scale
from .metadata import ImageMeta

__all__ = ["bilinear", "sample_reflectance", "sample_reflectance_from_file"]


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


def sample_reflectance_from_file(
    path: str | Path,
    meta: ImageMeta,
    sensor,
    camera: geometry.CameraPose,
    lon,
    lat,
    height,
    *,
    margin: int = 2,
    backend: str = "auto",
) -> np.ndarray:
    """Reflectance at ground points, reading only the pixels the points touch.

    A 1456x1088 RedEdge-P band is 3.2 MB; sampling twenty plot centres needs a few
    kilobytes of it.  This projects the points first, reads the bounding window of the
    hits, and samples that.  When the points are spread over the whole frame the window
    is the whole frame and this is equivalent to :func:`sample_reflectance`.

    The vignetting is evaluated at the *global* pixel coordinates -- the polynomial is
    defined over the full frame, not over the window.

    Parameters
    ----------
    path
        Raw band image.
    margin
        Extra pixels added around the bounding box, so the bilinear taps are always
        inside the window.

    Notes
    -----
    The windowed path uses the numpy backend, because the C kernel computes pixel
    coordinates against the image it is given and the window offset would have to be
    applied inside the kernel.  That is the right trade when the points touch a small part
    of the frame.  When they cover most of it -- which is the case for dense grids and for
    millions of points -- reading the whole frame and using the native kernel is faster,
    so that is what happens: the window is only used if it is meaningfully smaller than
    the frame.
    """
    # Windowed reading trades a cheaper read for a more expensive sampler. Measured on a
    # RedEdge-P frame: reading costs ~0.75 ns per pixel of the frame (~1.2 ms for 1.58 M
    # px from page cache), while the numpy sampler costs ~130 ns/point against the
    # kernel's ~34 ns/point. Windowed reading therefore wins only while the pixels saved
    # outweigh the extra sampling cost:
    #     (frame_area - window_area) * 0.75 ns  >  n * 96 ns
    # which rearranges to the test below. For a handful of plot centres the window is
    # almost always worth it; for a dense million-point grid it never is.
    pixels_saved_break_even = 128  # = 96 ns/point / 0.75 ns/px
    from .readers import read_dn, read_window

    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    height = np.asarray(height, dtype=np.float64)
    out = np.full(lon.shape, np.nan, dtype=np.float64)
    frame_area = sensor.width * sensor.height

    # Cheap pre-check that needs no projection: beyond this many points the window can
    # never repay the sampler it forces, so read the frame and use the kernel directly.
    if pixels_saved_break_even * lon.size >= frame_area:
        dn = read_dn(path)
        return sample_reflectance(dn, meta, sensor, camera, lon, lat, height, backend=backend)

    u, v = geometry.project(camera, sensor, lon, lat, height)
    inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= sensor.width - 1) & (v >= 0) & (v <= sensor.height - 1)
    if not inside.any():
        return out

    x0 = max(0, int(np.floor(u[inside].min())) - margin)
    y0 = max(0, int(np.floor(v[inside].min())) - margin)
    x1 = min(sensor.width, int(np.ceil(u[inside].max())) + margin + 1)
    y1 = min(sensor.height, int(np.ceil(v[inside].max())) + margin + 1)

    area = (x1 - x0) * (y1 - y0)
    if area + pixels_saved_break_even * int(np.count_nonzero(inside)) >= frame_area:
        dn = read_dn(path)
        return sample_reflectance(dn, meta, sensor, camera, lon, lat, height, backend=backend)

    window = read_window(path, x0, y0, x1, y1).astype(np.float64)
    values = bilinear(window, u - x0, v - y0)
    # vignetting at global coordinates
    result = dn_to_reflectance(values, meta, sensor, u, v)
    return result


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
