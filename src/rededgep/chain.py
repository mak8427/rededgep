"""The raw-DN to reflectance chain.

Metashape's RedEdge-P reflectance is reproduced by, in this order::

    DN_corr = (DN - black_level) / (V / mean(V))
    L       = a1 * DN_corr / (gain * exposure * 2**16)
    rho     = pi * L / E_horizontal

with ``gain = ISO / 100`` and the vignetting polynomial

    V(x, y) = exp( sum_ij c_ij * x**i * y**j ),  x = 2*(col+0.5)/w - 1,  y = 2*(row+0.5)/h - 1

Two details matter and are easy to get wrong:

* the vignetting is **divided**, not multiplied;
* it is normalised to unit mean, which is why dividing by ``V`` alone leaves a constant
  ~22 % offset.

MicaSense's reference implementation also applies a row-gradient term ``R(y)``; Metashape
does not, and enabling it moves Green/Red/NIR by ~5 % in the wrong direction, so it is
off by default.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .metadata import ImageMeta

__all__ = [
    "vignetting",
    "vignetting_mean",
    "dn_to_radiance",
    "dn_to_reflectance",
    "reflectance_scale",
]

_VIGNETTING_MEAN_CACHE: dict[tuple, float] = {}


def _coeff_key(vignetting: Sequence[tuple[int, int, float]]) -> tuple:
    return tuple((int(i), int(j), float(c)) for i, j, c in vignetting)


def vignetting(
    vignetting: Sequence[tuple[int, int, float]],
    u: np.ndarray,
    v: np.ndarray,
    width: int = 1456,
    height: int = 1088,
) -> np.ndarray:
    """Evaluate ``V(x, y) = exp(sum c_ij x^i y^j)`` at pixel coordinates ``u, v``."""
    x = 2.0 * (np.asarray(u, dtype=np.float64) + 0.5) / width - 1.0
    y = 2.0 * (np.asarray(v, dtype=np.float64) + 0.5) / height - 1.0
    exponent = np.zeros(x.shape, dtype=np.float64)
    for i, j, coefficient in vignetting:
        exponent += coefficient * (x**i) * (y**j)
    return np.exp(np.clip(exponent, -50.0, 50.0))


def vignetting_mean(
    vignetting: Sequence[tuple[int, int, float]],
    width: int = 1456,
    height: int = 1088,
) -> float:
    """Mean of :func:`vignetting` over the full frame (cached; constant per sensor)."""
    key = (_coeff_key(vignetting), width, height)
    value = _VIGNETTING_MEAN_CACHE.get(key)
    if value is None:
        vv, uu = np.mgrid[0:height, 0:width]
        reference = vignetting_impl(vignetting, uu.astype(float), vv.astype(float), width, height)
        value = float(reference.mean())
        _VIGNETTING_MEAN_CACHE[key] = value
    return value


# ``vignetting`` above shadows its own parameter name inside ``vignetting_mean``; keep a
# private alias so the cache can call it unambiguously.
vignetting_impl = vignetting


def reflectance_scale(meta: ImageMeta) -> float:
    """Per-image factor that converts corrected DN to radiance."""
    return meta.a1 / (meta.gain * meta.exposure_s * 2.0**16)


def dn_to_radiance(
    dn: np.ndarray,
    meta: ImageMeta,
    sensor,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
) -> np.ndarray:
    """Raw DN to calibrated radiance (W/m^2/sr/nm)."""
    corrected = np.asarray(dn, dtype=np.float64) - meta.black_level
    np.clip(corrected, 0.0, None, out=corrected)
    if u is not None and v is not None:
        weight = vignetting(sensor.vignetting, u, v, sensor.width, sensor.height)
        corrected = corrected / (weight / vignetting_mean(sensor.vignetting, sensor.width, sensor.height))
    return reflectance_scale(meta) * corrected


def dn_to_reflectance(
    dn: np.ndarray,
    meta: ImageMeta,
    sensor,
    u: np.ndarray | None = None,
    v: np.ndarray | None = None,
) -> np.ndarray:
    """Raw DN to reflectance (dimensionless, 0-1).

    Parameters
    ----------
    dn
        Raw digital numbers, as read from the TIFF.
    meta
        Metadata for this image, from :func:`rededgep.metadata.read_metadata`.
    sensor
        Calibration for this band, from :class:`rededgep.calibration.Sensor`.
    u, v
        Pixel coordinates at which to evaluate the vignetting.  Pass these when ``dn``
        holds sampled values (one per point) rather than a full image.  When omitted the
        vignetting is skipped, which is only correct for a flat-fielded image.

    Raises
    ------
    ValueError
        If the image carries no usable DLS irradiance.
    """
    if not meta.has_irradiance:
        raise ValueError(
            f"{meta.path} has no DLS horizontal irradiance; reflectance cannot be computed. "
            "Use a reflectance panel instead, or supply the irradiance explicitly."
        )
    radiance = dn_to_radiance(dn, meta, sensor, u, v)
    return np.pi * radiance / meta.horizontal_irradiance
