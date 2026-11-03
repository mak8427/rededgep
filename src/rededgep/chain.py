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

Performance notes
-----------------
``V`` is a fixed pattern for a given optical calibration and frame shape, so full-frame
maps are built once, normalised to unit mean, stored as ``float32`` and reused by every
subsequent image of the same band and size.  The cache is bounded by
:data:`_VIGNETTING_CACHE_SIZE` maps and can be dropped with :func:`clear_caches`.
:meth:`rededgep.capture.Capture.reflectance` additionally keeps the full-frame chain in
``float32``, which is ample precision for reflectance values in 0..1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from .metadata import ImageMeta

__all__ = [
    "vignetting",
    "vignetting_mean",
    "dn_to_radiance",
    "dn_to_reflectance",
    "reflectance_scale",
    "clear_caches",
]

#: Number of full-frame vignetting maps kept in memory.  One RedEdge-P map is
#: 1088 * 1456 * 4 B ~= 6.3 MB, so eight entries cap the cache at roughly 50 MB:
#: enough for all bands of one capture at a couple of frame sizes.
_VIGNETTING_CACHE_SIZE = 8

#: Clip bound for the exponential.  Metashape's exponent stays around -1, but the
#: bound keeps pathological calibrations from overflowing.
_EXPONENT_CLIP = 50.0


@dataclass(frozen=True)
class _VignettingMap:
    """A cached full-frame vignetting map.

    ``values`` is ``V / mean(V)`` as a read-only ``float32`` array of shape
    ``(height, width)`` -- exactly the quantity the chain divides DN by.  ``mean`` is
    the raw (unnormalised) float64 mean over the frame, as returned by
    :func:`vignetting_mean`.
    """

    values: np.ndarray
    mean: float


def _coeff_key(vignetting: Sequence[tuple[int, int, float]]) -> tuple:
    """Hashable, dtype-stable key for a vignetting coefficient sequence."""
    return tuple((int(i), int(j), float(c)) for i, j, c in vignetting)


def _power_table(base: np.ndarray, exponents: Sequence[int]) -> dict[int, np.ndarray]:
    """Return ``{e: base ** e}`` for the distinct exponents in ``exponents``.

    NumPy's integer ``**`` goes through the generic ``pow`` ufunc, which is much slower
    than a multiplication.  Powers are computed cumulatively once and shared by every
    polynomial term that needs them.  Negative exponents (not produced by the RedEdge-P
    calibrations, but valid in the Metashape model) fall back to ``np.power``.
    """
    table: dict[int, np.ndarray] = {}
    maximum = -1
    for exponent in exponents:
        if exponent >= 0:
            maximum = max(maximum, exponent)
        else:
            table[exponent] = np.power(base, exponent)
    if maximum >= 0:
        power = np.ones_like(base)
        table[0] = power
        for exponent in range(1, maximum + 1):
            power = power * base
            table[exponent] = power
    return table


def _vignetting_exponent(
    x: np.ndarray,
    y: np.ndarray,
    coefficients: tuple,
    *,
    grid: bool = False,
) -> np.ndarray:
    """Evaluate ``sum c_ij * x**i * y**j`` with a shared power table.

    ``grid=True`` is the full-frame layout: ``x`` is a 1-D vector of column coordinates
    (length ``width``) and ``y`` a 1-D vector of row coordinates (length ``height``), so
    the result is the ``(height, width)`` outer grid built without materialising
    ``meshgrid`` arrays.  With ``grid=False`` ordinary NumPy broadcasting applies, which
    is what the public :func:`vignetting` uses for arbitrary sample coordinates.

    Coefficients are grouped by ``i``, so each distinct power of ``x`` costs one
    full-array pass rather than one pass per polynomial term.
    """
    x_powers = _power_table(x, [i for i, _, _ in coefficients])
    y_powers = _power_table(y, [j for _, j, _ in coefficients])

    groups: dict[int, list[tuple[int, float]]] = {}
    for i, j, coefficient in coefficients:
        groups.setdefault(i, []).append((j, coefficient))

    exponent = None
    for i, terms in groups.items():
        # 1-D polynomial in y for this power of x.
        j, coefficient = terms[0]
        poly = y_powers[j] * coefficient
        for j, coefficient in terms[1:]:
            poly = poly + y_powers[j] * coefficient
        if grid:
            contribution = x_powers[i][None, :] * poly[:, None]
        else:
            contribution = x_powers[i] * poly
        exponent = contribution if exponent is None else exponent + contribution

    if exponent is None:  # no coefficients: V == 1 everywhere
        shape = np.broadcast_shapes(np.shape(x), np.shape(y))
        exponent = np.zeros(shape, dtype=np.float64)
    # NEP 50 folds 0-d array arithmetic to NumPy scalars; keep returning an ndarray so
    # callers (and the in-place clip/exp below) see the same type as before.
    return np.asarray(exponent, dtype=np.float64)


@lru_cache(maxsize=_VIGNETTING_CACHE_SIZE)
def _vignetting_map(coefficients: tuple, width: int, height: int) -> _VignettingMap:
    """Build (and cache) the unit-mean vignetting map of one frame shape.

    The polynomial is evaluated in float64 on the two 1-D coordinate vectors and only
    the finished map is narrowed to float32.  ``lru_cache`` is thread-safe; two threads
    racing on the same new key may both build the map, but they store identical values.
    """
    x = 2.0 * (np.arange(width, dtype=np.float64) + 0.5) / width - 1.0
    y = 2.0 * (np.arange(height, dtype=np.float64) + 0.5) / height - 1.0
    values = _vignetting_exponent(x, y, coefficients, grid=True)
    np.clip(values, -_EXPONENT_CLIP, _EXPONENT_CLIP, out=values)
    np.exp(values, out=values)

    mean = float(values.mean())
    normalised = (values / mean).astype(np.float32)
    normalised.flags.writeable = False
    return _VignettingMap(values=normalised, mean=mean)


def clear_caches() -> None:
    """Drop every cached full-frame vignetting map.

    Call this after changing a calibration, or to release the memory held by maps for
    frame shapes that are no longer in use.
    """
    _vignetting_map.cache_clear()


def vignetting(
    vignetting: Sequence[tuple[int, int, float]],
    u: np.ndarray,
    v: np.ndarray,
    width: int = 1456,
    height: int = 1088,
) -> np.ndarray:
    """Evaluate ``V(x, y) = exp(sum c_ij x^i y^j)`` at pixel coordinates ``u, v``.

    ``u``/``v`` may be scalars or broadcast against each other; the result has their
    broadcast shape and is ``float64``.  For full images, prefer
    :func:`dn_to_reflectance` or :meth:`~rededgep.capture.Capture.reflectance`, which
    reuse a cached full-frame map.
    """
    x = 2.0 * (np.asarray(u, dtype=np.float64) + 0.5) / width - 1.0
    y = 2.0 * (np.asarray(v, dtype=np.float64) + 0.5) / height - 1.0
    exponent = _vignetting_exponent(x, y, _coeff_key(vignetting))
    np.clip(exponent, -_EXPONENT_CLIP, _EXPONENT_CLIP, out=exponent)
    return np.exp(exponent, out=exponent)


def vignetting_mean(
    vignetting: Sequence[tuple[int, int, float]],
    width: int = 1456,
    height: int = 1088,
) -> float:
    """Mean of :func:`vignetting` over the full frame (cached; constant per sensor)."""
    return _vignetting_map(_coeff_key(vignetting), width, height).mean


def reflectance_scale(meta: ImageMeta) -> float:
    """Per-image factor that converts corrected DN to radiance."""
    return meta.a1 / (meta.gain * meta.exposure_s * 2.0**16)


def _require_irradiance(meta: ImageMeta) -> None:
    if not meta.has_irradiance:
        raise ValueError(
            f"{meta.path} has no DLS horizontal irradiance; reflectance cannot be computed. "
            "Use a reflectance panel instead, or supply the irradiance explicitly."
        )


def _is_full_frame_grid(u: np.ndarray, v: np.ndarray, width: int, height: int) -> bool:
    """True when ``u, v`` are exactly the integer pixel grid of a ``height x width`` frame."""
    if u.shape != (height, width) or v.shape != (height, width):
        return False
    columns = np.broadcast_to(np.arange(width), (height, width))
    rows = np.broadcast_to(np.arange(height)[:, None], (height, width))
    return bool(np.array_equal(u, columns) and np.array_equal(v, rows))


def _full_frame_values(coefficients: tuple, width: int, height: int) -> np.ndarray:
    """Unit-mean float32 vignetting map for one calibration and frame shape."""
    return _vignetting_map(coefficients, width, height).values


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
        u_array = np.asarray(u)
        v_array = np.asarray(v)
        if corrected.shape == (sensor.height, sensor.width) and _is_full_frame_grid(
            u_array, v_array, sensor.width, sensor.height
        ):
            # Full image on the sensor grid: reuse the cached, already normalised map.
            corrected /= _full_frame_values(_coeff_key(sensor.vignetting), sensor.width, sensor.height)
        else:
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
    _require_irradiance(meta)
    radiance = dn_to_radiance(dn, meta, sensor, u, v)
    return np.pi * radiance / meta.horizontal_irradiance


def _dn_to_reflectance_full_frame(dn: np.ndarray, meta: ImageMeta, sensor) -> np.ndarray:
    """Fast float32 whole-frame reflectance, using the cached vignetting map.

    This is the private fast path behind :meth:`rededgep.capture.Capture.reflectance`.
    It returns the same values as ``dn_to_reflectance(...).astype(np.float32)`` to
    within a few float32 ulp (see the equivalence check in the benchmark notes).  If
    the DN array does not cover the sensor frame the historical meshgrid path is used,
    so odd shape/metadata combinations keep their old behaviour.
    """
    _require_irradiance(meta)
    if np.shape(dn) != (sensor.height, sensor.width):
        rows, cols = np.shape(dn)
        u, v = np.meshgrid(np.arange(cols, dtype=np.float64), np.arange(rows, dtype=np.float64))
        return dn_to_reflectance(dn, meta, sensor, u, v).astype(np.float32)

    values = _full_frame_values(_coeff_key(sensor.vignetting), sensor.width, sensor.height)

    # One fused uint16 -> float32 conversion and black-level subtraction; the
    # intermediate never exists in float64.
    corrected = np.empty(np.shape(dn), dtype=np.float32)
    np.subtract(dn, meta.black_level, out=corrected, dtype=np.float32)
    np.maximum(corrected, 0.0, out=corrected)

    # Divide by V / mean(V), then apply a1/(gain*exposure*2**16) * pi / E_horizontal.
    np.divide(corrected, values, out=corrected)
    corrected *= np.float32(np.pi * reflectance_scale(meta) / meta.horizontal_irradiance)
    return corrected
