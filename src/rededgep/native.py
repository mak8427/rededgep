"""Optional C acceleration.

The C kernel fuses the geodetic conversion, projection, distortion, bilinear sampling,
vignetting and the radiometric conversion into one pass.  It is optional: if the shared
library is absent the numpy implementation is used and results agree to floating-point
rounding.

Build it with::

    python -m rededgep.native --build

or simply ``make native`` from a source checkout.

If the compiler supports OpenMP the kernel is built with it; ``REDEDGEP_THREADS`` then
sets the number of threads used for a call.  The default is one thread, which keeps
results bit-for-bit deterministic.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_LIBRARY = _HERE / "librededgep.so"
_SOURCE = _HERE / "_native.c"

_lib = None
_load_attempted = False

__all__ = ["available", "build", "library"]


def _build_command(out: Path, openmp: bool = True) -> list[str]:
    # -fno-math-errno lets sqrt() compile to a single instruction.  It does not change
    # any result (unlike -ffast-math, which is deliberately not used).
    command = ["gcc", "-O3", "-march=native", "-funroll-loops", "-fno-math-errno"]
    if openmp:
        command.append("-fopenmp")  # optional; build() retries without it
    command += ["-shared", "-fPIC", "-o", str(out), str(_SOURCE), "-lm"]
    return command


def build(force: bool = False, out: Path | None = None) -> Path:
    """Compile the C kernel with the system compiler and return the library path."""
    out = Path(out) if out is not None else _LIBRARY
    if out.is_file() and not force:
        return out
    if not _SOURCE.is_file():
        raise FileNotFoundError(f"C source missing: {_SOURCE}")
    result = subprocess.run(_build_command(out), capture_output=True, text=True)
    if result.returncode != 0:
        # Compilers without libgomp (e.g. clang without libomp) still get a serial build.
        result = subprocess.run(_build_command(out, openmp=False), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed to compile native kernel:\n{result.stderr}")
    return out


def _declare(lib: ctypes.CDLL) -> None:
    double = ctypes.POINTER(ctypes.c_double)
    lib.rp_sample_reflectance.restype = ctypes.c_long
    lib.rp_sample_reflectance.argtypes = [
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int, ctypes.c_int,
        double, double, double, ctypes.c_long,
        double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        double, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_long,
        ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ubyte),
    ]
    if not hasattr(lib, "rp_sample_reflectance_geodetic"):
        return
    lib.rp_sample_reflectance_geodetic.restype = ctypes.c_long
    lib.rp_sample_reflectance_geodetic.argtypes = [
        ctypes.POINTER(ctypes.c_ushort), ctypes.c_int, ctypes.c_int,
        double, double, double, ctypes.c_long,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        double, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_long,
        ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        double,
    ]


def library():
    """Return the loaded ``ctypes`` library, or ``None`` if unavailable."""
    global _lib, _load_attempted
    if _load_attempted:
        return _lib
    _load_attempted = True
    if not _LIBRARY.is_file():
        try:
            build()
        except Exception:
            return None
    try:
        lib = ctypes.CDLL(str(_LIBRARY))
        _declare(lib)
        _lib = lib
        return _lib
    except Exception:
        _lib = None
        return None


def available() -> bool:
    """True when the C kernel is loaded and usable."""
    return library() is not None


def _ptr(array, dtype):
    return np.ascontiguousarray(array, dtype=dtype).ctypes.data_as(ctypes.POINTER(dtype))


def call_kernel(
    dn: np.ndarray,
    east: np.ndarray,
    north: np.ndarray,
    up: np.ndarray,
    rotation: np.ndarray,
    focal_px: float,
    cx: float,
    cy: float,
    distortion: tuple[float, float, float, float, float],
    vignetting: tuple[tuple[int, int, float], ...],
    vignetting_mean: float,
    black_level: float,
    scale: float,
    irradiance: float,
) -> np.ndarray:
    """Invoke the C kernel with points already in local ENU.  Prefer :func:`rededgep.sampling.sample_reflectance`."""
    lib = library()
    if lib is None:
        raise RuntimeError("native kernel not available")
    dn = np.ascontiguousarray(dn, dtype=np.uint16)
    height, width = dn.shape
    n = east.size
    out = np.empty(n, dtype=np.float32)
    valid = np.zeros(n, dtype=np.ubyte)
    k1, k2, k3, p1, p2 = distortion
    vig_c = np.array([c for _, _, c in vignetting], dtype=np.float64)
    vig_i = np.array([i for i, _, _ in vignetting], dtype=np.int32)
    vig_j = np.array([j for _, j, _ in vignetting], dtype=np.int32)
    lib.rp_sample_reflectance(
        _ptr(dn, ctypes.c_ushort), height, width,
        _ptr(east, ctypes.c_double), _ptr(north, ctypes.c_double), _ptr(up, ctypes.c_double), n,
        _ptr(rotation, ctypes.c_double),
        focal_px, cx, cy,
        k1, k2, k3, p1, p2,
        _ptr(vig_c, ctypes.c_double), _ptr(vig_i, ctypes.c_int), _ptr(vig_j, ctypes.c_int),
        vignetting.__len__(), vignetting_mean,
        black_level, scale, irradiance,
        _ptr(out, ctypes.c_float), _ptr(valid, ctypes.c_ubyte),
    )
    return out


def call_kernel_geodetic(
    dn: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    height: np.ndarray,
    camera_lon: float,
    camera_lat: float,
    camera_alt: float,
    rotation: np.ndarray,
    focal_px: float,
    cx: float,
    cy: float,
    distortion: tuple[float, float, float, float, float],
    vignetting: tuple[tuple[int, int, float], ...],
    vignetting_mean: float,
    black_level: float,
    scale: float,
    irradiance: float,
) -> np.ndarray:
    """Invoke the C kernel on geodetic points, flattened, as float64.

    The kernel converts lon/lat/height to ECEF and rotates into the camera frame itself,
    so the caller does no per-point numpy work.  Prefer
    :func:`rededgep.sampling.sample_reflectance`.
    """
    lib = library()
    if lib is None or not hasattr(lib, "rp_sample_reflectance_geodetic"):
        raise RuntimeError("native kernel not available")
    dn = np.ascontiguousarray(dn, dtype=np.uint16)
    image_height, image_width = dn.shape
    lon = np.ascontiguousarray(lon, dtype=np.float64).ravel()
    lat = np.ascontiguousarray(lat, dtype=np.float64).ravel()
    height = np.ascontiguousarray(height, dtype=np.float64).ravel()
    n = lon.size
    if lat.size != n or height.size != n:
        raise ValueError("lon, lat and height must have the same size")
    out = np.empty(n, dtype=np.float64)
    k1, k2, k3, p1, p2 = distortion
    vig_c = np.array([c for _, _, c in vignetting], dtype=np.float64)
    vig_i = np.array([i for i, _, _ in vignetting], dtype=np.int32)
    vig_j = np.array([j for _, j, _ in vignetting], dtype=np.int32)
    lib.rp_sample_reflectance_geodetic(
        _ptr(dn, ctypes.c_ushort), image_height, image_width,
        _ptr(lon, ctypes.c_double), _ptr(lat, ctypes.c_double), _ptr(height, ctypes.c_double), n,
        camera_lon, camera_lat, camera_alt,
        _ptr(rotation, ctypes.c_double),
        focal_px, cx, cy,
        k1, k2, k3, p1, p2,
        _ptr(vig_c, ctypes.c_double), _ptr(vig_i, ctypes.c_int), _ptr(vig_j, ctypes.c_int),
        len(vignetting), vignetting_mean,
        black_level, scale, irradiance,
        _ptr(out, ctypes.c_double),
    )
    return out


def _main(argv: list[str]) -> int:
    force = "--force" in argv
    path = build(force=force)
    print(f"built {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
