"""Optional C acceleration.

The C kernel fuses projection, distortion, bilinear sampling, vignetting and the
radiometric conversion into one pass.  It is optional: if the shared library is absent
the numpy implementation is used and results are identical to floating-point rounding.

Build it with::

    python -m rededgep.native --build

or simply ``make native`` from a source checkout.
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


def _build_command(out: Path) -> list[str]:
    return ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", str(out), str(_SOURCE), "-lm"]


def build(force: bool = False, out: Path | None = None) -> Path:
    """Compile the C kernel with the system compiler and return the library path."""
    out = Path(out) if out is not None else _LIBRARY
    if out.is_file() and not force:
        return out
    if not _SOURCE.is_file():
        raise FileNotFoundError(f"C source missing: {_SOURCE}")
    command = _build_command(out)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed to compile native kernel:\n{result.stderr}")
    return out


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
        lib.rp_sample_reflectance.restype = ctypes.c_long
        lib.rp_sample_reflectance.argtypes = [
            ctypes.POINTER(ctypes.c_ushort), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.c_long,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int), ctypes.c_long,
            ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_ubyte),
        ]
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
    """Invoke the C kernel directly.  Prefer :func:`rededgep.sampling.sample_reflectance`."""
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


def _main(argv: list[str]) -> int:
    force = "--force" in argv
    path = build(force=force)
    print(f"built {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
