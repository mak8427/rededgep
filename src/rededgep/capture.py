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

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import Calibration, Sensor
from .chain import _dn_to_reflectance_full_frame
from .metadata import BANDS, SUFFIX_TO_BAND, ImageMeta, read_metadata, MetadataError
from .readers import read_dn

__all__ = ["Capture", "DEFAULT_WORKERS", "set_worker_count"]

#: Canonical order used for exported stacks.
STACK_ORDER = ("Blue", "Green", "Red", "Red edge", "NIR")

#: Environment variable that overrides the default worker count.
_WORKERS_ENV = "REDEDGEP_WORKERS"

#: Default number of threads used to process the bands of a capture.  NumPy releases
#: the GIL for the large ufuncs in the chain, and I/O is already done by the time
#: reflectance runs, so a handful of threads is enough to overlap the memory traffic
#: without oversubscribing the machine.
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))

#: Module-level override set by :func:`set_worker_count`; ``None`` means "auto".
_WORKERS: int | None = None


def set_worker_count(count: int | None) -> None:
    """Set how many bands :meth:`Capture.reflectance` processes in parallel.

    Parameters
    ----------
    count
        Number of worker threads (``>= 1``), or ``None`` to restore the default:
        ``min(4, os.cpu_count())``, overridable with the ``REDEDGEP_WORKERS``
        environment variable.  ``1`` disables parallelism.

    Raises
    ------
    ValueError
        If ``count`` is less than one.
    """
    global _WORKERS
    if count is None:
        _WORKERS = None
        return
    count = int(count)
    if count < 1:
        raise ValueError(f"worker count must be at least 1, got {count}")
    _WORKERS = count


def _worker_count() -> int:
    """Resolve the configured worker count (explicit setting, env var, default)."""
    if _WORKERS is not None:
        return _WORKERS
    raw = os.environ.get(_WORKERS_ENV)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return DEFAULT_WORKERS


def _load_band(job: tuple[str, Path]) -> tuple[str, np.ndarray | None, ImageMeta | None]:
    """Read one band image and its metadata; safe to call from a thread."""
    band, path = job
    try:
        array = read_dn(path)
    except Exception:
        return band, None, None
    try:
        info = read_metadata(path)
    except MetadataError:
        info = None
    return band, array, info


def _compute_band(job: tuple[str, np.ndarray, ImageMeta, Sensor]) -> np.ndarray:
    """Reflectance for one band; kept at module level so threads can call it."""
    _, dn, meta, sensor = job
    return _dn_to_reflectance_full_frame(dn, meta, sensor)


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
    def from_directory(
        cls,
        directory: str | Path,
        name: str = "IMG_0000",
        *,
        workers: int | None = None,
    ) -> "Capture":
        """Load a capture given the directory holding ``IMG_<id>_1..6.tif``.

        The six bands are read in parallel: decoding releases the GIL, and reading is
        what dominates the wall clock -- a single band costs ~8 ms from disk against
        0.24 ms of reflectance maths.

        Parameters
        ----------
        directory
            Folder holding the images.
        name
            Capture stem, e.g. ``IMG_0000``.
        workers
            Thread count; defaults to the configured worker count (see
            :func:`set_worker_count`, ``REDEDGEP_WORKERS``).

        Raises
        ------
        FileNotFoundError
            If the directory holds no images for ``name``.
        """
        directory = Path(directory)
        jobs = []
        for suffix, band in SUFFIX_TO_BAND.items():
            path = directory / f"{name}_{suffix}.tif"
            if path.is_file():
                jobs.append((band, path))
        if not jobs:
            raise FileNotFoundError(f"no {name}_*.tif in {directory}")

        count = _worker_count() if workers is None else max(1, int(workers))
        if count <= 1 or len(jobs) == 1:
            loaded = [_load_band(job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=count, thread_name_prefix="rededgep-io") as pool:
                loaded = list(pool.map(_load_band, jobs))

        dn: dict[str, np.ndarray] = {}
        meta: dict[str, ImageMeta] = {}
        for band, array, info in loaded:
            if array is None:
                continue
            dn[band] = array
            if info is not None:
                meta[band] = info
        return cls(name=name, dn=dn, meta=meta)

    def reflectance(self, calibration: Calibration | None = None) -> dict[str, np.ndarray]:
        """Compute reflectance for every band.

        The result is a full-frame reflectance image per band.  The vignetting map for
        a band and frame size is built once and reused, and the bands are independent,
        so they are processed in a small thread pool (see :func:`set_worker_count`).
        The returned dictionary keeps the band order of :attr:`dn`.
        """
        calibration = calibration or Calibration.rededgep()
        jobs: list[tuple[str, np.ndarray, ImageMeta, Sensor]] = []
        for band, dn in self.dn.items():
            meta = self.meta.get(band)
            if meta is None:
                continue
            sensor = calibration.sensor(band).scaled(width=meta.width, height=meta.height)
            jobs.append((band, dn, meta, sensor))
        if not jobs:
            return {}

        workers = min(_worker_count(), len(jobs))
        if workers == 1:
            results = [_compute_band(job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rededgep") as pool:
                results = list(pool.map(_compute_band, jobs))
        return dict(zip((job[0] for job in jobs), results))

    def stack(self, calibration: Calibration | None = None) -> np.ndarray:
        """Reflectance as a ``(5, rows, cols)`` float32 array in canonical band order."""
        reflectance = self.reflectance(calibration)
        missing = [b for b in STACK_ORDER if b not in reflectance]
        if missing:
            raise ValueError(f"capture is missing bands: {missing}")
        return np.stack([reflectance[b] for b in STACK_ORDER])
