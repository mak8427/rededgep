"""Image readers with a graceful fallback.

GeoTIFF and large rasters are read with GDAL when it is installed; plain TIFFs fall back
to Pillow so the core package needs only numpy and pillow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["read_dn", "read_dn_window", "read_window", "read_raster", "gdal_available"]


def gdal_available() -> bool:
    try:
        from osgeo import gdal  # noqa: F401

        return True
    except Exception:
        return False


def read_dn(path: str | Path) -> np.ndarray:
    """Read a single-band raw DN image as ``uint16``."""
    try:
        from osgeo import gdal

        gdal.UseExceptions()
        dataset = gdal.Open(str(path))
        if dataset is not None:
            return dataset.GetRasterBand(1).ReadAsArray()
    except Exception:
        pass
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image)


def read_window(path: str | Path, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Read the half-open pixel window ``[x0, x1) x [y0, y1)`` as ``uint16``.

    Only the requested tiles are read when GDAL is available, which is what makes
    sampling a handful of plots from a 1456x1088 frame cheap.  Without GDAL the whole
    image is decoded and cropped, so the result is still correct, just not faster.

    Parameters
    ----------
    x0, y0
        Top-left column and row of the window (inclusive).
    x1, y1
        Bottom-right column and row of the window (exclusive).

    Raises
    ------
    ValueError
        If the window is empty or lies outside the image.
    """
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty window ({x0}, {y0}, {x1}, {y1})")
    try:
        from osgeo import gdal

        gdal.UseExceptions()
        dataset = gdal.Open(str(path))
        if dataset is not None:
            width, height = dataset.RasterXSize, dataset.RasterYSize
            if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                raise ValueError(
                    f"window ({x0}, {y0}, {x1}, {y1}) outside image {width}x{height}"
                )
            return dataset.GetRasterBand(1).ReadAsArray(x0, y0, x1 - x0, y1 - y0)
    except ValueError:
        raise
    except Exception:
        pass
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.crop((x0, y0, x1, y1)))


def read_dn_window(path: str | Path, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Alias of :func:`read_window` kept for symmetry with :func:`read_dn`."""
    return read_window(path, x0, y0, x1, y1)


def read_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read a georeferenced raster and its geotransform.

    Returns
    -------
    (array, info)
        ``array`` is ``(bands, rows, cols)``, ``info`` holds ``transform``, ``crs``,
        ``width``, ``height`` and ``nodata``.
    """
    from osgeo import gdal

    gdal.UseExceptions()
    dataset = gdal.Open(str(path))
    if dataset is None:
        raise FileNotFoundError(path)
    array = dataset.ReadAsArray()
    if array.ndim == 2:
        array = array[None, ...]
    info = {
        "transform": dataset.GetGeoTransform(),
        "crs": dataset.GetProjection(),
        "width": dataset.RasterXSize,
        "height": dataset.RasterYSize,
        "nodata": dataset.GetRasterBand(1).GetNoDataValue(),
    }
    return array, info
