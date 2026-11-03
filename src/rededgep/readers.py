"""Image readers with a graceful fallback.

GeoTIFF and large rasters are read with GDAL when it is installed; plain TIFFs fall back
to Pillow so the core package needs only numpy and pillow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["read_dn", "read_raster", "gdal_available"]


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
