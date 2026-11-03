"""rededgep — MicaSense RedEdge-P raw imagery to calibrated reflectance.

A small, dependency-light library that turns raw RedEdge-P captures into reflectance,
reproducing the chain Agisoft Metashape applies (see the README for the validation).

Quick start
-----------
>>> import rededgep as rp
>>> capture = rp.Capture.from_directory("flight/0004SET/000", "IMG_0000")
>>> reflectance = capture.reflectance()          # {band: float32 array}
>>> stack = capture.stack()                      # (5, rows, cols) in canonical order

Sample at ground points instead of a whole image:

>>> cameras = rp.load_cameras_txt("cameras.txt")
>>> rho = rp.sample_reflectance(dn, meta, sensor, cameras[0], lon, lat, height)

Get the sensor calibration from a Metashape project, or use the built-in one:

>>> calibration = rp.Calibration.from_psx("project.psx")
>>> calibration = rp.Calibration.rededgep()
"""

from __future__ import annotations

__version__ = "0.1.0"

from .calibration import Calibration, Sensor
from .capture import Capture, DEFAULT_WORKERS, STACK_ORDER, set_worker_count
from .chain import clear_caches, dn_to_radiance, dn_to_reflectance, reflectance_scale, vignetting, vignetting_mean
from .geometry import CameraPose, load_cameras_txt, project
from .metadata import BANDS, SUFFIX_TO_BAND, ImageMeta, MetadataError, read_metadata
from .readers import gdal_available, read_dn, read_raster
from .sampling import bilinear, sample_reflectance
from . import native

__all__ = [
    "__version__",
    "BANDS",
    "SUFFIX_TO_BAND",
    "STACK_ORDER",
    "DEFAULT_WORKERS",
    "Calibration",
    "Sensor",
    "Capture",
    "set_worker_count",
    "clear_caches",
    "ImageMeta",
    "MetadataError",
    "CameraPose",
    "read_metadata",
    "load_cameras_txt",
    "project",
    "dn_to_radiance",
    "dn_to_reflectance",
    "reflectance_scale",
    "vignetting",
    "vignetting_mean",
    "sample_reflectance",
    "bilinear",
    "read_dn",
    "read_raster",
    "gdal_available",
    "native",
]
