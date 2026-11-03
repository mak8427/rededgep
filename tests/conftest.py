"""Shared fixtures.

The suite is written for the standard library ``unittest`` runner so it needs no test
dependencies, but pytest can run it too.  Everything here is synthetic: no field data is
required, so the tests run anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rededgep.calibration import Calibration  # noqa: E402
from rededgep.geometry import CameraPose  # noqa: E402
from rededgep.metadata import ImageMeta  # noqa: E402


def make_meta(band: str = "Green", **overrides) -> ImageMeta:
    values = dict(
        path="synthetic.tif",
        band=band,
        iso=100.0,
        exposure_s=0.0015,
        black_level=3840.0,
        a1=1.7e-4,
        a2=0.0,
        a3=0.0,
        horizontal_irradiance=1.5,
        spectral_irradiance=1.52,
        solar_elevation_rad=1.0,
        solar_azimuth_rad=3.0,
        width=1456,
        height=1088,
    )
    values.update(overrides)
    return ImageMeta(**values)


def make_camera(lon: float = 9.9146, lat: float = 51.5628, alt: float = 207.0) -> CameraPose:
    """A synthetic nadir camera: ENU up maps to camera -Z, so depth is positive below."""
    return CameraPose(lon=lon, lat=lat, alt=alt, rotation=np.eye(3), label="synthetic")


def make_calibration() -> Calibration:
    return Calibration.rededgep()


_XMP_TEMPLATE = """<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
 <rdf:Description xmlns:Camera="http://pix4d.com/camera/1.0">
  <Camera:BandName>{band}</Camera:BandName>
 </rdf:Description>
 <rdf:Description xmlns:MicaSense="http://micasense.com/MicaSense/1.0/">
  <MicaSense:RadiometricCalibration><rdf:Seq>
   <rdf:li>0.0001703973</rdf:li><rdf:li>0.0</rdf:li><rdf:li>9.020843e-05</rdf:li>
  </rdf:Seq></MicaSense:RadiometricCalibration>
 </rdf:Description>
 <rdf:Description xmlns:DLS="http://micasense.com/DLS/1.0/">
  <DLS:HorizontalIrradiance>154.25</DLS:HorizontalIrradiance>
  <DLS:SpectralIrradiance>156.25</DLS:SpectralIrradiance>
 </rdf:Description>
</rdf:RDF></x:xmpmeta>"""


def write_band(path, band="Green", shape=(64, 96), value=9000, texture=False):
    """Write a synthetic RedEdge-P band TIFF carrying XMP and EXIF."""
    from PIL import Image

    if texture:
        rng = np.random.default_rng(abs(hash(band)) % 1000)
        array = rng.integers(4000, 40000, size=shape, dtype=np.uint16)
    else:
        array = np.full(shape, value, dtype=np.uint16)
    info = {
        700: _XMP_TEMPLATE.format(band=band).encode(),
        34867: 100,        # ISO
        33434: (1, 667),   # exposure, stored as a rational
        50714: 3840,       # black level
    }
    Image.fromarray(array).save(path, tiffinfo=info)
    return path


def write_capture(directory, name="IMG_0000", shape=(64, 96), bands=None):
    """Write a full six-band synthetic capture into ``directory``."""
    from pathlib import Path

    from rededgep.metadata import SUFFIX_TO_BAND

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for suffix, band in SUFFIX_TO_BAND.items():
        if bands is not None and band not in bands:
            continue
        write_band(directory / f"{name}_{suffix}.tif", band=band, shape=shape, texture=True)
    return directory
