"""Per-image radiometric metadata for MicaSense RedEdge-P captures.

Everything is read from the image itself: EXIF for gain/exposure/black level and the
XMP packet for the radiometric calibration coefficients and the downwelling light
sensor (DLS) readings.  No Agisoft Metashape project is required.

The XMP packet is stored in TIFF tag 700, which Pillow exposes directly, so reading
metadata needs only ``pillow``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

BANDS = ("Blue", "Green", "Red", "Red edge", "NIR", "Panchro")
#: Raw RedEdge-P file suffix (IMG_<id>_<n>.tif) to band name.
SUFFIX_TO_BAND = {1: "Blue", 2: "Green", 3: "Red", 4: "NIR", 5: "Red edge", 6: "Panchro"}
#: DLS2 irradiance tags are reported in units that need this factor for W/m^2/nm.
DLS_SCALE_TO_SI = 0.01

_EXIF_IFD = 34665
_ISO_TAGS = (34867, 34855)  # ISOSpeed, ISOSpeedRatings
_EXPOSURE_TAG = 33434
_BLACK_LEVEL_TAG = 50714
_XMP_TAG = 700


class MetadataError(RuntimeError):
    """Raised when an image does not carry the metadata required for calibration."""


@dataclass(frozen=True)
class ImageMeta:
    """Radiometric metadata for a single RedEdge-P image."""

    path: str
    band: str
    iso: float
    exposure_s: float
    black_level: float
    a1: float
    a2: float
    a3: float
    horizontal_irradiance: float
    spectral_irradiance: float
    solar_elevation_rad: float
    solar_azimuth_rad: float
    width: int
    height: int

    @property
    def gain(self) -> float:
        """Sensor gain, ``ISO / 100``."""
        return self.iso / 100.0

    @property
    def has_irradiance(self) -> bool:
        return self.horizontal_irradiance > 0.0


def _xmp_packet(image) -> str:
    raw = image.tag_v2.get(_XMP_TAG)
    if raw is None:
        return ""
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", "ignore")
    return str(raw)


def _find(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _float(value, default: float = float("nan")) -> float:
    """Coerce an XMP string, a number, or a ``(numerator, denominator)`` rational."""
    if value is None:
        return default
    if isinstance(value, tuple):
        if len(value) == 2 and value[1]:
            return float(value[0]) / float(value[1])
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class XmpFields:
    """The radiometric fields we take from the XMP packet."""

    band: str
    a1: float
    a2: float
    a3: float
    horizontal_irradiance: float
    spectral_irradiance: float
    solar_elevation_rad: float
    solar_azimuth_rad: float


def parse_xmp(xmp: str) -> XmpFields:
    """Parse the radiometric fields out of an XMP packet.

    Raises
    ------
    MetadataError
        If the band name or the radiometric calibration block is missing.
    """
    band = _find(r"<Camera:BandName>([^<]+)</Camera:BandName>", xmp)
    if band is None:
        raise MetadataError("no Camera:BandName in XMP")
    block = re.search(r"<MicaSense:RadiometricCalibration>(.*?)</MicaSense:RadiometricCalibration>", xmp, re.S)
    if block is None:
        raise MetadataError("no RadiometricCalibration block in XMP")
    values = [_float(v) for v in re.findall(r"<rdf:li>([-0-9.eE+]+)</rdf:li>", block.group(1))]
    if len(values) < 3:
        raise MetadataError("incomplete RadiometricCalibration block")
    return XmpFields(
        band=band.strip(),
        a1=values[0],
        a2=values[1],
        a3=values[2],
        horizontal_irradiance=_float(_find(r"<DLS:HorizontalIrradiance>([-0-9.eE+]+)", xmp), 0.0) * DLS_SCALE_TO_SI,
        spectral_irradiance=_float(_find(r"<DLS:SpectralIrradiance>([-0-9.eE+]+)", xmp), 0.0) * DLS_SCALE_TO_SI,
        solar_elevation_rad=_float(_find(r"<DLS:SolarElevation>([-0-9.eE+]+)", xmp), 0.0),
        solar_azimuth_rad=_float(_find(r"<DLS:SolarAzimuth>([-0-9.eE+]+)", xmp), 0.0),
    )


def read_metadata(path: str | Path) -> ImageMeta:
    """Read radiometric metadata from a MicaSense RedEdge-P TIFF.

    Parameters
    ----------
    path
        Path to one band of one capture, e.g. ``IMG_0000_2.tif``.

    Raises
    ------
    MetadataError
        If the file has no XMP packet or lacks the band name / calibration block.
    """
    from PIL import Image

    path = Path(path)
    with Image.open(path) as image:
        exif = image.getexif()
        exif_ifd = exif.get_ifd(_EXIF_IFD)
        xmp = _xmp_packet(image)
        width, height = image.size

    if not xmp:
        raise MetadataError(f"no XMP packet in {path}")
    try:
        fields = parse_xmp(xmp)
    except MetadataError as error:
        raise MetadataError(f"{error} in {path}") from None

    iso = float("nan")
    for tag in _ISO_TAGS:
        value = exif_ifd.get(tag) or exif.get(tag)
        if value:
            iso = float(value)
            break
    if not (iso == iso) or iso <= 0:
        raise MetadataError(f"no ISO tag in {path}")

    # Some writers put these in the Exif sub-IFD, others in the top-level IFD; accept
    # either, which is what the ISO lookup already did.
    exposure = _float(exif_ifd.get(_EXPOSURE_TAG) or exif.get(_EXPOSURE_TAG))
    if not (exposure == exposure) or exposure <= 0:
        raise MetadataError(f"no exposure time in {path}")

    black = exif_ifd.get(_BLACK_LEVEL_TAG) or exif.get(_BLACK_LEVEL_TAG)
    if isinstance(black, (tuple, list)) and black:
        black = sum(float(v) for v in black) / len(black)
    black_level = _float(black, 0.0)

    return ImageMeta(
        path=str(path),
        band=fields.band,
        iso=iso,
        exposure_s=exposure,
        black_level=black_level,
        a1=fields.a1,
        a2=fields.a2,
        a3=fields.a3,
        horizontal_irradiance=fields.horizontal_irradiance,
        spectral_irradiance=fields.spectral_irradiance,
        solar_elevation_rad=fields.solar_elevation_rad,
        solar_azimuth_rad=fields.solar_azimuth_rad,
        width=width,
        height=height,
    )
