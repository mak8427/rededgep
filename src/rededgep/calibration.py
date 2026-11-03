"""Sensor calibration for a MicaSense RedEdge-P rig.

A :class:`Sensor` holds everything needed to turn raw DN into reflectance for one band:
the interior orientation, the distortion model, the black level, and the vignetting
polynomial.  A :class:`Calibration` is simply the six sensors of the rig.

Two ways to obtain one:

``Calibration.rededgep()``
    Built-in values extracted from Metashape projects; identical across every 2025
    project inspected.  Use when no Metashape project is at hand.

``Calibration.from_psx(path)``
    Read the values straight out of an Agisoft Metashape project file.
"""

from __future__ import annotations

import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

from ._defaults import SENSORS
from .metadata import BANDS

_CALIBRATION_KEYS = ("focal_px", "cx", "cy", "k1", "k2", "k3", "p1", "p2")


@dataclass(frozen=True)
class Sensor:
    """Radiometric and geometric calibration for one spectral band."""

    band: str
    focal_px: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    p1: float
    p2: float
    black_level: float
    sensitivity: float
    vignetting: tuple[tuple[int, int, float], ...]
    width: int = 1456
    height: int = 1088
    _mean_cache: list = field(default_factory=list, repr=False, compare=False)

    def vignetting_mean(self) -> float:
        """Mean of the vignetting polynomial over the full frame.

        Metashape applies ``V / mean(V)``, i.e. a correction normalised to unity mean.
        This value is constant per sensor, so it is computed once and cached.
        """
        if not self._mean_cache:
            import numpy as np

            from .chain import vignetting

            vv, uu = np.mgrid[0 : self.height, 0 : self.width]
            reference = vignetting(self.vignetting, uu.astype(float), vv.astype(float), self.width, self.height)
            self._mean_cache.append(float(reference.mean()))
        return self._mean_cache[0]

    def scaled(self, *, width: int, height: int) -> "Sensor":
        """Return a copy rescaled to another image size (principal point scales with it)."""
        if (width, height) == (self.width, self.height):
            return self
        sx = width / self.width
        sy = height / self.height
        return Sensor(
            band=self.band,
            focal_px=self.focal_px * 0.5 * (sx + sy),
            cx=self.cx * sx,
            cy=self.cy * sy,
            k1=self.k1,
            k2=self.k2,
            k3=self.k3,
            p1=self.p1,
            p2=self.p2,
            black_level=self.black_level,
            sensitivity=self.sensitivity,
            vignetting=self.vignetting,
            width=width,
            height=height,
        )


@dataclass(frozen=True)
class Calibration:
    """The six sensors of a RedEdge-P rig."""

    sensors: dict[str, Sensor]
    source: str = "builtin"

    def sensor(self, band: str) -> Sensor:
        try:
            return self.sensors[band]
        except KeyError:
            raise KeyError(f"no sensor for band {band!r}; available: {sorted(self.sensors)}") from None

    @property
    def bands(self) -> tuple[str, ...]:
        return tuple(b for b in BANDS if b in self.sensors)

    @classmethod
    def rededgep(cls) -> "Calibration":
        """Built-in RedEdge-P calibration (Metashape 2.2.1 values)."""
        sensors = {}
        for band, values in SENSORS.items():
            sensors[band] = Sensor(
                band=band,
                focal_px=values["focal_px"],
                cx=values["cx"],
                cy=values["cy"],
                k1=values["k1"],
                k2=values["k2"],
                k3=values["k3"],
                p1=values["p1"],
                p2=values["p2"],
                black_level=values["black_level"],
                sensitivity=values["sensitivity"],
                vignetting=tuple(values["vignetting"]),
            )
        return cls(sensors=sensors, source="builtin:rededge-p")

    @classmethod
    def from_psx(cls, path: str | Path) -> "Calibration":
        """Read sensor calibration from an Agisoft Metashape ``.psx`` project.

        A ``.psx`` written by Metashape 2.x is a small XML pointer to ``<name>.files/``,
        which holds ``project.zip`` and one ``chunk.zip`` per chunk.  No licence is
        needed: the calibration is plain XML inside those archives.
        """
        path = Path(path).expanduser().resolve()
        sidecar = path.with_suffix(".files")
        if not sidecar.is_dir():
            raise FileNotFoundError(f"{path} has no sidecar directory {sidecar.name}/")

        chunks: list[str] = []
        project = _read_zip_xml(sidecar / "project.zip")
        if project is not None:
            chunks = [c.get("path", "") for c in project.findall("chunks/chunk")]
        if not chunks:
            chunks = [p.relative_to(sidecar).as_posix() for p in sorted(sidecar.glob("*/chunk.zip"))]

        for relative in reversed(chunks):  # the last chunk is the active one
            root = _read_zip_xml(sidecar / relative)
            if root is None:
                continue
            sensors = _parse_sensors(root)
            if sensors:
                return cls(sensors=sensors, source=str(path))
        raise ValueError(f"no sensor calibration found in {path}")


def _read_zip_xml(path: Path) -> ET.Element | None:
    if not path.is_file():
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return ET.fromstring(archive.read("doc.xml" if "doc.xml" in names else names[0]))
    except (zipfile.BadZipFile, ET.ParseError, IndexError, OSError):
        return None


def _properties(element: ET.Element) -> dict[str, str]:
    return {p.get("name", ""): p.get("value", "") for p in element.iter("property")}


def _parse_sensors(chunk: ET.Element) -> dict[str, Sensor]:
    sensors: dict[str, Sensor] = {}
    for sensor in chunk.iter("sensor"):
        bands = [b.get("label") for b in sensor.findall("bands/band")]
        band = bands[0] if bands else sensor.get("label")
        if not band:
            continue
        adjusted = None
        for cal in sensor.findall("calibration"):
            if cal.get("class") == "adjusted":
                adjusted = cal
                break
        if adjusted is None:
            adjusted = sensor.find("calibration")
        if adjusted is None:
            continue
        values = {key: _to_float(adjusted.findtext(key)) for key in _CALIBRATION_KEYS}
        values["focal_px"] = values.pop("focal_px", None) or _to_float(adjusted.findtext("f"))
        resolution = sensor.find("resolution")
        width = int(resolution.get("width", 1456)) if resolution is not None else 1456
        height = int(resolution.get("height", 1088)) if resolution is not None else 1088
        vignetting = tuple(
            sorted(
                ((int(c.get("i", 0)), int(c.get("j", 0)), _to_float(c.text)) for c in sensor.findall("vignetting/coeff")),
                key=lambda item: (item[1], item[0]),
            )
        )
        sensors[band] = Sensor(
            band=band,
            focal_px=values["focal_px"],
            cx=values["cx"],
            cy=values["cy"],
            k1=values["k1"],
            k2=values["k2"],
            k3=values["k3"],
            p1=values["p1"],
            p2=values["p2"],
            black_level=_to_float(sensor.findtext("black_level")),
            sensitivity=_to_float(sensor.findtext("sensitivity")),
            vignetting=vignetting,
            width=width,
            height=height,
        )
    return sensors


def _to_float(text: str | None, default: float = float("nan")) -> float:
    try:
        return float(text) if text is not None else default
    except (TypeError, ValueError):
        return default
