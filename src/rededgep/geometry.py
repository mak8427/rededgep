"""Camera geometry: projecting ground points into a RedEdge-P image.

Camera poses are read from Metashape's ``cameras.txt`` export, which lists, per camera,
geodetic position and a 3x3 rotation.  The projection uses the Agisoft (Brown-Conrady)
model with the per-band adjusted intrinsics::

    u = w/2 + f * xd + cx
    v = h/2 - f * yd + cy

Note the minus sign on ``v``: Metashape's camera frame has y up, the image y down.
Using one sensor's intrinsics for every band misregisters by up to 45 px, so each band
must be projected with its own calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)

__all__ = ["CameraPose", "load_cameras_txt", "project", "lla_to_ecef", "enu_basis"]


@dataclass(frozen=True)
class CameraPose:
    """A camera position (geodetic) and orientation (ENU to camera)."""

    lon: float
    lat: float
    alt: float
    rotation: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
        object.__setattr__(self, "rotation", rotation)


def lla_to_ecef(lon, lat, height):
    """Geodetic coordinates to Earth-centred Earth-fixed."""
    lon = np.radians(lon)
    lat = np.radians(lat)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.stack(
        [
            (n + height) * np.cos(lat) * np.cos(lon),
            (n + height) * np.cos(lat) * np.sin(lon),
            (n * (1 - _E2) + height) * np.sin(lat),
        ],
        axis=-1,
    )


def enu_basis(lon, lat) -> np.ndarray:
    """Rotation matrix taking ECEF vectors to local east-north-up at ``lon, lat``."""
    lon = np.radians(lon)
    lat = np.radians(lat)
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


def load_cameras_txt(path: str | Path) -> list[CameraPose]:
    """Parse a Metashape ``cameras.txt`` export.

    Columns are ``PhotoID, X(lon), Y(lat), Z(alt), omega, phi, kappa, r11..r33``.
    """
    poses: list[CameraPose] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 16:
            continue
        poses.append(
            CameraPose(
                label=parts[0],
                lon=float(parts[1]),
                lat=float(parts[2]),
                alt=float(parts[3]),
                rotation=np.array(parts[7:16], dtype=np.float64).reshape(3, 3),
            )
        )
    return poses


def project(camera: CameraPose, sensor, lon, lat, height):
    """Project ground points into a camera.

    Returns ``(u, v)`` in pixels.  Points behind the camera produce ``nan``.
    """
    centre = lla_to_ecef(camera.lon, camera.lat, camera.alt)
    local = np.einsum("ij,...j->...i", enu_basis(camera.lon, camera.lat), lla_to_ecef(lon, lat, height) - centre)
    cam = np.einsum("ij,...j->...i", camera.rotation, local)

    depth = -cam[..., 2]
    behind = ~(depth > 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = cam[..., 0] / depth
        y = cam[..., 1] / depth

    r2 = x * x + y * y
    radial = 1.0 + sensor.k1 * r2 + sensor.k2 * r2**2 + sensor.k3 * r2**3
    xd = x * radial + 2.0 * sensor.p1 * x * y + sensor.p2 * (r2 + 2.0 * x * x)
    yd = y * radial + sensor.p1 * (r2 + 2.0 * y * y) + 2.0 * sensor.p2 * x * y

    u = sensor.width / 2.0 + sensor.focal_px * xd + sensor.cx
    v = sensor.height / 2.0 - sensor.focal_px * yd + sensor.cy
    invalid = ~np.isfinite(u) | ~np.isfinite(v) | behind
    u = np.where(invalid, np.nan, u)
    v = np.where(invalid, np.nan, v)
    return u, v
