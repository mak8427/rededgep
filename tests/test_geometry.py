"""Camera geometry: nadir behaviour, finiteness and distortion."""

from __future__ import annotations

import unittest

import numpy as np

from conftest import make_calibration, make_camera

from rededgep.geometry import CameraPose, enu_basis, lla_to_ecef, project


class TestProjection(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor = make_calibration().sensor("Green")
        self.camera = make_camera()

    def test_nadir_point_lands_on_principal_point(self) -> None:
        # Identity rotation: a point directly below the camera is on the optical axis.
        u, v = project(self.camera, self.sensor, self.camera.lon, self.camera.lat, self.camera.alt - 20.0)
        self.assertAlmostEqual(float(u), self.sensor.width / 2.0 + self.sensor.cx, places=3)
        self.assertAlmostEqual(float(v), self.sensor.height / 2.0 + self.sensor.cy, places=3)

    def test_shifting_north_moves_up_in_image(self) -> None:
        u0, v0 = project(self.camera, self.sensor, self.camera.lon, self.camera.lat, self.camera.alt - 20.0)
        u1, v1 = project(self.camera, self.sensor, self.camera.lon, self.camera.lat + 2e-5, self.camera.alt - 20.0)
        self.assertLess(float(v1), float(v0))  # image y is flipped

    def test_shifting_east_moves_right(self) -> None:
        u0, _ = project(self.camera, self.sensor, self.camera.lon, self.camera.lat, self.camera.alt - 20.0)
        u1, _ = project(self.camera, self.sensor, self.camera.lon + 2e-5, self.camera.lat, self.camera.alt - 20.0)
        self.assertGreater(float(u1), float(u0))

    def test_point_above_camera_is_invalid(self) -> None:
        u, v = project(self.camera, self.sensor, self.camera.lon, self.camera.lat, self.camera.alt + 5.0)
        self.assertTrue(np.isnan(u))
        self.assertTrue(np.isnan(v))

    def test_rotation_shape_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            CameraPose(lon=0, lat=0, alt=0, rotation=np.eye(2))


class TestLocalFrames(unittest.TestCase):
    def test_enu_basis_is_orthonormal(self) -> None:
        basis = enu_basis(9.9, 51.5)
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(basis)), 1.0, places=12)

    def test_ecef_magnitude_is_earth_radius_scale(self) -> None:
        vector = lla_to_ecef(0.0, 0.0, 0.0)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 6378137.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
