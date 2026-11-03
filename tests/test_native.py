"""The optional C kernel must agree with the numpy implementation."""

from __future__ import annotations

import unittest

import numpy as np

from conftest import make_calibration, make_camera, make_meta

from rededgep import native
from rededgep.sampling import sample_reflectance

try:
    native.build()
    HAVE_NATIVE = native.available()
except Exception:  # no compiler, no library
    HAVE_NATIVE = False


@unittest.skipUnless(HAVE_NATIVE, "C kernel not available (needs gcc)")
class TestNativeMatchesNumpy(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor = make_calibration().sensor("Green")
        self.meta = make_meta()
        self.camera = make_camera()
        rng = np.random.default_rng(7)
        n = 3000
        self.lon = self.camera.lon + (rng.random(n) - 0.5) * 5e-5
        self.lat = self.camera.lat + (rng.random(n) - 0.5) * 4e-5
        self.height = np.full(n, self.camera.alt - 20.0)
        self.dn = np.full((1088, 1456), 9000, dtype=np.uint16)

    def test_flat_field_agreement(self) -> None:
        a = sample_reflectance(self.dn, self.meta, self.sensor, self.camera,
                               self.lon, self.lat, self.height, backend="numpy")
        b = sample_reflectance(self.dn, self.meta, self.sensor, self.camera,
                               self.lon, self.lat, self.height, backend="native")
        ok = np.isfinite(a) & np.isfinite(b)
        self.assertGreater(ok.sum(), 100)
        np.testing.assert_allclose(a[ok], b[ok], rtol=1e-6, atol=1e-9)

    def test_textured_field_agreement(self) -> None:
        rng = np.random.default_rng(11)
        dn = rng.integers(4000, 40000, size=(1088, 1456), dtype=np.uint16)
        a = sample_reflectance(dn, self.meta, self.sensor, self.camera,
                               self.lon, self.lat, self.height, backend="numpy")
        b = sample_reflectance(dn, self.meta, self.sensor, self.camera,
                               self.lon, self.lat, self.height, backend="native")
        ok = np.isfinite(a) & np.isfinite(b)
        self.assertGreater(ok.sum(), 100)
        np.testing.assert_allclose(a[ok], b[ok], rtol=1e-5, atol=1e-9)

    def test_frame_edge_agrees(self) -> None:
        # Regression: numpy clamped u/v to width-1.001 while the kernel used the exact
        # coordinate, giving a ~0.1% disagreement in the last pixel column/row.
        from rededgep.geometry import project

        rng = np.random.default_rng(3)
        # the frame spans roughly +-9.6 m x +-7.5 m at 20 m altitude, so sample wider
        # than that to guarantee frame-edge points
        lon = self.camera.lon + (rng.random(6000) - 0.5) * 3.4e-4
        lat = self.camera.lat + (rng.random(6000) - 0.5) * 2.6e-4
        height = np.full(lon.size, self.camera.alt - 20.0)
        u, v = project(self.camera, self.sensor, lon, lat, height)
        edge = np.zeros(u.size, bool)
        finite = np.isfinite(u)
        edge[finite] = (u[finite] < 2) | (u[finite] > 1454) | (v[finite] < 2) | (v[finite] > 1086)
        if edge.sum() < 10:
            self.skipTest("no frame-edge points in this sample")
        a = sample_reflectance(self.dn, self.meta, self.sensor, self.camera, lon, lat, height, backend="numpy")
        b = sample_reflectance(self.dn, self.meta, self.sensor, self.camera, lon, lat, height, backend="native")
        ok = edge & np.isfinite(a) & np.isfinite(b)
        np.testing.assert_allclose(a[ok], b[ok], rtol=1e-6, atol=1e-12)

    def test_behind_camera_does_not_crash_numpy_backend(self) -> None:
        # Regression: behind-camera points give NaN coordinates, which used to crash
        # the numpy backend with IndexError instead of returning NaN.
        lon = np.array([self.camera.lon, self.camera.lon + 1e-5])
        lat = np.array([self.camera.lat, self.camera.lat])
        height = np.array([self.camera.alt + 5.0, self.camera.alt - 20.0])
        out = sample_reflectance(self.dn, self.meta, self.sensor, self.camera, lon, lat, height, backend="numpy")
        self.assertTrue(np.isnan(out[0]))
        self.assertTrue(np.isfinite(out[1]))

    def test_invalid_points_agree(self) -> None:
        lon = np.array([self.camera.lon + 1e-1])  # far off-frame
        lat = np.array([self.camera.lat])
        height = np.array([self.camera.alt - 20.0])
        a = sample_reflectance(self.dn, self.meta, self.sensor, self.camera, lon, lat, height, backend="numpy")
        b = sample_reflectance(self.dn, self.meta, self.sensor, self.camera, lon, lat, height, backend="native")
        self.assertTrue(np.isnan(a[0]))
        self.assertTrue(np.isnan(b[0]))


class TestFallback(unittest.TestCase):
    def test_numpy_backend_always_works(self) -> None:
        sensor = make_calibration().sensor("Green")
        meta = make_meta()
        camera = make_camera()
        dn = np.full((1088, 1456), 9000, dtype=np.uint16)
        out = sample_reflectance(dn, meta, sensor, camera,
                                 np.array([camera.lon]), np.array([camera.lat]),
                                 np.array([camera.alt - 20.0]), backend="numpy")
        self.assertTrue(np.isfinite(out[0]))


if __name__ == "__main__":
    unittest.main()
