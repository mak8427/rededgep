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
