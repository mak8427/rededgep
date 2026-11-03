"""Windowed reading, parallel loading and windowed point sampling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from conftest import make_calibration, make_camera, make_meta, write_band, write_capture

import rededgep as rp
from rededgep.capture import Capture
from rededgep.readers import read_dn, read_window
from rededgep.sampling import sample_reflectance, sample_reflectance_from_file


class TestReadWindow(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = write_band(Path(self._tmp.name) / "IMG_0000_2.tif", shape=(64, 96), texture=True)
        self.full = read_dn(self.path)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_window_matches_slice(self) -> None:
        window = read_window(self.path, 10, 8, 50, 40)
        self.assertEqual(window.shape, (32, 40))
        np.testing.assert_array_equal(window, self.full[8:40, 10:50])

    def test_full_window_matches_whole_image(self) -> None:
        np.testing.assert_array_equal(read_window(self.path, 0, 0, 96, 64), self.full)

    def test_empty_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            read_window(self.path, 10, 10, 10, 20)

    def test_outside_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            read_window(self.path, 0, 0, 200, 64)


class TestParallelLoad(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = write_capture(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parallel_matches_serial(self) -> None:
        serial = Capture.from_directory(self.dir, "IMG_0000", workers=1)
        parallel = Capture.from_directory(self.dir, "IMG_0000", workers=4)
        self.assertEqual(serial.bands, parallel.bands)
        for band in serial.dn:
            np.testing.assert_array_equal(serial.dn[band], parallel.dn[band])
        self.assertEqual(set(serial.meta), set(parallel.meta))
        for band in serial.meta:
            self.assertEqual(serial.meta[band], parallel.meta[band])

    def test_metadata_is_read(self) -> None:
        capture = Capture.from_directory(self.dir, "IMG_0000", workers=2)
        self.assertEqual(set(capture.meta), set(rp.BANDS))
        self.assertTrue(all(m.iso == 100.0 for m in capture.meta.values()))

    def test_reflectance_is_finite(self) -> None:
        capture = Capture.from_directory(self.dir, "IMG_0000", workers=2)
        for band, values in capture.reflectance().items():
            self.assertTrue(np.all(np.isfinite(values)), band)


class TestWindowedSampling(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = write_band(Path(self._tmp.name) / "IMG_0000_2.tif", shape=(64, 96), texture=True)
        self.calibration = make_calibration()
        self.sensor = self.calibration.sensor("Green").scaled(width=96, height=64)
        self.meta = make_meta(width=96, height=64)
        self.camera = make_camera()
        rng = np.random.default_rng(5)
        self.lon = self.camera.lon + (rng.random(300) - 0.5) * 2e-5
        self.lat = self.camera.lat + (rng.random(300) - 0.5) * 1.4e-5
        self.height = np.full(300, self.camera.alt - 20.0)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_windowed_matches_full_image(self) -> None:
        full = sample_reflectance(read_dn(self.path), self.meta, self.sensor, self.camera,
                                  self.lon, self.lat, self.height, backend="numpy")
        windowed = sample_reflectance_from_file(self.path, self.meta, self.sensor, self.camera,
                                                self.lon, self.lat, self.height)
        ok = np.isfinite(full) & np.isfinite(windowed)
        self.assertGreater(ok.sum(), 50)
        np.testing.assert_allclose(full[ok], windowed[ok], rtol=1e-12, atol=0)
        np.testing.assert_array_equal(np.isnan(full), np.isnan(windowed))

    def test_offscreen_points_give_nan(self) -> None:
        lon = np.array([self.camera.lon + 1e-2])
        lat = np.array([self.camera.lat])
        height = np.array([self.camera.alt - 20.0])
        out = sample_reflectance_from_file(self.path, self.meta, self.sensor, self.camera, lon, lat, height)
        self.assertTrue(np.isnan(out[0]))


if __name__ == "__main__":
    unittest.main()
