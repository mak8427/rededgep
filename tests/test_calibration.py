"""Calibration container and the built-in RedEdge-P sensor defaults."""

from __future__ import annotations

import unittest

from conftest import make_calibration

from rededgep.calibration import Calibration, Sensor
from rededgep.metadata import BANDS


class TestBuiltinCalibration(unittest.TestCase):
    def test_has_all_six_sensors(self) -> None:
        calibration = make_calibration()
        self.assertEqual(set(calibration.bands), set(BANDS))

    def test_sensor_lookup(self) -> None:
        calibration = make_calibration()
        sensor = calibration.sensor("NIR")
        self.assertIsInstance(sensor, Sensor)
        self.assertEqual(sensor.band, "NIR")
        self.assertGreater(sensor.focal_px, 1000.0)

    def test_unknown_band_raises(self) -> None:
        with self.assertRaises(KeyError):
            make_calibration().sensor("Ultraviolet")

    def test_panchro_has_longer_focal_length(self) -> None:
        calibration = make_calibration()
        # the panchro sensor has a ~10.4 mm lens against 5.5 mm for the spectral bands
        self.assertGreater(calibration.sensor("Panchro").focal_px, 1.5 * calibration.sensor("Blue").focal_px)

    def test_vignetting_terms_are_indexed_pairs(self) -> None:
        for band in BANDS:
            for i, j, coefficient in make_calibration().sensor(band).vignetting:
                self.assertIsInstance(i, int)
                self.assertIsInstance(j, int)
                self.assertIsInstance(coefficient, float)


class TestScaling(unittest.TestCase):
    def test_same_size_returns_same_object(self) -> None:
        sensor = make_calibration().sensor("Red")
        self.assertIs(sensor.scaled(width=sensor.width, height=sensor.height), sensor)

    def test_half_size_halves_focal_and_offsets(self) -> None:
        sensor = make_calibration().sensor("Red")
        scaled = sensor.scaled(width=sensor.width // 2, height=sensor.height // 2)
        self.assertAlmostEqual(scaled.focal_px, sensor.focal_px / 2, places=6)
        self.assertAlmostEqual(scaled.cx, sensor.cx / 2, places=6)
        self.assertAlmostEqual(scaled.cy, sensor.cy / 2, places=6)

    def test_distortion_is_preserved_by_scaling(self) -> None:
        sensor = make_calibration().sensor("Blue")
        scaled = sensor.scaled(width=728, height=544)
        self.assertEqual((scaled.k1, scaled.k2, scaled.k3), (sensor.k1, sensor.k2, sensor.k3))


class TestFromPsx(unittest.TestCase):
    def test_missing_project_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Calibration.from_psx("/nonexistent/project.psx")

    def test_project_without_sidecar_raises(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.psx"
            path.write_text("<document/>")
            with self.assertRaises(FileNotFoundError):
                Calibration.from_psx(path)


if __name__ == "__main__":
    unittest.main()
