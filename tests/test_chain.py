"""Reflectance chain: vignetting, scale and the DN to reflectance conversion."""

from __future__ import annotations

import unittest

import numpy as np

from conftest import make_calibration, make_meta

from rededgep.chain import dn_to_radiance, dn_to_reflectance, reflectance_scale, vignetting, vignetting_mean


class TestVignetting(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor = make_calibration().sensor("Green")

    def test_centre_is_near_unity(self) -> None:
        u = np.array([self.sensor.width / 2.0])
        v = np.array([self.sensor.height / 2.0])
        value = vignetting(self.sensor.vignetting, u, v, self.sensor.width, self.sensor.height)[0]
        self.assertAlmostEqual(float(value), 1.0, delta=0.05)

    def test_corner_is_attenuated(self) -> None:
        corner = vignetting(self.sensor.vignetting, np.array([0.0]), np.array([0.0]))[0]
        self.assertLess(float(corner), 0.75)

    def test_mean_is_between_centre_and_corner(self) -> None:
        mean = vignetting_mean(self.sensor.vignetting, self.sensor.width, self.sensor.height)
        self.assertGreater(mean, 0.6)
        self.assertLess(mean, 1.0)

    def test_mean_is_cached_and_stable(self) -> None:
        first = vignetting_mean(self.sensor.vignetting)
        second = vignetting_mean(self.sensor.vignetting)
        self.assertEqual(first, second)


class TestReflectance(unittest.TestCase):
    def setUp(self) -> None:
        self.sensor = make_calibration().sensor("Green")
        self.meta = make_meta()

    def test_scale_matches_definition(self) -> None:
        expected = self.meta.a1 / (self.meta.gain * self.meta.exposure_s * 2.0**16)
        self.assertAlmostEqual(reflectance_scale(self.meta), expected, places=18)

    def test_black_level_removes_pedestal(self) -> None:
        below = dn_to_radiance(np.array([self.meta.black_level - 10.0]), self.meta, self.sensor)
        self.assertEqual(float(below[0]), 0.0)

    def test_reflectance_is_proportional_to_dn(self) -> None:
        a = dn_to_reflectance(np.array([6000.0]), self.meta, self.sensor)[0]
        b = dn_to_reflectance(np.array([12000.0]), self.meta, self.sensor)[0]
        self.assertAlmostEqual(float(b / a), (12000 - 3840) / (6000 - 3840), places=6)

    def test_pi_relation(self) -> None:
        dn = np.array([8000.0])
        radiance = dn_to_radiance(dn, self.meta, self.sensor)[0]
        reflectance = dn_to_reflectance(dn, self.meta, self.sensor)[0]
        self.assertAlmostEqual(float(reflectance), float(np.pi * radiance / self.meta.horizontal_irradiance), places=12)

    def test_missing_irradiance_raises(self) -> None:
        meta = make_meta(horizontal_irradiance=0.0)
        with self.assertRaises(ValueError):
            dn_to_reflectance(np.array([8000.0]), meta, self.sensor)

    def test_vignetting_only_applies_when_coordinates_given(self) -> None:
        dn = np.array([9000.0])
        plain = dn_to_reflectance(dn, self.meta, self.sensor)[0]
        corner = dn_to_reflectance(dn, self.meta, self.sensor, np.array([0.0]), np.array([0.0]))[0]
        self.assertGreater(float(corner), float(plain))


if __name__ == "__main__":
    unittest.main()
