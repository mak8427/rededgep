"""Public API surface: the pieces a user touches directly."""

from __future__ import annotations

import unittest

import numpy as np

from conftest import make_calibration, make_meta

import rededgep as rp
from rededgep.capture import Capture, STACK_ORDER
from rededgep.sampling import bilinear


class TestPublicApi(unittest.TestCase):
    def test_version_and_exports(self) -> None:
        self.assertTrue(rp.__version__)
        for name in ("Capture", "Calibration", "sample_reflectance", "read_metadata", "project"):
            self.assertTrue(hasattr(rp, name), name)

    def test_band_constants_are_consistent(self) -> None:
        self.assertEqual(len(rp.BANDS), 6)
        self.assertEqual(set(rp.SUFFIX_TO_BAND.values()), set(rp.BANDS))
        self.assertEqual(set(STACK_ORDER), set(rp.BANDS) - {"Panchro"})


class TestBilinear(unittest.TestCase):
    def test_exact_at_integer_coordinates(self) -> None:
        array = np.arange(100, dtype=float).reshape(10, 10)
        self.assertEqual(float(bilinear(array, np.array([3.0]), np.array([4.0]))[0]), 43.0)

    def test_midpoint_is_mean_of_four(self) -> None:
        array = np.array([[0.0, 2.0], [4.0, 6.0]])
        self.assertAlmostEqual(float(bilinear(array, np.array([0.5]), np.array([0.5]))[0]), 3.0)

    def test_outside_is_nan(self) -> None:
        array = np.zeros((5, 5))
        self.assertTrue(np.isnan(bilinear(array, np.array([10.0]), np.array([0.0]))[0]))

    def test_non_finite_coordinates_do_not_crash(self) -> None:
        # Regression: NaN was cast to INT64_MIN before masking, raising IndexError.
        # Behind-camera points produce NaN coordinates, so this crashed in normal use.
        array = np.arange(100, dtype=float).reshape(10, 10)
        for bad in (np.nan, np.inf, -np.inf, -1e9, 1e9):
            result = bilinear(array, np.array([bad]), np.array([bad]))
            self.assertTrue(np.isnan(result[0]), f"{bad} should give nan")

    def test_matched_mask_with_non_finite(self) -> None:
        array = np.arange(100, dtype=float).reshape(10, 10)
        result = bilinear(array, np.array([np.nan, 3.0]), np.array([np.nan, 4.0]))
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(float(result[1]), 43.0)

    def test_last_pixel_column_is_reachable(self) -> None:
        array = np.arange(100, dtype=float).reshape(10, 10)
        self.assertEqual(float(bilinear(array, np.array([9.0]), np.array([9.0]))[0]), 99.0)


class TestCapture(unittest.TestCase):
    def _capture(self, bands=("Blue", "Green", "Red", "Red edge", "NIR")) -> Capture:
        shape = (64, 96)
        dn = {b: np.full(shape, 9000, dtype=np.uint16) for b in bands}
        meta = {b: make_meta(band=b, width=shape[1], height=shape[0]) for b in bands}
        return Capture(name="synthetic", dn=dn, meta=meta)

    def test_reflectance_shapes_and_dtype(self) -> None:
        capture = self._capture()
        result = capture.reflectance(make_calibration())
        self.assertEqual(set(result), set(("Blue", "Green", "Red", "Red edge", "NIR")))
        for array in result.values():
            self.assertEqual(array.shape, (64, 96))
            self.assertEqual(array.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(array)))

    def test_stack_canonical_order(self) -> None:
        capture = self._capture()
        stack = capture.stack(make_calibration())
        self.assertEqual(stack.shape, (5, 64, 96))
        plain = capture.reflectance(make_calibration())
        for index, band in enumerate(STACK_ORDER):
            np.testing.assert_allclose(stack[index], plain[band])

    def test_stack_rejects_missing_band(self) -> None:
        capture = self._capture(bands=("Blue", "Green"))
        with self.assertRaises(ValueError):
            capture.stack(make_calibration())

    def test_missing_directory_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Capture.from_directory("/nonexistent", "IMG_0000")


if __name__ == "__main__":
    unittest.main()
