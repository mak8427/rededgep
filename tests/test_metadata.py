"""XMP parsing, including the failure modes that matter in the field."""

from __future__ import annotations

import unittest

from rededgep.metadata import DLS_SCALE_TO_SI, MetadataError, parse_xmp

GOOD_XMP = """<x:xmpmeta>
 <rdf:Description xmlns:Camera="http://pix4d.com/camera/1.0">
  <Camera:BandName>Green</Camera:BandName>
 </rdf:Description>
 <rdf:Description xmlns:MicaSense="http://micasense.com/MicaSense/1.0/">
  <MicaSense:RadiometricCalibration>
   <rdf:Seq>
    <rdf:li>0.0001703973</rdf:li>
    <rdf:li>-1.474282e-10</rdf:li>
    <rdf:li>9.020843e-05</rdf:li>
   </rdf:Seq>
  </MicaSense:RadiometricCalibration>
 </rdf:Description>
 <rdf:Description xmlns:DLS="http://micasense.com/DLS/1.0/">
  <DLS:HorizontalIrradiance>154.25</DLS:HorizontalIrradiance>
  <DLS:SpectralIrradiance>156.25</DLS:SpectralIrradiance>
  <DLS:SolarElevation>1.0431650693189627</DLS:SolarElevation>
  <DLS:SolarAzimuth>3.0651266870820053</DLS:SolarAzimuth>
 </rdf:Description>
</x:xmpmeta>"""


class TestParseXmp(unittest.TestCase):
    def test_reads_band_and_coefficients(self) -> None:
        fields = parse_xmp(GOOD_XMP)
        self.assertEqual(fields.band, "Green")
        self.assertAlmostEqual(fields.a1, 0.0001703973, places=12)
        self.assertAlmostEqual(fields.a2, -1.474282e-10, places=20)
        self.assertAlmostEqual(fields.a3, 9.020843e-05, places=18)

    def test_applies_dls_scale(self) -> None:
        fields = parse_xmp(GOOD_XMP)
        self.assertAlmostEqual(fields.horizontal_irradiance, 154.25 * DLS_SCALE_TO_SI, places=12)
        self.assertAlmostEqual(fields.spectral_irradiance, 156.25 * DLS_SCALE_TO_SI, places=12)

    def test_reads_solar_geometry(self) -> None:
        fields = parse_xmp(GOOD_XMP)
        self.assertAlmostEqual(fields.solar_elevation_rad, 1.0431650693189627, places=12)
        self.assertAlmostEqual(fields.solar_azimuth_rad, 3.0651266870820053, places=12)

    def test_missing_band_raises(self) -> None:
        with self.assertRaises(MetadataError):
            parse_xmp("<x:xmpmeta></x:xmpmeta>")

    def test_missing_calibration_raises(self) -> None:
        with self.assertRaises(MetadataError):
            parse_xmp("<Camera:BandName>Red</Camera:BandName>")

    def test_short_calibration_raises(self) -> None:
        xmp = GOOD_XMP.replace("<rdf:li>9.020843e-05</rdf:li>", "")
        with self.assertRaises(MetadataError):
            parse_xmp(xmp)

    def test_missing_dls_gives_zero(self) -> None:
        xmp = GOOD_XMP.split("<rdf:Description xmlns:DLS")[0]
        fields = parse_xmp(xmp)
        self.assertEqual(fields.horizontal_irradiance, 0.0)


if __name__ == "__main__":
    unittest.main()
