# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [semantic versioning](https://semver.org/).

## [0.1.0] - 2026-11-03

### Added

- Raw RedEdge-P metadata reader: EXIF gain/exposure/black level and XMP radiometric
  coefficients plus DLS irradiance, using Pillow only.
- Sensor calibration extracted from Agisoft Metashape projects (`Calibration.from_psx`)
  and a built-in RedEdge-P default for use without a project.
- The reflectance chain: black level, divided unit-mean vignetting, ISO gain, exposure,
  and DLS horizontal irradiance.
- Camera projection with the Agisoft distortion model and per-band intrinsics.
- `Capture` API for whole captures and `sample_reflectance` for ground points.
- Optional fused C kernel with a numpy fallback returning identical values.
- Unit suite (47 tests, synthetic fixtures, no test dependencies) and benchmarks.
