# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Changed

- Full-frame vignetting maps are computed once per calibration and frame shape and
  reused, stored as read-only float32 arrays normalised to unit mean. The bounded cache
  is released with `rededgep.chain.clear_caches()`; `Sensor.vignetting_mean` shares it.
- The vignetting polynomial now uses a shared power table grouped by powers of `x`,
  and full-frame maps are built from 1-D coordinate vectors instead of `meshgrid`.
- `Capture.reflectance` evaluates the full-frame chain in float32, with agreement
  against the previous float64 result well below 1e-6 relative, and processes bands in
  a thread pool (`rededgep.set_worker_count`, `REDEDGEP_WORKERS`).

### Performance

- `Capture.reflectance` for 5 bands of 1088×1456: ~400 ms before, ~4–5 ms warm and
  ~50 ms on the first call that builds the five maps (measured with
  `python benchmarks/bench_core.py`).

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
