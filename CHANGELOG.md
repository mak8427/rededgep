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

## [Unreleased]

### Performance

- Cache the vignetting map per (coefficients, shape) as a float32 unit-mean array instead
  of rebuilding a 1088x1456 polynomial for every image. `Capture.reflectance` for five
  bands drops from ~491 ms to 4.6 ms warm (44.5 ms for the first call, which builds the
  maps). Roughly 107x.
- Move the geodetic to ECEF to ENU conversion into the C kernel. It accounted for 68% of
  `sample_reflectance` at 1M points; that call now takes 34 ms instead of 82 ms, 4.9x the
  numpy backend.
- Build the vignetting polynomial with a cumulative power table and 1-D broadcasting
  rather than `meshgrid` and `x**i * y**j`.
- Parallelise `Capture.reflectance` across bands; worker count is configurable with
  `rededgep.set_worker_count(n)` or `REDEDGEP_WORKERS`.
- Add `rededgep.clear_caches()` to release cached vignetting maps.

- Fuse the frame transform into a single C pass (`rp_apply_frame`). The three-pass numpy
  form moved ~41 MB per band against the 15.8 MB strictly needed, capping it at 45 GB/s;
  the fused pass reaches 70 GB/s. Warm `Capture.reflectance` is now 2.1 ms.
- Cache the reciprocal of the vignetting map so the hot loop multiplies instead of
  dividing.
- Parallelise the sampling kernel with OpenMP (`REDEDGEP_THREADS`, default 1). 8.8x at 32
  threads, bit-identical output across thread counts.

### Added

- `read_window` / `read_dn_window`: read a pixel window without decoding the whole frame.
- `sample_reflectance_from_file`: reflectance at ground points, reading only the pixels
  they touch. 2.8x faster than loading the frame for plot-scale sampling, and it falls
  back to the full frame plus the C kernel when the points cover most of the image.
- `Capture.from_directory` reads the six bands in parallel (~1.8x).

### Fixed

- `read_metadata` only looked for the exposure time and black level in the Exif sub-IFD
  and never fell back to the top-level IFD, so files written that way were rejected.
  It also did not accept `(numerator, denominator)` rationals.
- `sampling.bilinear` raised `IndexError` for non-finite coordinates. Behind-camera points
  produce NaN, which was cast to `INT64_MIN` before masking, so the numpy backend crashed
  instead of returning NaN.
- `sampling.bilinear` clamped coordinates to `width - 1.001`, disagreeing with the C kernel
  by up to 0.1% in the last pixel column and row. Both now use the exact coordinate.

### Added

- `benchmarks/bench_equivalence.py` (`make verify`): a deterministic pre/post-optimisation
  harness that fingerprints every output and checks the two backends agree.

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
