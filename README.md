# rededgep

**MicaSense RedEdge-P raw imagery → calibrated reflectance.** Small, dependency-light,
and reproduces the radiometric chain Agisoft Metashape applies.

```python
import rededgep as rp

capture = rp.Capture.from_directory("flight/0004SET/000", "IMG_0000")
reflectance = capture.reflectance()        # {'Blue': float32 array, ...}
stack = capture.stack()                    # (5, rows, cols) in canonical band order
```

Raw DN in, reflectance out. Two dependencies (`numpy`, `pillow`); no Metashape licence,
no GDAL, no photogrammetry engine.

---

## Why

Most RedEdge-P pipelines can tell you *that* a reflectance product differs from
Metashape's, not *why*. `rededgep` implements the chain term by term so the difference is
attributable:

| Term | Implementation |
|---|---|
| Band order | raw `_1.._5` is Blue, Green, Red, **NIR, Red edge**; the exported stack is Blue, Green, Red, **Red edge, NIR** |
| Black level | per image, from EXIF, averaged over the four values |
| Vignetting | `V = exp(Σ cᵢⱼ xⁱ yʲ)`, **divided**, normalised to unit mean |
| Gain | `ISO / 100` |
| Radiance | `L = a1 · (DN − black) / (V/mean V) / (g · tₑ · 2¹⁶)` |
| Reflectance | `ρ = π · L / E_horizontal` |
| Geometry | Agisoft distortion model, **per-band** intrinsics (they differ by up to 45 px) |

Getting the vignetting direction wrong is the classic error: multiplying instead of
dividing still matches the whole-image *median* while leaving a **2.2× gradient** between
image centre and corner.

---

## Install

```bash
pip install rededgep
```

From a checkout:

```bash
git clone https://github.com/mak8427/rededgep
cd rededgep
pip install -e .
```

Optional C acceleration (needs a C compiler):

```bash
python -m rededgep.native --build     # or: make native
```

Without it everything still works; the numpy path returns identical results.

---

## Quickstart

### A whole capture

```python
import rededgep as rp

capture = rp.Capture.from_directory("flight/0004SET/000", "IMG_0000")
print(capture.bands)      # ('Blue', 'Green', 'Red', 'Red edge', 'NIR', 'Panchro')

stack = capture.stack()   # (5, 1088, 1456) float32, canonical order
```

### Reflectance at ground points

```python
import numpy as np
from PIL import Image

meta = rp.read_metadata("IMG_0000_2.tif")
dn = np.asarray(Image.open("IMG_0000_2.tif"))

calibration = rp.Calibration.rededgep()
sensor = calibration.sensor("Green")

cameras = rp.load_cameras_txt("cameras.txt")
rho = rp.sample_reflectance(dn, meta, sensor, cameras[0], lon, lat, height)
```

### Calibration from a Metashape project

```python
calibration = rp.Calibration.from_psx("project.psx")
```

A `.psx` is a small XML pointer to a sidecar directory of zipped XML. The sensor
calibration is plain XML inside; no licence is needed to read it.

---

## Validation

Validated against Agisoft Metashape 2.2.1 on a 2025 multi-temporal sugar-beet campaign
(RedEdge-P, DLS2, oblique acquisitions).

| Check | Result |
|---|---|
| Per-band spectral **shape** vs Metashape (6 captures, 3 weeks, 2 projects) | median **2.2 %**, max **7.3 %** |
| Absolute vs a physical reflectance panel (level-panel flight) | 4 of 5 bands within **4 %** |
| Pixel-paired agreement after the vignetting fix | r = **0.92–0.97** |

### The honest limitation

Metashape applies a **per-flight brightness coefficient** that it never writes to the
project (`calibrateReflectance()` returns nothing). Across flights it spans **0.83–1.06**,
and it is not recoverable from exposure, ISO, black level, either DLS channel, solar
geometry, DLS tilt, the panel calibration, or either vignetting polynomial — all checked.

For the *same ground* Metashape's own two flights disagree by **11–29 %**, so that
coefficient is partly Metashape's own inconsistency, not a modelling gap here.

**Practical consequence:** this library gives you Metashape-compatible *spectral shape* to
a few percent. Absolute level is comparable within ~±10 % between flights. If you need
tighter, fly a levelled reference target.

> The built-in sensor defaults in `_defaults.py` were extracted from Metashape projects for
> one camera. `Calibration.from_psx` is the right call for any other body.

---

## Performance

Measured on an i9-13900K (24 cores / 32 threads), numpy 2.5.4, gcc 16.2
(`python benchmarks/bench_core.py`):

| Operation | Before | Now | Speedup |
|---|---:|---:|---:|
| `Capture.reflectance`, 5 bands × 1088×1456 | 491 ms | **2.1 ms** warm | **232×** |
| ↳ first call (builds the 5 vignetting maps) | — | 45 ms | — |
| `sample_reflectance`, 1 M points, 1 thread | 82 ms | **34 ms** | 2.4× |
| `sample_reflectance`, 1 M points, 32 threads | 82 ms | **3.9 ms** | **21×** |
| `sample_reflectance`, 1 M points, numpy | 166 ms | 130 ms | 1.3× |

Both hot paths are now bandwidth-bound, and getting there was three algorithmic changes,
not compiler tuning:

1. **The vignetting map is a fixed pattern.** It was being rebuilt for every image —
   1088×1456 pixels, ten polynomial terms, one `exp` each, ~52 ms per band, 95% of the
   full-frame cost. It is cached per (coefficients, shape) as float32.
2. **Pass count beats arithmetic.** The frame transform was three numpy passes (subtract,
   clip, multiply) moving ~41 MB instead of the 15.8 MB it strictly needs, which capped it
   at 45 GB/s of the 57 GB/s this machine sustains. One fused C pass reaches **70 GB/s**
   and is 4.1× faster. Replacing the per-pixel divide with a cached reciprocal was worth
   only 11% — pass count was the real cost.
3. **The C kernel was not the bottleneck.** At 1 M points it took 31 ms while the
   Python-side geodetic→ECEF→ENU conversion took 56 ms. Moving that into C took the call
   from 82 ms to 34 ms.

### Threading

The sampling kernel is OpenMP-parallel across points; the worker count comes from
`REDEDGEP_THREADS` (default 1, for determinism). Scaling at 1 M points:

| threads | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| ms | 34.2 | 17.0 | 8.7 | 7.0 | 4.5 | 3.9 |
| ns/point | 34.2 | 17.0 | 8.7 | 7.0 | 4.5 | 3.9 |

Output is bit-identical across thread counts, and `make verify` checks that.

### I/O

Once the maths got cheap, reading the imagery dominated: a single RedEdge-P band costs
~8 ms from disk against 0.24 ms of reflectance work. Two changes address that.

**Parallel loading.** `Capture.from_directory` reads the six bands in parallel (PIL
releases the GIL during decode). On a warm cache, 12.9 ms becomes 7.0 ms.

**Windowed sampling.** `sample_reflectance_from_file` projects the points first, then
reads only the bounding window they touch, instead of the whole 3.2 MB frame:

| | time |
|---|---:|
| read whole frame + sample 24 plot centres | 0.47 ms |
| read only the window | **0.17 ms** |
| | **2.8×** |

It chooses automatically: the window is used only while the pixels saved outweigh the
slower sampler it forces. The break-even is derived, not guessed — reading costs ~0.75 ns
per pixel while the numpy sampler costs ~130 ns/point against the kernel's ~34 ns/point,
so windowing wins while `(frame - window) * 0.75 > n * 96`. A million-point grid
therefore reads the whole frame and uses the kernel (30.5 ms against 29.6 ms direct — no
regression).

### Where the remaining time goes

| Path | Bound by | What would still help |
|---|---|---|
| frame transform | memory bandwidth (70 GB/s of ~87 achievable) | float16 output, tiled processing |
| point sampling | cache-miss latency (4 loads/point) | more threads, or sorting points by image tile |
| map build (first call) | `exp` over 1.58 M pixels | persist the maps to disk between runs |

The claims above are enforced by `benchmarks/bench_equivalence.py`, which fingerprints
every output (sha256 of the raw bytes, per band, per backend):

```bash
make verify
```

## API

| Object | Purpose |
|---|---|
| `rp.Capture` | one capture: `from_directory`, `reflectance`, `stack` |
| `rp.Calibration` | `rededgep()` built-in defaults, or `from_psx(path)` |
| `rp.Sensor` | per-band intrinsics, distortion, vignetting, black level |
| `rp.CameraPose` | geodetic position + rotation |
| `rp.sample_reflectance` | reflectance at arbitrary ground points, numpy or C |
| `rp.sample_reflectance_from_file` | same, but reads only the pixels the points touch |
| `rp.read_window` | read a pixel window without decoding the whole frame |
| `rp.read_metadata` | EXIF + XMP → `ImageMeta` |
| `rp.read_dn`, `rp.read_raster` | TIFF readers (GDAL when present) |
| `rp.vignetting`, `rp.vignetting_mean` | the polynomial and its normalisation |
| `rp.clear_caches` | release cached full-frame vignetting maps |
| `rp.set_worker_count` | tune band-level parallelism in `Capture.reflectance` |

---

## Tests

```bash
python -m unittest discover -s tests
```

52 tests, no test dependencies, synthetic fixtures only — the suite runs anywhere.

---

## Limitations

- RedEdge-P only. RedEdge-MX / Altum sensors differ in calibration and are not tested here.
- Reflectance requires DLS irradiance. Panel-only calibration is not implemented.
- No orthomosaic writer: `sample_reflectance` covers point sampling, not raster products.
- Geometry requires poses from elsewhere (e.g. Metashape `cameras.txt`, or ODM/OpenSfM
  exports). This library does not do bundle adjustment.

---

## License

MIT — see [LICENSE](LICENSE).
