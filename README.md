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

Measured on 32 cores, numpy 2.5.4, gcc 16.2 (`python benchmarks/bench_core.py`):

| Operation | numpy | fused C |
|---|---:|---:|
| `sample_reflectance`, 10 k points | 1.1 ms | 0.7 ms |
| `sample_reflectance`, 100 k points | 11.6 ms | 6.9 ms |
| `sample_reflectance`, 1 M points | 166 ms | 81 ms |
| `Capture.reflectance`, 5 bands × 1088×1456 | ~4–5 ms warm (~50 ms first call) | — |

The C kernel fuses projection, distortion, bilinear sampling, vignetting and the
radiometric conversion into one pass. The speedup is bounded by the random memory access
of bilinear sampling, not by arithmetic — which is why it is ~2× and not 20×.

**The bigger win is algorithmic.** The vignetting is a fixed pattern per band and frame
size, so `rededgep` builds the full-frame map once, stores it as a read-only float32
array normalised to unit mean, and reuses it for every subsequent image. Recomputing it
per band costs ~55 ms; a cache hit is a dictionary lookup. That is why a warm five-band
`Capture.reflectance` is a few milliseconds instead of a few hundred, and why a flight
loop is minutes rather than hours. The cache is bounded and can be released with
`rededgep.chain.clear_caches()`.

---

## API

| Object | Purpose |
|---|---|
| `rp.Capture` | one capture: `from_directory`, `reflectance`, `stack` |
| `rp.Calibration` | `rededgep()` built-in defaults, or `from_psx(path)` |
| `rp.Sensor` | per-band intrinsics, distortion, vignetting, black level |
| `rp.CameraPose` | geodetic position + rotation |
| `rp.sample_reflectance` | reflectance at arbitrary ground points, numpy or C |
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

47 tests, no test dependencies, synthetic fixtures only — the suite runs anywhere.

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
