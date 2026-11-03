#!/usr/bin/env python3
"""Pre/post-optimisation equivalence harness for rededgep.

Run this before an optimisation and again after it: the fingerprint lines on
stdout must be byte-identical, while the timing table (stderr) should improve.
Because timings are not reproducible they go to stderr, so stdout stays fully
deterministic and ``diff before.txt after.txt`` is meaningful::

    python3 benchmarks/bench_equivalence.py > before.txt
    # ... apply the optimisation ...
    python3 benchmarks/bench_equivalence.py > after.txt
    diff -u before.txt after.txt

What is fingerprinted
---------------------
sampling
    ``sample_reflectance`` with the numpy backend and, when available, with the
    native kernel, on three synthetic DN images (flat field, linear gradient,
    seeded white-noise texture) and three seeded ground-point sets (1k, 10k,
    100k).  Each point set is 95 % in-frame points plus a 5 % ring of points
    well outside the image, so both backends must agree on the valid mask as
    well.  For every image/point-set/backend combination the harness prints
    the number of in-frame points, mean/std/min/max of the finite reflectance
    values, and a sha256 of the sorted finite values as little-endian float64.
capture
    ``Capture.reflectance`` and ``Capture.stack`` on a five-band synthetic
    capture.  Per band: mean/std/min/max and a sha256 of the raw little-endian
    float32 bytes (plus one hash for the whole stack).

Cross-backend equivalence
-------------------------
The numpy backend is the reference.  A combination passes when both backends
mark exactly the same points as valid and, for every valid point,

    |native - numpy| <= atol + rtol * |numpy|

with ``atol = 1e-8`` and ``rtol`` taken from ``--tolerance`` (default 1e-6).
The absolute floor keeps reflectance values near zero from being judged by a
meaningless relative error; the relative term absorbs the float32 rounding of
the native kernel (its output is float32, while numpy works in float64).  A
violation prints ``status=FAIL`` lines and exits non-zero.

Determinism
-----------
Every input comes from a fixed seed, a fixed camera pose and fixed constants;
no data files, clock or entropy source touch the computation.  On the same
build, two runs produce byte-identical stdout.

Exit status is 0 when all checks pass (checks skipped because the native
library is unavailable also count as a pass) and 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The package is not installed in every checkout; make ``python3
# benchmarks/bench_equivalence.py`` work straight from the worktree root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rededgep as rp  # noqa: E402  (import after the path setup)
from rededgep import native  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed workload.  Changing any of these constants changes the fingerprints.
# ---------------------------------------------------------------------------

WIDTH = 1456
HEIGHT = 1088
BAND = "Green"

IMAGE_NAMES = ("flat", "gradient", "textured")
POINT_COUNTS = (1_000, 10_000, 100_000)

SEED_TEXTURED = 20251103
#: Point set of size ``n`` is generated from ``SEED_POINTS_BASE + n``, so adding
#: a size later does not perturb the other sets.
SEED_POINTS_BASE = 1_000_003

CAMERA_LON = 9.9146
CAMERA_LAT = 51.5628
CAMERA_ALT = 207.0
CAMERA_AGL = 20.0  # ground plane 20 m below the camera

#: Interior ground rectangle, metres east/north of the camera.  At 20 m AGL the
#: image footprint is roughly east [-8.8, 9.6] and north [-7.5, 6.3] metres;
#: these bounds keep every interior point at least ~24 px inside the frame.
#: The margin is deliberate: the numpy reference clamps sample coordinates to
#: ``width - 1.001`` while the C kernel uses the exact coordinate, so a valid
#: point in the last 0.001 px of the frame can differ by ~1e-5 in reflectance
#: on a high-contrast image.  That pre-existing edge effect is not something an
#: optimisation would introduce, and it would otherwise swamp the tolerance.
INTERIOR_EAST_M = (-8.5, 9.2)
INTERIOR_NORTH_M = (-7.2, 6.0)

#: 5 % of each point set lies well outside the frame, so both backends must
#: agree on the NaN mask too.  Keep these on the ground plane: behind-camera
#: points produce NaN coordinates, and the numpy ``bilinear`` helper converts
#: those to out-of-range indices (a pre-existing issue, unrelated to speed).
OFF_FRAME_FRACTION = 0.05
OFF_FRAME_EAST_M = (25.0, 40.0)
OFF_FRAME_NORTH_M = (-10.0, 10.0)

METRES_PER_DEGREE = 111_320.0

DEFAULT_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 1e-8
TIMING_REPEATS = 3

#: Five-band capture: Panchro is not part of ``Capture.stack``.  Reusing the
#: three images keeps the workload small while every band still gets its own
#: sensor, black level and vignetting polynomial.
CAPTURE_IMAGE_FOR_BAND = {
    "Blue": "flat",
    "Green": "gradient",
    "Red": "textured",
    "Red edge": "flat",
    "NIR": "gradient",
}


@dataclass(frozen=True)
class Workload:
    """All deterministic inputs, built once and reused by every section."""

    calibration: rp.Calibration
    sensor: rp.Sensor
    meta: rp.ImageMeta
    camera: rp.CameraPose
    images: dict[str, np.ndarray]
    points: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]
    capture: rp.Capture


# ---------------------------------------------------------------------------
# Synthetic workload
# ---------------------------------------------------------------------------


def make_meta(band: str = BAND, black_level: float = 3828.0) -> rp.ImageMeta:
    """Radiometric metadata for a synthetic RedEdge-P image."""
    return rp.ImageMeta(
        path=f"synthetic/{band}.tif",
        band=band,
        iso=100.0,
        exposure_s=0.0015,
        black_level=black_level,
        a1=1.7e-4,
        a2=0.0,
        a3=0.0,
        horizontal_irradiance=1.5,
        spectral_irradiance=1.52,
        solar_elevation_rad=1.0,
        solar_azimuth_rad=3.0,
        width=WIDTH,
        height=HEIGHT,
    )


def make_images() -> dict[str, np.ndarray]:
    """Flat, linearly ramped and seeded white-noise DN images (uint16)."""
    flat = np.full((HEIGHT, WIDTH), 9000, dtype=np.uint16)

    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    x = cols.astype(np.float64) / (WIDTH - 1)
    y = rows.astype(np.float64) / (HEIGHT - 1)
    gradient = np.rint(4000.0 + 34000.0 * (0.65 * x + 0.35 * y)).astype(np.uint16)

    # High-frequency texture on purpose: a changed sampling coordinate shows up
    # immediately, whereas a smooth image could hide sub-pixel regressions.
    rng = np.random.default_rng(SEED_TEXTURED)
    textured = rng.integers(4000, 40001, size=(HEIGHT, WIDTH), dtype=np.uint16)

    return {"flat": flat, "gradient": gradient, "textured": textured}


def make_points(n: int, camera: rp.CameraPose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n`` seeded ground points (degrees, degrees, metres) around the camera.

    The first 95 % are inside the frame (the ``INTERIOR_*`` rectangle); the rest
    are far enough east to fall outside it, which exercises the NaN path of both
    backends.
    """
    rng = np.random.default_rng(SEED_POINTS_BASE + n)
    n_off_frame = max(1, int(round(n * OFF_FRAME_FRACTION)))
    n_interior = n - n_off_frame

    east = np.empty(n, dtype=np.float64)
    north = np.empty(n, dtype=np.float64)
    height = np.full(n, camera.alt - CAMERA_AGL, dtype=np.float64)

    east[:n_interior] = rng.uniform(*INTERIOR_EAST_M, n_interior)
    north[:n_interior] = rng.uniform(*INTERIOR_NORTH_M, n_interior)
    east[n_interior:] = rng.uniform(*OFF_FRAME_EAST_M, n_off_frame)
    north[n_interior:] = rng.uniform(*OFF_FRAME_NORTH_M, n_off_frame)

    lat = camera.lat + north / METRES_PER_DEGREE
    # A degree of longitude shrinks with cos(latitude).
    lon = camera.lon + east / (METRES_PER_DEGREE * np.cos(np.radians(camera.lat)))
    return lon, lat, height


def make_capture(calibration: rp.Calibration, images: dict[str, np.ndarray]) -> rp.Capture:
    """Five-band capture with per-band metadata, reusing the synthetic images."""
    dn: dict[str, np.ndarray] = {}
    meta: dict[str, rp.ImageMeta] = {}
    for band in rp.STACK_ORDER:
        dn[band] = images[CAPTURE_IMAGE_FOR_BAND[band]]
        meta[band] = make_meta(band, calibration.sensor(band).black_level)
    return rp.Capture(name="synthetic", dn=dn, meta=meta)


def build_workload() -> Workload:
    calibration = rp.Calibration.rededgep()
    sensor = calibration.sensor(BAND)
    camera = rp.CameraPose(
        lon=CAMERA_LON,
        lat=CAMERA_LAT,
        alt=CAMERA_ALT,
        rotation=np.eye(3),  # nadir: ENU up maps to camera -Z
        label="synthetic-nadir",
    )
    images = make_images()
    points = {n: make_points(n, camera) for n in POINT_COUNTS}
    return Workload(
        calibration=calibration,
        sensor=sensor,
        meta=make_meta(BAND, sensor.black_level),
        camera=camera,
        images=images,
        points=points,
        capture=make_capture(calibration, images),
    )


def sample(workload: Workload, image_name: str, n: int, backend: str) -> np.ndarray:
    """Run ``sample_reflectance`` for one workload combination."""
    lon, lat, height = workload.points[n]
    return rp.sample_reflectance(
        workload.images[image_name],
        workload.meta,
        workload.sensor,
        workload.camera,
        lon,
        lat,
        height,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _num(value: float, digits: int = 12) -> str:
    """Stable text form for a float (``nan`` included)."""
    return f"{value:.{digits}g}"


def finite_stats(values: np.ndarray) -> tuple[int, float, float, float, float]:
    """Count, mean, std, min and max over the finite entries."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        missing = float("nan")
        return 0, missing, missing, missing, missing
    return (
        int(finite.size),
        float(finite.mean()),
        float(finite.std()),
        float(finite.min()),
        float(finite.max()),
    )


def sha256_float64_sorted(values: np.ndarray) -> str:
    """sha256 of the sorted finite values as little-endian float64 bytes."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    ordered = np.sort(finite).astype("<f8", copy=False)
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def sha256_float32_raw(values: np.ndarray) -> str:
    """sha256 of the raw little-endian float32 bytes, in array order."""
    data = np.ascontiguousarray(values, dtype="<f4")
    return hashlib.sha256(data.tobytes()).hexdigest()


def sample_fingerprint_line(image_name: str, n: int, backend: str, values: np.ndarray) -> str:
    valid, mean, std, low, high = finite_stats(values)
    return (
        f"[fp.sample] image={image_name} points={n} backend={backend} valid={valid} "
        f"mean={_num(mean)} std={_num(std)} min={_num(low)} max={_num(high)} "
        f"sha256={sha256_float64_sorted(values)}"
    )


def capture_fingerprint_line(kind: str, band: str, values: np.ndarray) -> str:
    valid, mean, std, low, high = finite_stats(values)
    return (
        f"[fp.{kind}] band={band} valid={valid} "
        f"mean={_num(mean, 9)} std={_num(std, 9)} min={_num(low, 9)} max={_num(high, 9)} "
        f"sha256={sha256_float32_raw(values)}"
    )


def collect_sample_fingerprints(
    workload: Workload, backends: tuple[str, ...], rtol: float, atol: float
) -> tuple[list[str], list[tuple[bool, str]]]:
    """Fingerprint every image/point-set/backend combination and compare backends."""
    lines: list[str] = []
    checks: list[tuple[bool, str]] = []
    for image_name in IMAGE_NAMES:
        for n in POINT_COUNTS:
            results: dict[str, np.ndarray] = {}
            for backend in backends:
                values = sample(workload, image_name, n, backend)
                results[backend] = values
                lines.append(sample_fingerprint_line(image_name, n, backend, values))
            if "native" in results:
                line, ok, message = sample_check_line(
                    image_name, n, results["numpy"], results["native"], rtol, atol
                )
                lines.append(line)
                checks.append((ok, message))
    return lines, checks


def sample_check_line(
    image_name: str,
    n: int,
    numpy_values: np.ndarray,
    native_values: np.ndarray,
    rtol: float,
    atol: float,
) -> tuple[str, bool, str]:
    """Compare the native result against numpy at one workload combination."""
    reference = np.asarray(numpy_values, dtype=np.float64)
    candidate = np.asarray(native_values, dtype=np.float64)
    finite_reference = np.isfinite(reference)
    finite_candidate = np.isfinite(candidate)
    valid_reference = int(finite_reference.sum())
    valid_candidate = int(finite_candidate.sum())
    prefix = f"[chk.sample] image={image_name} points={n}"

    if not np.array_equal(finite_reference, finite_candidate):
        mismatches = int(np.count_nonzero(finite_reference != finite_candidate))
        message = (
            f"sample image={image_name} points={n}: valid mask differs at {mismatches} "
            f"point(s) (numpy={valid_reference} valid, native={valid_candidate} valid)"
        )
        return (
            f"{prefix} valid_numpy={valid_reference} valid_native={valid_candidate} "
            f"mask_mismatches={mismatches} max_abs_diff=nan max_allowed=nan "
            f"status=FAIL detail=valid_mask_mismatch",
            False,
            message,
        )

    if valid_reference == 0:
        message = f"sample image={image_name} points={n}: no valid points to compare"
        return (
            f"{prefix} valid_numpy=0 valid_native=0 max_abs_diff=nan max_allowed=nan "
            f"status=FAIL detail=no_valid_points",
            False,
            message,
        )

    difference = np.abs(reference[finite_reference] - candidate[finite_reference])
    allowed = atol + rtol * np.abs(reference[finite_reference])
    # Report the point that is closest to failing: it is the informative one.
    worst = int(np.argmax(difference / allowed))
    max_diff = float(difference[worst])
    max_allowed = float(allowed[worst])
    ok = bool(np.all(difference <= allowed))
    status = "OK" if ok else "FAIL"
    suffix = "" if ok else " detail=tolerance_exceeded"
    message = ""
    if not ok:
        message = (
            f"sample image={image_name} points={n}: |native-numpy|={max_diff:.6g} "
            f"exceeds {max_allowed:.6g}"
        )
    return (
        f"{prefix} valid_numpy={valid_reference} valid_native={valid_candidate} "
        f"max_abs_diff={_num(max_diff, 6)} max_allowed={_num(max_allowed, 6)} "
        f"status={status}{suffix}",
        ok,
        message,
    )


def collect_capture_fingerprints(workload: Workload) -> list[str]:
    """Fingerprint ``Capture.reflectance`` and ``Capture.stack`` per band."""
    lines: list[str] = []
    reflectance = workload.capture.reflectance(workload.calibration)
    for band in rp.STACK_ORDER:
        lines.append(capture_fingerprint_line("capture.reflectance", band, reflectance[band]))

    stack = workload.capture.stack(workload.calibration)
    for index, band in enumerate(rp.STACK_ORDER):
        lines.append(capture_fingerprint_line("capture.stack", band, stack[index]))
    lines.append(f"[fp.capture.stack] band=ALL sha256={sha256_float32_raw(stack)}")
    return lines


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_best(function, repeats: int = TIMING_REPEATS) -> float:
    """Best-of-N wall time in seconds."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def print_timings(workload: Workload, native_ready: bool) -> None:
    """Emit the timing table on stderr (timings are not reproducible)."""
    stream = sys.stderr
    print("", file=stream)
    print(
        f"timing (best of {TIMING_REPEATS}; stderr only, not part of the fingerprint)",
        file=stream,
    )
    print(
        f"  {'image':<9s} {'points':>8s} {'numpy ms':>10s} {'native ms':>10s} {'speedup':>8s}",
        file=stream,
    )
    for image_name in IMAGE_NAMES:
        for n in POINT_COUNTS:
            # Bind the loop variables: time_best calls the thunk immediately, but
            # explicit defaults keep the lambdas safe if that ever changes.
            t_numpy = time_best(
                lambda name=image_name, count=n: sample(workload, name, count, "numpy")
            )
            if native_ready:
                t_native = time_best(
                    lambda name=image_name, count=n: sample(workload, name, count, "native")
                )
                print(
                    f"  {image_name:<9s} {n:>8,d} {t_numpy * 1e3:>10.3f} "
                    f"{t_native * 1e3:>10.3f} {t_numpy / t_native:>7.2f}x",
                    file=stream,
                )
            else:
                print(
                    f"  {image_name:<9s} {n:>8,d} {t_numpy * 1e3:>10.3f} "
                    f"{'n/a':>10s} {'n/a':>8s}",
                    file=stream,
                )

    # Supplementary full-frame timings; also best-of-N after the caches warm.
    t_reflectance = time_best(lambda: workload.capture.reflectance(workload.calibration))
    t_stack = time_best(lambda: workload.capture.stack(workload.calibration))
    print("", file=stream)
    print(f"  capture.reflectance (5 bands): {t_reflectance * 1e3:8.1f} ms", file=stream)
    print(f"  capture.stack       (5 bands): {t_stack * 1e3:8.1f} ms", file=stream)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def positive_float(text: str) -> float:
    """argparse type for a strictly positive finite tolerance."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not np.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("tolerance must be a positive finite number")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fingerprint rededgep outputs, verify numpy/native agreement and time "
            "the sampling backends."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tolerance",
        metavar="REL",
        type=positive_float,
        default=DEFAULT_TOLERANCE,
        help=(
            "relative tolerance for native-vs-numpy agreement; the absolute floor "
            f"is {ABSOLUTE_TOLERANCE:g}"
        ),
    )
    parser.add_argument(
        "--no-native",
        action="store_true",
        help="do not load or build the native kernel; fingerprint numpy only",
    )
    parser.add_argument(
        "--rebuild-native",
        action="store_true",
        help="force a fresh native build before running (useful after editing _native.c)",
    )
    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="skip the timing table (it goes to stderr in any case)",
    )
    return parser.parse_args(argv)


def prepare_native(args: argparse.Namespace) -> bool:
    """Load the native kernel, optionally rebuilding it; never abort on failure."""
    if args.no_native:
        return False
    if args.rebuild_native:
        try:
            native.build(force=True)
        except Exception as error:  # no compiler, no source, compile error
            print(f"# native rebuild failed: {error}", file=sys.stderr)
            print("# continuing without the native backend", file=sys.stderr)
            return False
    return native.available()


def environment_lines(native_ready: bool, args: argparse.Namespace) -> list[str]:
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    native_state = "disabled" if args.no_native else ("available" if native_ready else "unavailable")
    return [
        f"[env] rededgep={rp.__version__} python={python_version} numpy={np.__version__} native={native_state}",
        (
            f"[env] workload=band={BAND} width={WIDTH} height={HEIGHT} "
            f"images={','.join(IMAGE_NAMES)} points={','.join(str(n) for n in POINT_COUNTS)}"
        ),
        (
            f"[env] tolerance_rel={_num(args.tolerance, 6)} "
            f"tolerance_abs={_num(ABSOLUTE_TOLERANCE, 6)}"
        ),
        "# stdout is deterministic; timings are written to stderr. Diff two runs of stdout.",
        "# timing table: stderr, best of 3, excluded from this deterministic report.",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    native_ready = prepare_native(args)
    backends = ("numpy", "native") if native_ready else ("numpy",)

    workload = build_workload()

    lines = environment_lines(native_ready, args)
    if not native_ready:
        reason = "disabled by --no-native" if args.no_native else "native kernel unavailable"
        lines.append(f"[skip] cross-backend checks: {reason}")

    sample_lines, checks = collect_sample_fingerprints(
        workload, backends, args.tolerance, ABSOLUTE_TOLERANCE
    )
    lines.extend(sample_lines)
    lines.extend(collect_capture_fingerprints(workload))
    print("\n".join(lines), flush=True)

    if not args.no_timing:
        print_timings(workload, native_ready)

    total_combinations = len(IMAGE_NAMES) * len(POINT_COUNTS)
    skipped = total_combinations - len(checks)
    failed = [message for ok, message in checks if not ok]
    status = "PASS" if not failed else "FAIL"
    print(
        f"[summary] checks={len(checks)} passed={len(checks) - len(failed)} "
        f"failed={len(failed)} skipped={skipped} status={status}"
    )

    if failed:
        print("", file=sys.stderr)
        print(
            f"!!! EQUIVALENCE FAILED: {len(failed)} of {len(checks)} "
            "native-vs-numpy checks exceeded tolerance !!!",
            file=sys.stderr,
        )
        for message in failed:
            print(f"!!!   {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
