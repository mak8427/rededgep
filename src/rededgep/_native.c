/*
 * Fused RedEdge-P sampling kernel.
 *
 * The numpy path performs projection, distortion, bilinear interpolation, vignetting and
 * the radiometric conversion as separate array operations, allocating temporaries at each
 * step.  For scattered ground points that dominates the runtime.  This kernel does all of
 * it in a single pass over the points, touching each input once and writing only the
 * result.
 *
 * Two entry points share the same per-point evaluator:
 *
 *   rp_sample_reflectance
 *       Legacy interface: the caller supplies local east/north/up offsets, i.e. it has
 *       already done the geodetic -> ECEF -> ENU conversion in numpy.  Kept so existing
 *       ctypes bindings keep working; writes float32.
 *
 *   rp_sample_reflectance_geodetic
 *       Accepts lon/lat/height and performs the geodetic -> ECEF conversion and the
 *       ECEF -> camera rotation here, for every point, in the same pass.  The camera
 *       dependent work (ECEF of the camera, the ENU basis, the combined ECEF -> camera
 *       rotation) is hoisted out of the loop.  Writes float64, which is what the numpy
 *       backend produces.
 *
 * Build:  gcc -O3 -march=native -shared -fPIC -o librededgep.so _native.c -lm
 *
 * Optional: compile with -fopenmp and set REDEDGEP_THREADS to parallelise over points.
 * The default is one thread, so results and performance are deterministic.
 */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* WGS84 ellipsoid, matching rededgep.geometry. */
#define RP_A 6378137.0
#define RP_F (1.0 / 298.257223563)
#define RP_E2 (RP_F * (2.0 - RP_F))
#define RP_DEG2RAD (M_PI / 180.0)

/* Largest vignetting exponent supported by the compact coefficient table. */
#define RP_MAX_DEG 7

/* Points are processed in blocks of this many; see rp_sample_reflectance_geodetic. */
#define RP_BLOCK 64

/*
 * exp() without a libm call.
 *
 * Vignetting needs one exponential per point.  The call itself measured ~7 ms per
 * million points, more than the inline evaluation below costs, so the kernel evaluates
 * it here with the usual argument reduction
 *
 *     exp(x) = 2^(k/2) * exp(v),  v = (x log2 e - k/2) ln 2,  k = round(2 x log2 e)
 *
 * which keeps |v| <= ln(2)/4 = 0.173.  A degree-10 Taylor series is then accurate to
 * 7e-17 relative (< 0.5 ulp) and the half-integer split - the multiply by sqrt(2) for
 * odd k - is what makes such a short series enough.
 *
 * Measured against libm exp(): max relative error 2e-15 over x in [-30, 30] and 6e-16
 * over the range a real vignetting polynomial produces (|x| < 2).  Arguments that could
 * overflow (|x| >= 700, only reachable with a corrupt calibration) defer to libm so
 * that infinities and NaN behave exactly as before.
 */
static inline double rp_exp(double x)
{
    const double log2e = 1.4426950408889634;   /* log2(e) */
    const double ln2 = 0.6931471805599453;
    union { unsigned long long u; double d; } scale;
    double t, v, v2, v4, v8, p;
    int k, half;

    if (!(x > -700.0 && x < 700.0)) {
        return exp(x);
    }
    t = x * log2e;
    k = (int)nearbyint(2.0 * t);
    v = (t - 0.5 * (double)k) * ln2;

    /* exp(v) = sum v^n / n!, grouped by Estrin to keep the dependency chain short. */
    v2 = v * v;
    v4 = v2 * v2;
    v8 = v4 * v4;
    p = ((1.0 + v) + v2 * (0.5 + v * (1.0 / 6.0)))
      + v4 * (((1.0 / 24.0) + v * (1.0 / 120.0))
              + v2 * ((1.0 / 720.0) + v * (1.0 / 5040.0)))
      + v8 * (((1.0 / 40320.0) + v * (1.0 / 362880.0)) + v2 * (1.0 / 3628800.0));

    /* 2^(k/2) = 2^(k>>1) * (sqrt(2) if k is odd).  The scaling is done by building the
     * exponent bits directly; ldexp() would be a function call per point.  |x| < 700
     * keeps k>>1 inside [-1023, 1023], so the biased exponent is always in range. */
    half = k >> 1;
    if (k & 1) {
        p *= 1.4142135623730951;   /* sqrt(2) */
    }
    scale.u = (unsigned long long)(half + 1023) << 52;
    return p * scale.d;
}

/*
 * Force the compiler to round a product to double before a following add or subtract.
 *
 * rededgep.geometry computes `ecef - centre` with every intermediate rounded to double.
 * With -ffp-contract=fast (the default) gcc fuses `a * b - c` into an FMA, which rounds
 * once instead of twice.  That is normally the better choice, and it is used everywhere
 * else in this file; but here the operands are ~6.4e6 m and the difference is only
 * ~200 m, so a different rounding of the product is worth ~0.5 nm of ground position,
 * which in turn moves a sampled reflectance by ~1e-9.  Blocking the fusion on these few
 * expressions keeps the C and numpy backends within a few 1e-15 of each other instead
 * of a few 1e-9.
 *
 * The empty asm is a pure scheduling barrier - it emits no instruction.
 */
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
static inline double rp_rounded(double x)
{
    __asm__("" : "+x"(x));
    return x;
}
#else
static inline double rp_rounded(double x)
{
    volatile double rounded = x;
    return rounded;
}
#endif

/* Everything that is constant for the whole call. */
typedef struct {
    const unsigned short *dn;
    int h, w;
    double half_w, half_h, max_u, max_v;
    double focal, cx, cy;
    double k1, k2, k3, p1, p2;
    double black;    /* black level subtracted from DN */
    double gain;     /* pi * scale * vignetting_mean / irradiance */
    double xn_a, xn_b; /* normalised vignetting coordinate: xn = xn_a * px + xn_b */
    double yn_a, yn_b;
    int deg_i, deg_j;
    double vig[RP_MAX_DEG + 1][RP_MAX_DEG + 1];
} rp_ctx;

/* Thread count from $REDEDGEP_THREADS, defaulting to 1 (deterministic, no surprises).
 * Read on every call so applications and benchmarks can change it at runtime. */
static int rp_threads(void)
{
    const char *text = getenv("REDEDGEP_THREADS");
    int n;

    if (text == NULL) {
        return 1;
    }
    n = atoi(text);
    return n < 1 ? 1 : n;
}

static void rp_ctx_init(rp_ctx *ctx, int h, int w,
                        double focal, double cx, double cy,
                        double k1, double k2, double k3, double p1, double p2,
                        const double *vig_c, const int *vig_i, const int *vig_j, long nvig,
                        double vig_mean,
                        double black, double scale, double irradiance)
{
    long t;
    int i, j;

    ctx->h = h;
    ctx->w = w;
    ctx->half_w = 0.5 * (double)w;
    ctx->half_h = 0.5 * (double)h;
    ctx->max_u = (double)(w - 1);
    ctx->max_v = (double)(h - 1);
    ctx->focal = focal;
    ctx->cx = cx;
    ctx->cy = cy;
    ctx->k1 = k1;
    ctx->k2 = k2;
    ctx->k3 = k3;
    ctx->p1 = p1;
    ctx->p2 = p2;
    ctx->black = black;
    ctx->gain = M_PI * scale * vig_mean / irradiance;
    /* x = 2 * (col + 0.5) / w - 1, with col == px. */
    ctx->xn_a = 2.0 / (double)w;
    ctx->xn_b = 1.0 / (double)w - 1.0;
    ctx->yn_a = 2.0 / (double)h;
    ctx->yn_b = 1.0 / (double)h - 1.0;

    /* Compact 4x4-style table + Horner evaluation: one multiply-add per coefficient
     * instead of pow()/power tables and an indirect index per term. */
    for (i = 0; i <= RP_MAX_DEG; i++) {
        for (j = 0; j <= RP_MAX_DEG; j++) {
            ctx->vig[i][j] = 0.0;
        }
    }
    ctx->deg_i = 0;
    ctx->deg_j = 0;
    for (t = 0; t < nvig; t++) {
        i = vig_i[t];
        j = vig_j[t];
        if (i < 0 || j < 0 || i > RP_MAX_DEG || j > RP_MAX_DEG) {
            continue;  /* same silent clamp as the numpy path's power table */
        }
        ctx->vig[i][j] += vig_c[t];
        if (i > ctx->deg_i) ctx->deg_i = i;
        if (j > ctx->deg_j) ctx->deg_j = j;
    }
}

/* sum_ij c_ij x^i y^j, Horner in x over polynomials in y. */
static inline double rp_vignetting_exponent(const rp_ctx *ctx, double x, double y)
{
    double acc;
    int i, j;

    if (ctx->deg_i <= 3 && ctx->deg_j <= 3) {
        /* The RedEdge-P polynomials are 3x3; unrolled, this is 15 multiply-adds. */
        double a = ((ctx->vig[0][3] * y + ctx->vig[0][2]) * y + ctx->vig[0][1]) * y + ctx->vig[0][0];
        double b = ((ctx->vig[1][3] * y + ctx->vig[1][2]) * y + ctx->vig[1][1]) * y + ctx->vig[1][0];
        double c = ((ctx->vig[2][3] * y + ctx->vig[2][2]) * y + ctx->vig[2][1]) * y + ctx->vig[2][0];
        double d = ((ctx->vig[3][3] * y + ctx->vig[3][2]) * y + ctx->vig[3][1]) * y + ctx->vig[3][0];
        return ((d * x + c) * x + b) * x + a;
    }
    acc = 0.0;
    for (i = ctx->deg_i; i >= 0; i--) {
        double poly = 0.0;
        for (j = ctx->deg_j; j >= 0; j--) {
            poly = poly * y + ctx->vig[i][j];
        }
        acc = acc * x + poly;
    }
    return acc;
}

/*
 * Evaluate one point already expressed in camera coordinates (x right, y up, z back).
 * Returns the reflectance, or NaN when the point does not land on the image.
 */
static inline double rp_camera_point(const rp_ctx *ctx, double cam_x, double cam_y, double cam_z)
{
    const double depth = -cam_z;
    double x, y, r2, radial, xd, yd, px, py, value, xn, yn;

    if (!(depth > 1e-9)) {
        return NAN;
    }

    /* Pinhole projection plus Brown-Conrady distortion. */
    x = cam_x / depth;
    y = cam_y / depth;
    r2 = x * x + y * y;
    radial = 1.0 + r2 * (ctx->k1 + r2 * (ctx->k2 + r2 * ctx->k3));
    xd = x * radial + 2.0 * ctx->p1 * x * y + ctx->p2 * (r2 + 2.0 * x * x);
    yd = y * radial + ctx->p1 * (r2 + 2.0 * y * y) + 2.0 * ctx->p2 * x * y;

    px = ctx->half_w + ctx->focal * xd + ctx->cx;
    py = ctx->half_h - ctx->focal * yd + ctx->cy;

    if (!(px >= 0.0 && px <= ctx->max_u && py >= 0.0 && py <= ctx->max_v)) {
        return NAN;
    }

    /* Bilinear sample.  px, py are non-negative here, so truncation is floor(). */
    {
        const long x0 = (long)px;
        const long y0 = (long)py;
        const long x1 = x0 + 1 < ctx->w ? x0 + 1 : ctx->w - 1;
        const long y1 = y0 + 1 < ctx->h ? y0 + 1 : ctx->h - 1;
        const double fx = px - (double)x0;
        const double fy = py - (double)y0;
        const unsigned short *row0 = ctx->dn + (size_t)y0 * (size_t)ctx->w;
        const unsigned short *row1 = ctx->dn + (size_t)y1 * (size_t)ctx->w;
        const double s00 = (double)row0[x0];
        const double s01 = (double)row0[x1];
        const double s10 = (double)row1[x0];
        const double s11 = (double)row1[x1];
        value = (s00 * (1.0 - fx) + s01 * fx) * (1.0 - fy)
              + (s10 * (1.0 - fx) + s11 * fx) * fy;
    }

    /* Vignetting, normalised to unit mean and divided out: 1/weight = mean(V)/V.
     * Multiplication by exp(-exponent) avoids both a division and a log.  The dark
     * level is removed first, exactly like the numpy chain. */
    value -= ctx->black;
    if (value < 0.0) {
        value = 0.0;
    }
    xn = ctx->xn_a * px + ctx->xn_b;
    yn = ctx->yn_a * py + ctx->yn_b;
    value *= rp_exp(-rp_vignetting_exponent(ctx, xn, yn));
    return ctx->gain * value;
}

/*
 * rp_sample_reflectance
 * ---------------------
 * Legacy entry point: points already projected to local ENU offsets from the camera.
 *
 * dn            : DN image, h*w, row-major
 * east/north/up : local ENU offsets of each ground point from the camera
 * rot           : 3x3 row-major rotation taking ENU to camera coordinates
 * vig_c         : vignetting coefficients, paired with integer exponents vig_i / vig_j
 * scale         : a1 / (gain * exposure * 2^16)
 * irradiance    : horizontal irradiance in W/m^2/nm
 * out           : reflectance (NaN where the point does not land on the image)
 * valid         : 1 where out is valid, 0 otherwise
 *
 * Returns the number of valid points.
 */
long rp_sample_reflectance(
    const unsigned short *dn, int h, int w,
    const double *east, const double *north, const double *up, long n,
    const double *rot,
    double focal, double cx, double cy,
    double k1, double k2, double k3, double p1, double p2,
    const double *vig_c, const int *vig_i, const int *vig_j, long nvig,
    double vig_mean,
    double black, double scale, double irradiance,
    float *out, unsigned char *valid)
{
    rp_ctx ctx;
    long i;
    long good = 0;

    rp_ctx_init(&ctx, h, w, focal, cx, cy, k1, k2, k3, p1, p2,
                vig_c, vig_i, vig_j, nvig, vig_mean, black, scale, irradiance);
    ctx.dn = dn;

    for (i = 0; i < n; i++) {
        const double e = east[i];
        const double nn = north[i];
        const double uu = up[i];
        const double cam_x = rot[0] * e + rot[1] * nn + rot[2] * uu;
        const double cam_y = rot[3] * e + rot[4] * nn + rot[5] * uu;
        const double cam_z = rot[6] * e + rot[7] * nn + rot[8] * uu;
        const double value = rp_camera_point(&ctx, cam_x, cam_y, cam_z);

        out[i] = (float)value;
        valid[i] = value == value ? 1 : 0;
        good += value == value ? 1 : 0;
    }
    return good;
}

/*
 * rp_sample_reflectance_geodetic
 * ------------------------------
 * Accepts geodetic ground points and does the coordinate conversion here.
 *
 * lon/lat   : degrees
 * height    : metres above the ellipsoid
 * rot       : 3x3 row-major rotation taking local ENU to camera coordinates
 * out       : float64 reflectance (NaN where the point does not land on the image)
 *
 * Returns the number of valid points.
 */
long rp_sample_reflectance_geodetic(
    const unsigned short *dn, int h, int w,
    const double *lon, const double *lat, const double *height, long n,
    double cam_lon, double cam_lat, double cam_alt,
    const double *rot,
    double focal, double cx, double cy,
    double k1, double k2, double k3, double p1, double p2,
    const double *vig_c, const int *vig_i, const int *vig_j, long nvig,
    double vig_mean,
    double black, double scale, double irradiance,
    double *out)
{
    rp_ctx ctx;
    double cam_lat_rad, cam_lon_rad;
    double enu[3][3];   /* rows: east, north, up in ECEF */
    double m[3][3];     /* camera <- ECEF, i.e. rot * enu */
    double centre[3];   /* camera position in ECEF */
    double n_cam;
    long i;
    long good = 0;
    const int threads = rp_threads();
    const long nblocks = n > 0 ? (n + RP_BLOCK - 1) / RP_BLOCK : 0;
    long block;

    rp_ctx_init(&ctx, h, w, focal, cx, cy, k1, k2, k3, p1, p2,
                vig_c, vig_i, vig_j, nvig, vig_mean, black, scale, irradiance);
    ctx.dn = dn;

    /* Geodetic -> ECEF of the camera, and the ENU basis at the camera, both once. */
    cam_lat_rad = cam_lat * RP_DEG2RAD;
    cam_lon_rad = cam_lon * RP_DEG2RAD;
    {
        double s1, c1, s2, c2, q;
        s1 = sin(cam_lat_rad);
        c1 = cos(cam_lat_rad);
        s2 = sin(cam_lon_rad);
        c2 = cos(cam_lon_rad);
        q = 1.0 - RP_E2 * (s1 * s1);
        n_cam = RP_A / sqrt(q);
        centre[0] = (n_cam + cam_alt) * c1 * c2;
        centre[1] = (n_cam + cam_alt) * c1 * s2;
        centre[2] = (n_cam * (1.0 - RP_E2) + cam_alt) * s1;

        enu[0][0] = -s2;          enu[0][1] = c2;           enu[0][2] = 0.0;
        enu[1][0] = -s1 * c2;     enu[1][1] = -s1 * s2;     enu[1][2] = c1;
        enu[2][0] = c1 * c2;      enu[2][1] = c1 * s2;      enu[2][2] = s1;
    }
    for (i = 0; i < 3; i++) {
        int k;
        for (k = 0; k < 3; k++) {
            m[i][k] = rot[3 * i + 0] * enu[0][k]
                    + rot[3 * i + 1] * enu[1][k]
                    + rot[3 * i + 2] * enu[2][k];
        }
    }

    /* Points are processed in blocks of RP_BLOCK: each block is first reduced to the
     * sin/cos of its latitudes and longitudes in a tight loop, then everything else is
     * computed from those values in a second loop that makes no library calls.
     *
     * That is not cosmetic.  sin/cos and exp are the only out-of-line calls left, and
     * every call forces the compiler to spill the 16 caller-saved SSE registers.  With
     * the calls interleaved in the main loop gcc spilled ~200 live doubles per unrolled
     * iteration and the kernel ran ~35 % slower than the same arithmetic with the calls
     * moved out.  A block is 4 * RP_BLOCK doubles, i.e. 2 KiB, so it stays in L1 and
     * both loops schedule (and partly vectorise) cleanly. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1) \
    reduction(+ : good)
#endif
    for (block = 0; block < nblocks; block++) {
        const long start = block * RP_BLOCK;
        const long end = start + RP_BLOCK < n ? start + RP_BLOCK : n;
        double sph[RP_BLOCK], cph[RP_BLOCK], slm[RP_BLOCK], clm[RP_BLOCK];
        long j;

        for (j = start; j < end; j++) {
            const double lat_rad = lat[j] * RP_DEG2RAD;
            const double lon_rad = lon[j] * RP_DEG2RAD;
            const long b = j - start;
            sph[b] = sin(lat_rad);
            cph[b] = cos(lat_rad);
            slm[b] = sin(lon_rad);
            clm[b] = cos(lon_rad);
        }

        for (j = start; j < end; j++) {
            const long b = j - start;
            const double s = sph[b];
            const double c = cph[b];
            double q, nn, px, py, pz, value;

            /* Geodetic -> ECEF, minus the camera position: the ENU vector in ECEF.
             * The parenthesisation and the rp_rounded() barriers mirror
             * rededgep.geometry.lla_to_ecef, so that the two backends agree to a few
             * 1e-15 instead of a few 1e-9 once the vector is sampled. */
            q = 1.0 - rp_rounded(RP_E2 * (s * s));
            nn = RP_A / sqrt(q);
            px = rp_rounded((nn + height[j]) * c * clm[b]) - centre[0];
            py = rp_rounded((nn + height[j]) * c * slm[b]) - centre[1];
            pz = rp_rounded((rp_rounded(nn * (1.0 - RP_E2)) + height[j]) * s) - centre[2];

            value = rp_camera_point(&ctx,
                                    m[0][0] * px + m[0][1] * py + m[0][2] * pz,
                                    m[1][0] * px + m[1][1] * py + m[1][2] * pz,
                                    m[2][0] * px + m[2][1] * py + m[2][2] * pz);
            out[j] = value;
            good += value == value ? 1 : 0;
        }
    }
    return good;
}

const char *rp_version(void) { return "0.1.0"; }
