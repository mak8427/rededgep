/*
 * Fused RedEdge-P sampling kernel.
 *
 * The numpy path performs projection, distortion, bilinear interpolation, vignetting and
 * the radiometric conversion as separate array operations, allocating temporaries at each
 * step.  For scattered ground points that dominates the runtime.  This kernel does all of
 * it in a single pass over the points, touching each input once and writing only the
 * result.
 *
 * Build:  gcc -O3 -march=native -shared -fPIC -o librededgep.so _native.c -lm
 */
#include <math.h>
#include <stddef.h>

/*
 * rp_sample_reflectance
 * ---------------------
 * dn          : DN image, h*w, row-major
 * east/north/up: local ENU offsets of each ground point from the camera
 * rot         : 3x3 row-major rotation taking ENU to camera coordinates
 * vig_c       : vignetting coefficients, paired with integer exponents vig_i / vig_j
 * scale       : a1 / (gain * exposure * 2^16)
 * irradiance  : horizontal irradiance in W/m^2/nm
 * out         : reflectance (NaN where the point does not land on the image)
 * valid       : 1 where out is valid, 0 otherwise
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
    const double half_w = 0.5 * (double)w;
    const double half_h = 0.5 * (double)h;
    const double max_u = (double)(w - 1);
    const double max_v = (double)(h - 1);
    long i;
    long good = 0;
    int max_i = 0, max_j = 0;
    long t;

    for (t = 0; t < nvig; t++) {
        if (vig_i[t] > max_i) max_i = vig_i[t];
        if (vig_j[t] > max_j) max_j = vig_j[t];
    }
    if (max_i > 7) max_i = 7;
    if (max_j > 7) max_j = 7;

    for (i = 0; i < n; i++) {
        const double e = east[i];
        const double nn = north[i];
        const double uu = up[i];

        const double cam_x = rot[0] * e + rot[1] * nn + rot[2] * uu;
        const double cam_y = rot[3] * e + rot[4] * nn + rot[5] * uu;
        const double cam_z = rot[6] * e + rot[7] * nn + rot[8] * uu;
        const double depth = -cam_z;

        if (!(depth > 1e-9)) {
            out[i] = NAN;
            valid[i] = 0;
            continue;
        }

        const double x = cam_x / depth;
        const double y = cam_y / depth;
        const double r2 = x * x + y * y;
        const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        const double xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
        const double yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;

        const double px = half_w + focal * xd + cx;
        const double py = half_h - focal * yd + cy;

        if (!(px >= 0.0 && px <= max_u && py >= 0.0 && py <= max_v)) {
            out[i] = NAN;
            valid[i] = 0;
            continue;
        }

        /* bilinear sample */
        const long x0 = (long)floor(px);
        const long y0 = (long)floor(py);
        const long x1 = x0 + 1 < w ? x0 + 1 : w - 1;
        const long y1 = y0 + 1 < h ? y0 + 1 : h - 1;
        const double fx = px - (double)x0;
        const double fy = py - (double)y0;
        const double s00 = (double)dn[y0 * w + x0];
        const double s01 = (double)dn[y0 * w + x1];
        const double s10 = (double)dn[y1 * w + x0];
        const double s11 = (double)dn[y1 * w + x1];
        const double value = s00 * (1.0 - fx) * (1.0 - fy) + s01 * fx * (1.0 - fy)
                           + s10 * (1.0 - fx) * fy + s11 * fx * fy;

        /* vignetting, normalised to unit mean.
         * Exponents are small integers, so build the power tables once per point
         * instead of calling pow() or a loop for every term. */
        const double xn = 2.0 * (px + 0.5) / (double)w - 1.0;
        const double yn = 2.0 * (py + 0.5) / (double)h - 1.0;
        double xp[8], yp[8];
        xp[0] = 1.0; yp[0] = 1.0;
        {
            int p;
            for (p = 1; p <= max_i; p++) xp[p] = xp[p - 1] * xn;
            for (p = 1; p <= max_j; p++) yp[p] = yp[p - 1] * yn;
        }
        double exponent = 0.0;
        long k;
        for (k = 0; k < nvig; k++) {
            exponent += vig_c[k] * xp[vig_i[k]] * yp[vig_j[k]];
        }
        const double weight = exp(exponent) / vig_mean;

        double corrected = value - black;
        if (corrected < 0.0) {
            corrected = 0.0;
        }
        out[i] = (float)(M_PI * scale * (corrected / weight) / irradiance);
        valid[i] = 1;
        good++;
    }
    return good;
}

const char *rp_version(void) { return "0.1.0"; }
