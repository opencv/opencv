// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <opencv2/core.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include <cfloat>
#include <algorithm>
#include <cmath>

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// padding values: 0 = zeros, 1 = border, 2 = reflection.

void gridSampleBilinear2D_f32_(
    const float* X, const float* G, float* Y,
    int N, int C, int H, int W, int Hout, int Wout,
    bool align_corners, int padding);

void gridSampleNearest2D_f32_(
    const float* X, const float* G, float* Y,
    int N, int C, int H, int W, int Hout, int Wout,
    bool align_corners, int padding);

void gridSampleBicubic2D_f32_(
    const float* X, const float* G, float* Y,
    int N, int C, int H, int W, int Hout, int Wout,
    bool align_corners, int padding, float cubic_alpha);

void gridSampleBilinear3D_f32_(
    const float* X, const float* G, float* Y,
    int N, int C, int D, int H, int W, int Dout, int Hout, int Wout,
    bool align_corners, int padding);

void gridSampleNearest3D_f32_(
    const float* X, const float* G, float* Y,
    int N, int C, int D, int H, int W, int Dout, int Hout, int Wout,
    bool align_corners, int padding);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

namespace gs_simd_internal {

enum { GS_ZEROS = 0, GS_BORDER = 1, GS_REFLECTION = 2 };

static inline float gs_reflect(float x, int limit, bool align_corners)
{
    if (limit <= 1) return 0.f;
    const float minv = align_corners ? 0.f : -0.5f;
    const float maxv = align_corners ? float(limit - 1) : float(limit) - 0.5f;
    const float m = maxv - minv;
    const float two_m = 2.f * m;
    float t = fmodf(x - minv, two_m);
    if (t < 0) t += two_m;
    if (t > m) t = two_m - t;
    return t + minv;
}

static inline void gs_cubic_coeffs(float x, float A, float* c)
{
    c[0] = ((A*(x + 1.0f) - 5.0f*A)*(x + 1.0f) + 8.0f*A)*(x + 1.0f) - 4.0f*A;
    c[1] = ((A + 2.0f)*x - (A + 3.0f))*x*x + 1.0f;
    c[2] = ((A + 2.0f)*(1.0f - x) - (A + 3.0f))*(1.0f - x)*(1.0f - x) + 1.0f;
    c[3] = 1.0f - c[0] - c[1] - c[2];
}

static inline void gs_xy_params(int W, int H, bool align_corners,
                                float& xs, float& ys, float& xd, float& yd)
{
    const float delta = align_corners ? 1.f : 0.f;
    xs = 0.5f * (W - delta);
    ys = 0.5f * (H - delta);
    xd = 0.5f * (W - delta) + 0.5f * (delta - 1.f);
    yd = 0.5f * (H - delta) + 0.5f * (delta - 1.f);
}

static inline void gs_xyz_params(int W, int H, int D, bool align_corners,
                                 float& xs, float& ys, float& zs,
                                 float& xd, float& yd, float& zd)
{
    const float delta = align_corners ? 1.f : 0.f;
    xs = 0.5f * (W - delta);
    ys = 0.5f * (H - delta);
    zs = 0.5f * (D - delta);
    xd = 0.5f * (W - delta) + 0.5f * (delta - 1.f);
    yd = 0.5f * (H - delta) + 0.5f * (delta - 1.f);
    zd = 0.5f * (D - delta) + 0.5f * (delta - 1.f);
}

template<int PAD>
static inline float gs_fetch2D(const float* baseNC, int yy, int xx,
                               int H, int W, size_t xHStride, bool align_corners)
{
    int xi = xx, yi = yy;
    if (PAD == GS_BORDER) {
        xi = std::max(0, std::min(W - 1, xi));
        yi = std::max(0, std::min(H - 1, yi));
    } else if (PAD == GS_REFLECTION) {
        xi = saturate_cast<int>(std::floor(gs_reflect((float)xi, W, align_corners) + 0.5f));
        yi = saturate_cast<int>(std::floor(gs_reflect((float)yi, H, align_corners) + 0.5f));
    }
    if (xi < 0 || yi < 0 || xi >= W || yi >= H) return 0.f;
    return baseNC[(size_t)yi * xHStride + (size_t)xi];
}

template<int PAD>
static inline float gs_fetch3D(const float* baseNC, int zz, int yy, int xx,
                               int D, int H, int W,
                               size_t xDStride, size_t xHStride, bool align_corners)
{
    int xi = xx, yi = yy, zi = zz;
    if (PAD == GS_BORDER) {
        xi = std::max(0, std::min(W - 1, xi));
        yi = std::max(0, std::min(H - 1, yi));
        zi = std::max(0, std::min(D - 1, zi));
    } else if (PAD == GS_REFLECTION) {
        xi = saturate_cast<int>(std::floor(gs_reflect((float)xi, W, align_corners) + 0.5f));
        yi = saturate_cast<int>(std::floor(gs_reflect((float)yi, H, align_corners) + 0.5f));
        zi = saturate_cast<int>(std::floor(gs_reflect((float)zi, D, align_corners) + 0.5f));
    }
    if (xi < 0 || yi < 0 || zi < 0 || xi >= W || yi >= H || zi >= D) return 0.f;
    return baseNC[(size_t)zi * xDStride + (size_t)yi * xHStride + (size_t)xi];
}

template<int PAD>
static inline void bilinear2D(const float* X, const float* G, float* Y,
                              int N, int C, int H, int W,
                              int Hout, int Wout, bool align_corners)
{
    const size_t xNStride = (size_t)C * H * W;
    const size_t xCStride = (size_t)H * W;
    const size_t xHStride = (size_t)W;
    const size_t gNStride = (size_t)Hout * Wout * 2;
    const size_t gHStride = (size_t)Wout * 2;
    const size_t yNStride = (size_t)C * Hout * Wout;
    const size_t yCStride = (size_t)Hout * Wout;
    const size_t yHStride = (size_t)Wout;

    float xscale, yscale, xdelta, ydelta;
    gs_xy_params(W, H, align_corners, xscale, yscale, xdelta, ydelta);

    parallel_for_(Range(0, N * Hout), [&](const Range& r) {
        for (int idx = r.start; idx < r.end; idx++) {
            int n = idx / Hout, h = idx - n * Hout;
            const float* baseN = X + (size_t)n * xNStride;
            const float* gRow = G + (size_t)n * gNStride + (size_t)h * gHStride;
            float* outBase = Y + (size_t)n * yNStride + (size_t)h * yHStride;

            for (int w = 0; w < Wout; w++) {
                float xf = gRow[w * 2 + 0] * xscale + xdelta;
                float yf = gRow[w * 2 + 1] * yscale + ydelta;
                int x0 = saturate_cast<int>(floorf(xf));
                int y0 = saturate_cast<int>(floorf(yf));
                float fx = xf - x0, fy = yf - y0;
                float w00 = (1.f - fx) * (1.f - fy);
                float w01 = fx * (1.f - fy);
                float w10 = (1.f - fx) * fy;
                float w11 = fx * fy;
                bool interior = (x0 >= 0 && y0 >= 0 && x0 + 1 < W && y0 + 1 < H);

                int c = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if (interior) {
                    const int L = VTraits<v_float32>::vlanes();
                    const size_t off00 = (size_t)y0 * xHStride + (size_t)x0;
                    const size_t off01 = off00 + 1;
                    const size_t off10 = off00 + xHStride;
                    const size_t off11 = off10 + 1;
                    const v_float32 vw00 = vx_setall_f32(w00);
                    const v_float32 vw01 = vx_setall_f32(w01);
                    const v_float32 vw10 = vx_setall_f32(w10);
                    const v_float32 vw11 = vx_setall_f32(w11);
                    for (; c + L <= C; c += L) {
                        float CV_DECL_ALIGNED(32) b00[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) b01[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) b10[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) b11[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) bo[VTraits<v_float32>::max_nlanes];
                        for (int k = 0; k < L; k++) {
                            const float* pNC = baseN + (size_t)(c + k) * xCStride;
                            b00[k] = pNC[off00];
                            b01[k] = pNC[off01];
                            b10[k] = pNC[off10];
                            b11[k] = pNC[off11];
                        }
                        v_float32 V00 = vx_load_aligned(b00);
                        v_float32 V01 = vx_load_aligned(b01);
                        v_float32 V10 = vx_load_aligned(b10);
                        v_float32 V11 = vx_load_aligned(b11);
                        v_float32 R = v_add(v_add(v_mul(V00, vw00), v_mul(V01, vw01)),
                                            v_add(v_mul(V10, vw10), v_mul(V11, vw11)));
                        v_store_aligned(bo, R);
                        for (int k = 0; k < L; k++) {
                            outBase[(size_t)(c + k) * yCStride + (size_t)w] = bo[k];
                        }
                    }
                }
#endif
                for (; c < C; c++) {
                    const float* baseNC = baseN + (size_t)c * xCStride;
                    float v00, v01, v10, v11;
                    if (interior) {
                        const float* py0 = baseNC + (size_t)y0 * xHStride;
                        const float* py1 = py0 + xHStride;
                        v00 = py0[x0]; v01 = py0[x0 + 1];
                        v10 = py1[x0]; v11 = py1[x0 + 1];
                    } else {
                        v00 = gs_fetch2D<PAD>(baseNC, y0,     x0,     H, W, xHStride, align_corners);
                        v01 = gs_fetch2D<PAD>(baseNC, y0,     x0 + 1, H, W, xHStride, align_corners);
                        v10 = gs_fetch2D<PAD>(baseNC, y0 + 1, x0,     H, W, xHStride, align_corners);
                        v11 = gs_fetch2D<PAD>(baseNC, y0 + 1, x0 + 1, H, W, xHStride, align_corners);
                    }
                    outBase[(size_t)c * yCStride + (size_t)w] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                }
            }
        }
    });
}

template<int PAD>
static inline void nearest2D(const float* X, const float* G, float* Y,
                             int N, int C, int H, int W,
                             int Hout, int Wout, bool align_corners)
{
    const size_t xNStride = (size_t)C * H * W;
    const size_t xCStride = (size_t)H * W;
    const size_t xHStride = (size_t)W;
    const size_t gNStride = (size_t)Hout * Wout * 2;
    const size_t gHStride = (size_t)Wout * 2;
    const size_t yNStride = (size_t)C * Hout * Wout;
    const size_t yCStride = (size_t)Hout * Wout;
    const size_t yHStride = (size_t)Wout;

    float xscale, yscale, xdelta, ydelta;
    gs_xy_params(W, H, align_corners, xscale, yscale, xdelta, ydelta);

    parallel_for_(Range(0, N * Hout), [&](const Range& r) {
        for (int idx = r.start; idx < r.end; idx++) {
            int n = idx / Hout, h = idx - n * Hout;
            const float* baseN = X + (size_t)n * xNStride;
            const float* gRow = G + (size_t)n * gNStride + (size_t)h * gHStride;
            float* outBase = Y + (size_t)n * yNStride + (size_t)h * yHStride;

            for (int w = 0; w < Wout; w++) {
                float xf = gRow[w * 2 + 0] * xscale + xdelta;
                float yf = gRow[w * 2 + 1] * yscale + ydelta;
                int xi = cvRound(xf);
                int yi = cvRound(yf);
                bool interior = (xi >= 0 && yi >= 0 && xi < W && yi < H);

                int c = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if (interior) {
                    const int L = VTraits<v_float32>::vlanes();
                    const size_t off = (size_t)yi * xHStride + (size_t)xi;
                    for (; c + L <= C; c += L) {
                        float CV_DECL_ALIGNED(32) b[VTraits<v_float32>::max_nlanes];
                        for (int k = 0; k < L; k++) {
                            b[k] = baseN[(size_t)(c + k) * xCStride + off];
                        }
                        for (int k = 0; k < L; k++) {
                            outBase[(size_t)(c + k) * yCStride + (size_t)w] = b[k];
                        }
                    }
                }
#endif
                for (; c < C; c++) {
                    const float* baseNC = baseN + (size_t)c * xCStride;
                    float v;
                    if (interior) v = baseNC[(size_t)yi * xHStride + (size_t)xi];
                    else v = gs_fetch2D<PAD>(baseNC, yi, xi, H, W, xHStride, align_corners);
                    outBase[(size_t)c * yCStride + (size_t)w] = v;
                }
            }
        }
    });
}

template<int PAD>
static inline void bicubic2D(const float* X, const float* G, float* Y,
                             int N, int C, int H, int W,
                             int Hout, int Wout, bool align_corners,
                             float cubic_alpha)
{
    const size_t xNStride = (size_t)C * H * W;
    const size_t xCStride = (size_t)H * W;
    const size_t xHStride = (size_t)W;
    const size_t gNStride = (size_t)Hout * Wout * 2;
    const size_t gHStride = (size_t)Wout * 2;
    const size_t yNStride = (size_t)C * Hout * Wout;
    const size_t yCStride = (size_t)Hout * Wout;
    const size_t yHStride = (size_t)Wout;

    float xscale, yscale, xdelta, ydelta;
    gs_xy_params(W, H, align_corners, xscale, yscale, xdelta, ydelta);

    parallel_for_(Range(0, N * Hout), [&](const Range& r) {
        for (int idx = r.start; idx < r.end; idx++) {
            int n = idx / Hout, h = idx - n * Hout;
            const float* baseN = X + (size_t)n * xNStride;
            const float* gRow = G + (size_t)n * gNStride + (size_t)h * gHStride;
            float* outBase = Y + (size_t)n * yNStride + (size_t)h * yHStride;

            for (int w = 0; w < Wout; w++) {
                float xf = gRow[w * 2 + 0] * xscale + xdelta;
                float yf = gRow[w * 2 + 1] * yscale + ydelta;
                int x1 = saturate_cast<int>(floorf(xf));
                int y1 = saturate_cast<int>(floorf(yf));
                float tx = xf - x1, ty = yf - y1;
                float wx[4], wy[4];
                gs_cubic_coeffs(tx, cubic_alpha, wx);
                gs_cubic_coeffs(ty, cubic_alpha, wy);
                bool interior = (x1 >= 1 && y1 >= 1 && x1 + 2 < W && y1 + 2 < H);

                int c = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if (interior) {
                    const int L = VTraits<v_float32>::vlanes();
                    const v_float32 vwx0 = vx_setall_f32(wx[0]);
                    const v_float32 vwx1 = vx_setall_f32(wx[1]);
                    const v_float32 vwx2 = vx_setall_f32(wx[2]);
                    const v_float32 vwx3 = vx_setall_f32(wx[3]);
                    const v_float32 vwy0 = vx_setall_f32(wy[0]);
                    const v_float32 vwy1 = vx_setall_f32(wy[1]);
                    const v_float32 vwy2 = vx_setall_f32(wy[2]);
                    const v_float32 vwy3 = vx_setall_f32(wy[3]);
                    for (; c + L <= C; c += L) {
                        float CV_DECL_ALIGNED(32) a[16][VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) bo[VTraits<v_float32>::max_nlanes];
                        for (int k = 0; k < L; k++) {
                            const float* pNC = baseN + (size_t)(c + k) * xCStride;
                            for (int j = 0; j < 4; j++) {
                                const float* p = pNC + (size_t)(y1 - 1 + j) * xHStride + (size_t)(x1 - 1);
                                a[j*4 + 0][k] = p[0];
                                a[j*4 + 1][k] = p[1];
                                a[j*4 + 2][k] = p[2];
                                a[j*4 + 3][k] = p[3];
                            }
                        }
                        v_float32 row0, row1, row2, row3;
                        v_float32 mn, mx;
                        {
                            v_float32 c0 = vx_load_aligned(a[0]);
                            v_float32 c1 = vx_load_aligned(a[1]);
                            v_float32 c2 = vx_load_aligned(a[2]);
                            v_float32 c3 = vx_load_aligned(a[3]);
                            row0 = v_add(v_add(v_mul(c0, vwx0), v_mul(c1, vwx1)),
                                         v_add(v_mul(c2, vwx2), v_mul(c3, vwx3)));
                            mn = v_min(v_min(c0, c1), v_min(c2, c3));
                            mx = v_max(v_max(c0, c1), v_max(c2, c3));
                        }
                        {
                            v_float32 c0 = vx_load_aligned(a[4]);
                            v_float32 c1 = vx_load_aligned(a[5]);
                            v_float32 c2 = vx_load_aligned(a[6]);
                            v_float32 c3 = vx_load_aligned(a[7]);
                            row1 = v_add(v_add(v_mul(c0, vwx0), v_mul(c1, vwx1)),
                                         v_add(v_mul(c2, vwx2), v_mul(c3, vwx3)));
                            mn = v_min(mn, v_min(v_min(c0, c1), v_min(c2, c3)));
                            mx = v_max(mx, v_max(v_max(c0, c1), v_max(c2, c3)));
                        }
                        {
                            v_float32 c0 = vx_load_aligned(a[8]);
                            v_float32 c1 = vx_load_aligned(a[9]);
                            v_float32 c2 = vx_load_aligned(a[10]);
                            v_float32 c3 = vx_load_aligned(a[11]);
                            row2 = v_add(v_add(v_mul(c0, vwx0), v_mul(c1, vwx1)),
                                         v_add(v_mul(c2, vwx2), v_mul(c3, vwx3)));
                            mn = v_min(mn, v_min(v_min(c0, c1), v_min(c2, c3)));
                            mx = v_max(mx, v_max(v_max(c0, c1), v_max(c2, c3)));
                        }
                        {
                            v_float32 c0 = vx_load_aligned(a[12]);
                            v_float32 c1 = vx_load_aligned(a[13]);
                            v_float32 c2 = vx_load_aligned(a[14]);
                            v_float32 c3 = vx_load_aligned(a[15]);
                            row3 = v_add(v_add(v_mul(c0, vwx0), v_mul(c1, vwx1)),
                                         v_add(v_mul(c2, vwx2), v_mul(c3, vwx3)));
                            mn = v_min(mn, v_min(v_min(c0, c1), v_min(c2, c3)));
                            mx = v_max(mx, v_max(v_max(c0, c1), v_max(c2, c3)));
                        }
                        v_float32 R = v_add(v_add(v_mul(row0, vwy0), v_mul(row1, vwy1)),
                                            v_add(v_mul(row2, vwy2), v_mul(row3, vwy3)));
                        R = v_max(mn, v_min(R, mx));
                        v_store_aligned(bo, R);
                        for (int k = 0; k < L; k++) {
                            outBase[(size_t)(c + k) * yCStride + (size_t)w] = bo[k];
                        }
                    }
                }
#endif
                for (; c < C; c++) {
                    const float* baseNC = baseN + (size_t)c * xCStride;
                    float a[4][4];
                    float minv = FLT_MAX, maxv = -FLT_MAX;
                    if (interior) {
                        const float* p = baseNC + (size_t)(y1 - 1) * xHStride + (size_t)(x1 - 1);
                        for (int j = 0; j < 4; j++) {
                            a[j][0] = p[0]; a[j][1] = p[1]; a[j][2] = p[2]; a[j][3] = p[3];
                            p += xHStride;
                        }
                    } else {
                        for (int j = 0; j < 4; j++) {
                            a[j][0] = gs_fetch2D<PAD>(baseNC, y1 - 1 + j, x1 - 1, H, W, xHStride, align_corners);
                            a[j][1] = gs_fetch2D<PAD>(baseNC, y1 - 1 + j, x1,     H, W, xHStride, align_corners);
                            a[j][2] = gs_fetch2D<PAD>(baseNC, y1 - 1 + j, x1 + 1, H, W, xHStride, align_corners);
                            a[j][3] = gs_fetch2D<PAD>(baseNC, y1 - 1 + j, x1 + 2, H, W, xHStride, align_corners);
                        }
                    }
                    float rowv[4];
                    for (int j = 0; j < 4; j++) {
                        rowv[j] = a[j][0] * wx[0] + a[j][1] * wx[1] + a[j][2] * wx[2] + a[j][3] * wx[3];
                        for (int i = 0; i < 4; i++) {
                            minv = std::min(minv, a[j][i]);
                            maxv = std::max(maxv, a[j][i]);
                        }
                    }
                    float outv = rowv[0] * wy[0] + rowv[1] * wy[1] + rowv[2] * wy[2] + rowv[3] * wy[3];
                    outv = std::max(minv, std::min(outv, maxv));
                    outBase[(size_t)c * yCStride + (size_t)w] = outv;
                }
            }
        }
    });
}

template<int PAD>
static inline void bilinear3D(const float* X, const float* G, float* Y,
                              int N, int C, int D, int H, int W,
                              int Dout, int Hout, int Wout, bool align_corners)
{
    const size_t xNStride = (size_t)C * D * H * W;
    const size_t xCStride = (size_t)D * H * W;
    const size_t xDStride = (size_t)H * W;
    const size_t xHStride = (size_t)W;
    const size_t gNStride = (size_t)Dout * Hout * Wout * 3;
    const size_t gDStride = (size_t)Hout * Wout * 3;
    const size_t gHStride = (size_t)Wout * 3;
    const size_t yNStride = (size_t)C * Dout * Hout * Wout;
    const size_t yCStride = (size_t)Dout * Hout * Wout;
    const size_t yDStride = (size_t)Hout * Wout;
    const size_t yHStride = (size_t)Wout;

    float xscale, yscale, zscale, xdelta, ydelta, zdelta;
    gs_xyz_params(W, H, D, align_corners, xscale, yscale, zscale, xdelta, ydelta, zdelta);

    parallel_for_(Range(0, N * Dout * Hout), [&](const Range& r) {
        for (int idx = r.start; idx < r.end; idx++) {
            int n = idx / (Dout * Hout);
            int rem = idx - n * Dout * Hout;
            int d = rem / Hout;
            int h = rem - d * Hout;
            const float* baseN = X + (size_t)n * xNStride;
            const float* gRow = G + (size_t)n * gNStride + (size_t)d * gDStride + (size_t)h * gHStride;
            float* outBase = Y + (size_t)n * yNStride + (size_t)d * yDStride + (size_t)h * yHStride;

            for (int w = 0; w < Wout; w++) {
                float xf = gRow[w * 3 + 0] * xscale + xdelta;
                float yf = gRow[w * 3 + 1] * yscale + ydelta;
                float zf = gRow[w * 3 + 2] * zscale + zdelta;
                int x0 = saturate_cast<int>(floorf(xf));
                int y0 = saturate_cast<int>(floorf(yf));
                int z0 = saturate_cast<int>(floorf(zf));
                float dx = xf - x0, dy = yf - y0, dz = zf - z0;
                float w000 = (1.f - dx) * (1.f - dy) * (1.f - dz);
                float w001 = dx        * (1.f - dy) * (1.f - dz);
                float w010 = (1.f - dx) * dy        * (1.f - dz);
                float w011 = dx        * dy        * (1.f - dz);
                float w100 = (1.f - dx) * (1.f - dy) * dz;
                float w101 = dx        * (1.f - dy) * dz;
                float w110 = (1.f - dx) * dy        * dz;
                float w111 = dx        * dy        * dz;
                bool interior = (x0 >= 0 && y0 >= 0 && z0 >= 0 &&
                                 x0 + 1 < W && y0 + 1 < H && z0 + 1 < D);

                int c = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if (interior) {
                    const int L = VTraits<v_float32>::vlanes();
                    const size_t off000 = (size_t)z0 * xDStride + (size_t)y0 * xHStride + (size_t)x0;
                    const size_t off001 = off000 + 1;
                    const size_t off010 = off000 + xHStride;
                    const size_t off011 = off010 + 1;
                    const size_t off100 = off000 + xDStride;
                    const size_t off101 = off100 + 1;
                    const size_t off110 = off100 + xHStride;
                    const size_t off111 = off110 + 1;
                    const v_float32 vw000 = vx_setall_f32(w000);
                    const v_float32 vw001 = vx_setall_f32(w001);
                    const v_float32 vw010 = vx_setall_f32(w010);
                    const v_float32 vw011 = vx_setall_f32(w011);
                    const v_float32 vw100 = vx_setall_f32(w100);
                    const v_float32 vw101 = vx_setall_f32(w101);
                    const v_float32 vw110 = vx_setall_f32(w110);
                    const v_float32 vw111 = vx_setall_f32(w111);
                    for (; c + L <= C; c += L) {
                        float CV_DECL_ALIGNED(32) b000[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) b001[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) b010[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) b011[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) b100[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) b101[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) b110[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) b111[VTraits<v_float32>::max_nlanes];
                        float CV_DECL_ALIGNED(32) bo[VTraits<v_float32>::max_nlanes];
                        for (int k = 0; k < L; k++) {
                            const float* pNC = baseN + (size_t)(c + k) * xCStride;
                            b000[k] = pNC[off000]; b001[k] = pNC[off001];
                            b010[k] = pNC[off010]; b011[k] = pNC[off011];
                            b100[k] = pNC[off100]; b101[k] = pNC[off101];
                            b110[k] = pNC[off110]; b111[k] = pNC[off111];
                        }
                        v_float32 V000 = vx_load_aligned(b000);
                        v_float32 V001 = vx_load_aligned(b001);
                        v_float32 V010 = vx_load_aligned(b010);
                        v_float32 V011 = vx_load_aligned(b011);
                        v_float32 V100 = vx_load_aligned(b100);
                        v_float32 V101 = vx_load_aligned(b101);
                        v_float32 V110 = vx_load_aligned(b110);
                        v_float32 V111 = vx_load_aligned(b111);
                        v_float32 R = v_add(
                            v_add(v_add(v_mul(V000, vw000), v_mul(V001, vw001)),
                                  v_add(v_mul(V010, vw010), v_mul(V011, vw011))),
                            v_add(v_add(v_mul(V100, vw100), v_mul(V101, vw101)),
                                  v_add(v_mul(V110, vw110), v_mul(V111, vw111))));
                        v_store_aligned(bo, R);
                        for (int k = 0; k < L; k++) {
                            outBase[(size_t)(c + k) * yCStride + (size_t)w] = bo[k];
                        }
                    }
                }
#endif
                for (; c < C; c++) {
                    const float* baseNC = baseN + (size_t)c * xCStride;
                    float v000, v001, v010, v011, v100, v101, v110, v111;
                    if (interior) {
                        const float* pz0 = baseNC + (size_t)z0 * xDStride;
                        const float* pz1 = pz0 + xDStride;
                        const float* p00 = pz0 + (size_t)y0 * xHStride;
                        const float* p01 = pz0 + (size_t)(y0 + 1) * xHStride;
                        const float* p10 = pz1 + (size_t)y0 * xHStride;
                        const float* p11 = pz1 + (size_t)(y0 + 1) * xHStride;
                        v000 = p00[x0]; v001 = p00[x0 + 1];
                        v010 = p01[x0]; v011 = p01[x0 + 1];
                        v100 = p10[x0]; v101 = p10[x0 + 1];
                        v110 = p11[x0]; v111 = p11[x0 + 1];
                    } else {
                        v000 = gs_fetch3D<PAD>(baseNC, z0,     y0,     x0,     D, H, W, xDStride, xHStride, align_corners);
                        v001 = gs_fetch3D<PAD>(baseNC, z0,     y0,     x0 + 1, D, H, W, xDStride, xHStride, align_corners);
                        v010 = gs_fetch3D<PAD>(baseNC, z0,     y0 + 1, x0,     D, H, W, xDStride, xHStride, align_corners);
                        v011 = gs_fetch3D<PAD>(baseNC, z0,     y0 + 1, x0 + 1, D, H, W, xDStride, xHStride, align_corners);
                        v100 = gs_fetch3D<PAD>(baseNC, z0 + 1, y0,     x0,     D, H, W, xDStride, xHStride, align_corners);
                        v101 = gs_fetch3D<PAD>(baseNC, z0 + 1, y0,     x0 + 1, D, H, W, xDStride, xHStride, align_corners);
                        v110 = gs_fetch3D<PAD>(baseNC, z0 + 1, y0 + 1, x0,     D, H, W, xDStride, xHStride, align_corners);
                        v111 = gs_fetch3D<PAD>(baseNC, z0 + 1, y0 + 1, x0 + 1, D, H, W, xDStride, xHStride, align_corners);
                    }
                    outBase[(size_t)c * yCStride + (size_t)w] =
                        w000 * v000 + w001 * v001 + w010 * v010 + w011 * v011 +
                        w100 * v100 + w101 * v101 + w110 * v110 + w111 * v111;
                }
            }
        }
    });
}

template<int PAD>
static inline void nearest3D(const float* X, const float* G, float* Y,
                             int N, int C, int D, int H, int W,
                             int Dout, int Hout, int Wout, bool align_corners)
{
    const size_t xNStride = (size_t)C * D * H * W;
    const size_t xCStride = (size_t)D * H * W;
    const size_t xDStride = (size_t)H * W;
    const size_t xHStride = (size_t)W;
    const size_t gNStride = (size_t)Dout * Hout * Wout * 3;
    const size_t gDStride = (size_t)Hout * Wout * 3;
    const size_t gHStride = (size_t)Wout * 3;
    const size_t yNStride = (size_t)C * Dout * Hout * Wout;
    const size_t yCStride = (size_t)Dout * Hout * Wout;
    const size_t yDStride = (size_t)Hout * Wout;
    const size_t yHStride = (size_t)Wout;

    float xscale, yscale, zscale, xdelta, ydelta, zdelta;
    gs_xyz_params(W, H, D, align_corners, xscale, yscale, zscale, xdelta, ydelta, zdelta);

    parallel_for_(Range(0, N * Dout * Hout), [&](const Range& r) {
        for (int idx = r.start; idx < r.end; idx++) {
            int n = idx / (Dout * Hout);
            int rem = idx - n * Dout * Hout;
            int d = rem / Hout;
            int h = rem - d * Hout;
            const float* baseN = X + (size_t)n * xNStride;
            const float* gRow = G + (size_t)n * gNStride + (size_t)d * gDStride + (size_t)h * gHStride;
            float* outBase = Y + (size_t)n * yNStride + (size_t)d * yDStride + (size_t)h * yHStride;

            for (int w = 0; w < Wout; w++) {
                float xf = gRow[w * 3 + 0] * xscale + xdelta;
                float yf = gRow[w * 3 + 1] * yscale + ydelta;
                float zf = gRow[w * 3 + 2] * zscale + zdelta;
                int xi = cvRound(xf), yi = cvRound(yf), zi = cvRound(zf);
                bool interior = (xi >= 0 && yi >= 0 && zi >= 0 && xi < W && yi < H && zi < D);

                int c = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
                if (interior) {
                    const int L = VTraits<v_float32>::vlanes();
                    const size_t off = (size_t)zi * xDStride + (size_t)yi * xHStride + (size_t)xi;
                    for (; c + L <= C; c += L) {
                        float CV_DECL_ALIGNED(32) b[VTraits<v_float32>::max_nlanes];
                        for (int k = 0; k < L; k++) {
                            b[k] = baseN[(size_t)(c + k) * xCStride + off];
                        }
                        for (int k = 0; k < L; k++) {
                            outBase[(size_t)(c + k) * yCStride + (size_t)w] = b[k];
                        }
                    }
                }
#endif
                for (; c < C; c++) {
                    const float* baseNC = baseN + (size_t)c * xCStride;
                    float v;
                    if (interior) v = baseNC[(size_t)zi * xDStride + (size_t)yi * xHStride + (size_t)xi];
                    else v = gs_fetch3D<PAD>(baseNC, zi, yi, xi, D, H, W, xDStride, xHStride, align_corners);
                    outBase[(size_t)c * yCStride + (size_t)w] = v;
                }
            }
        }
    });
}

} // namespace gs_simd_internal

void gridSampleBilinear2D_f32_(const float* X, const float* G, float* Y,
                               int N, int C, int H, int W, int Hout, int Wout,
                               bool align_corners, int padding)
{
    using namespace gs_simd_internal;
    if (padding == GS_ZEROS)        bilinear2D<GS_ZEROS>(X, G, Y, N, C, H, W, Hout, Wout, align_corners);
    else if (padding == GS_BORDER)  bilinear2D<GS_BORDER>(X, G, Y, N, C, H, W, Hout, Wout, align_corners);
    else                            bilinear2D<GS_REFLECTION>(X, G, Y, N, C, H, W, Hout, Wout, align_corners);
}

void gridSampleNearest2D_f32_(const float* X, const float* G, float* Y,
                              int N, int C, int H, int W, int Hout, int Wout,
                              bool align_corners, int padding)
{
    using namespace gs_simd_internal;
    if (padding == GS_ZEROS)        nearest2D<GS_ZEROS>(X, G, Y, N, C, H, W, Hout, Wout, align_corners);
    else if (padding == GS_BORDER)  nearest2D<GS_BORDER>(X, G, Y, N, C, H, W, Hout, Wout, align_corners);
    else                            nearest2D<GS_REFLECTION>(X, G, Y, N, C, H, W, Hout, Wout, align_corners);
}

void gridSampleBicubic2D_f32_(const float* X, const float* G, float* Y,
                              int N, int C, int H, int W, int Hout, int Wout,
                              bool align_corners, int padding, float cubic_alpha)
{
    using namespace gs_simd_internal;
    if (padding == GS_ZEROS)        bicubic2D<GS_ZEROS>(X, G, Y, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
    else if (padding == GS_BORDER)  bicubic2D<GS_BORDER>(X, G, Y, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
    else                            bicubic2D<GS_REFLECTION>(X, G, Y, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
}

void gridSampleBilinear3D_f32_(const float* X, const float* G, float* Y,
                               int N, int C, int D, int H, int W, int Dout, int Hout, int Wout,
                               bool align_corners, int padding)
{
    using namespace gs_simd_internal;
    if (padding == GS_ZEROS)        bilinear3D<GS_ZEROS>(X, G, Y, N, C, D, H, W, Dout, Hout, Wout, align_corners);
    else if (padding == GS_BORDER)  bilinear3D<GS_BORDER>(X, G, Y, N, C, D, H, W, Dout, Hout, Wout, align_corners);
    else                            bilinear3D<GS_REFLECTION>(X, G, Y, N, C, D, H, W, Dout, Hout, Wout, align_corners);
}

void gridSampleNearest3D_f32_(const float* X, const float* G, float* Y,
                              int N, int C, int D, int H, int W, int Dout, int Hout, int Wout,
                              bool align_corners, int padding)
{
    using namespace gs_simd_internal;
    if (padding == GS_ZEROS)        nearest3D<GS_ZEROS>(X, G, Y, N, C, D, H, W, Dout, Hout, Wout, align_corners);
    else if (padding == GS_BORDER)  nearest3D<GS_BORDER>(X, G, Y, N, C, D, H, W, Dout, Hout, Wout, align_corners);
    else                            nearest3D<GS_REFLECTION>(X, G, Y, N, C, D, H, W, Dout, Hout, Wout, align_corners);
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}  // cv::dnn

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
