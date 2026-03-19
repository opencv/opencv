// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/core/utility.hpp>

// ONNX operator: GridSample
// Spec: https://onnx.ai/onnx/operators/onnx__GridSample.html
// Supported opsets: 16-22

namespace cv {
namespace dnn {

static inline float reflectCoord(float x, int limit, bool align_corners) {
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

static inline void getCubicCoeffs(float x, float A, float* coeffs )
{
    coeffs[0] = ((A*(x + 1.0f) - 5.0f*A)*(x + 1.0f) + 8.0f*A)*(x + 1.0f) - 4.0f*A;
    coeffs[1] = ((A + 2.0f)*x - (A + 3.0f))*x*x + 1.0f;
    coeffs[2] = ((A + 2.0f)*(1.0f - x) - (A + 3.0f))*(1.0f - x)*(1.0f - x) + 1.0f;
    coeffs[3] = 1.0f - coeffs[0] - coeffs[1] - coeffs[2];
}

static inline void computeNormToPixParams(int W, int H, bool align_corners,
                                           float& xscale, float& yscale,
                                           float& xdelta, float& ydelta)
{
    const float delta = align_corners ? 1.f : 0.f;
    xscale = 0.5f * (W - delta);
    yscale = 0.5f * (H - delta);
    xdelta = 0.5f * (W - delta) + 0.5f * (delta - 1.f);
    ydelta = 0.5f * (H - delta) + 0.5f * (delta - 1.f);
}

static inline void computeNormToPixParams3D(int W, int H, int D, bool align_corners,
                                            float& xscale, float& yscale, float& zscale,
                                            float& xdelta, float& ydelta, float& zdelta)
{
    const float delta = align_corners ? 1.f : 0.f;
    xscale = 0.5f * (W - delta);
    yscale = 0.5f * (H - delta);
    zscale = 0.5f * (D - delta);
    xdelta = 0.5f * (W - delta) + 0.5f * (delta - 1.f);
    ydelta = 0.5f * (H - delta) + 0.5f * (delta - 1.f);
    zdelta = 0.5f * (D - delta) + 0.5f * (delta - 1.f);
}

static inline void normToPix(float nx, float ny,
                             float xscale, float yscale,
                             float xdelta, float ydelta,
                             float& x, float& y)
{
    x = nx * xscale + xdelta;
    y = ny * yscale + ydelta;
}

static inline void normToPix3D(float nx, float ny, float nz,
                               float xscale, float yscale, float zscale,
                               float xdelta, float ydelta, float zdelta,
                               float& x, float& y, float& z)
{
    x = nx * xscale + xdelta;
    y = ny * yscale + ydelta;
    z = nz * zscale + zdelta;
}

enum Mode { M_NEAREST=0, M_BILINEAR=1, M_BICUBIC=2 };
enum Pad  { P_ZEROS=0,  P_BORDER=1,  P_REFLECTION=2 };

template<typename T, int MODE, int PAD>
static inline void gridSampleComputeRows(
        const T* Xptr,
        const float* Gptr,
        T* Yptr,
        int N, int C, int H, int W,
        int Hout, int Wout,
        bool align_corners,
        float cubic_alpha)
{
    size_t xNStride = (size_t)C * H * W;
    size_t xCStride = (size_t)H * W;
    size_t xHStride = (size_t)W;

    size_t gNStride = (size_t)Hout * Wout * 2;
    size_t gHStride = (size_t)Wout * 2;
    size_t gWStride = 2;

    size_t yNStride = (size_t)C * Hout * Wout;
    size_t yCStride = (size_t)Hout * Wout;
    size_t yHStride = (size_t)Wout;

    auto fetch = [&](const T* baseNC, int yy, int xx)->float {
        int px = xx, py = yy;
        if (PAD == P_BORDER) {
            px = saturate_cast<int>(std::min(float(W - 1), std::max(0.f, float(px))));
            py = saturate_cast<int>(std::min(float(H - 1), std::max(0.f, float(py))));
        } else if (PAD == P_REFLECTION) {
            px = saturate_cast<int>(std::floor(reflectCoord((float)px, W, align_corners) + 0.5f));
            py = saturate_cast<int>(std::floor(reflectCoord((float)py, H, align_corners) + 0.5f));
        }
        if (px < 0 || py < 0 || px >= W || py >= H) return 0.f;
        const T* p = baseNC + (size_t)py * xHStride + (size_t)px;
        return (float)(*p);
    };

    int nstripes = Hout * C * N;
    parallel_for_(Range(0, nstripes), [&](const Range& range) {
        for (int row = range.start; row < range.end; row++) {
            int n = row / (Hout * C);
            int c = (row - n * Hout * C) / Hout;
            int h = row % Hout;

            const T* baseNC = Xptr + n * xNStride + c * xCStride;
            size_t yRowBase = n * yNStride + c * yCStride + h * yHStride;
            size_t gRowBase = n * gNStride + h * gHStride;

            float xscale, yscale, xdelta, ydelta;
            computeNormToPixParams(W, H, align_corners, xscale, yscale, xdelta, ydelta);
            for (int w = 0; w < Wout; w++) {
                float nx = Gptr[gRowBase + w * gWStride + 0];
                float ny = Gptr[gRowBase + w * gWStride + 1];
                float xf, yf; normToPix(nx, ny, xscale, yscale, xdelta, ydelta, xf, yf);

                float outv = 0.f;
                if (MODE == M_NEAREST) {
                    int px = cvRound(xf);
                    int py = cvRound(yf);
                    outv = fetch(baseNC, py, px);
                } else if (MODE == M_BILINEAR) {
                    int x0 = saturate_cast<int>(floorf(xf));
                    int y0 = saturate_cast<int>(floorf(yf));
                    float dx = xf - x0, dy = yf - y0;

                    if (x0 >= 0 && y0 >= 0 && x0 + 1 < W && y0 + 1 < H)
                    {
                        const T* p_y0 = baseNC + (size_t)y0 * xHStride;
                        const T* p_y1 = baseNC + (size_t)(y0 + 1) * xHStride;
                        float v00 = (float)p_y0[x0];
                        float v01 = (float)p_y0[x0 + 1];
                        float v10 = (float)p_y1[x0];
                        float v11 = (float)p_y1[x0 + 1];
                        float vx0 = v00 * (1.f - dx) + v01 * dx;
                        float vx1 = v10 * (1.f - dx) + v11 * dx;
                        outv = vx0 * (1.f - dy) + vx1 * dy;
                    }
                    else
                    {
                        float v00 = fetch(baseNC, y0,     x0);
                        float v01 = fetch(baseNC, y0,     x0 + 1);
                        float v10 = fetch(baseNC, y0 + 1, x0);
                        float v11 = fetch(baseNC, y0 + 1, x0 + 1);
                        float vx0 = v00 * (1.f - dx) + v01 * dx;
                        float vx1 = v10 * (1.f - dx) + v11 * dx;
                        outv = vx0 * (1.f - dy) + vx1 * dy;
                    }
                } else {
                    const float alpha = cubic_alpha;
                    int x1 = saturate_cast<int>(floorf(xf));
                    int y1 = saturate_cast<int>(floorf(yf));
                    float tx = xf - x1, ty = yf - y1;

                    float wx[4], wy[4];
                    getCubicCoeffs(tx, alpha, wx);
                    getCubicCoeffs(ty, alpha, wy);

                    if (x1 >= 1 && y1 >= 1 && x1 + 2 < W && y1 + 2 < H) {
                        const T* p = baseNC + (size_t)(y1 - 1) * xHStride + (x1 - 1);
                        float v00 = (float)p[0], v01 = (float)p[1], v02 = (float)p[2], v03 = (float)p[3];
                        float rowv0 = v00 * wx[0] + v01 * wx[1] + v02 * wx[2] + v03 * wx[3];
                        float minv = std::min(std::min(v00, v01), std::min(v02, v03));
                        float maxv = std::max(std::max(v00, v01), std::max(v02, v03));
                        p += xHStride;
                        float v10 = (float)p[0], v11 = (float)p[1], v12 = (float)p[2], v13 = (float)p[3];
                        float rowv1 = v10 * wx[0] + v11 * wx[1] + v12 * wx[2] + v13 * wx[3];
                        minv = std::min(minv, std::min(std::min(v10, v11), std::min(v12, v13)));
                        maxv = std::max(maxv, std::max(std::max(v10, v11), std::max(v12, v13)));
                        p += xHStride;
                        float v20 = (float)p[0], v21 = (float)p[1], v22 = (float)p[2], v23 = (float)p[3];
                        float rowv2 = v20 * wx[0] + v21 * wx[1] + v22 * wx[2] + v23 * wx[3];
                        minv = std::min(minv, std::min(std::min(v20, v21), std::min(v22, v23)));
                        maxv = std::max(maxv, std::max(std::max(v20, v21), std::max(v22, v23)));
                        p += xHStride;
                        float v30 = (float)p[0], v31 = (float)p[1], v32 = (float)p[2], v33 = (float)p[3];
                        float rowv3 = v30 * wx[0] + v31 * wx[1] + v32 * wx[2] + v33 * wx[3];
                        minv = std::min(minv, std::min(std::min(v30, v31), std::min(v32, v33)));
                        maxv = std::max(maxv, std::max(std::max(v30, v31), std::max(v32, v33)));
                        outv = rowv0 * wy[0] + rowv1 * wy[1] + rowv2 * wy[2] + rowv3 * wy[3];
                        outv = std::max(minv, std::min(outv, maxv));
                    } else {
                        float a00 = fetch(baseNC, y1 - 1, x1 - 1), a01 = fetch(baseNC, y1 - 1, x1    ),
                              a02 = fetch(baseNC, y1 - 1, x1 + 1), a03 = fetch(baseNC, y1 - 1, x1 + 2);
                        float rowv0 = a00 * wx[0] + a01 * wx[1] + a02 * wx[2] + a03 * wx[3];
                        float minv = std::min(std::min(a00, a01), std::min(a02, a03));
                        float maxv = std::max(std::max(a00, a01), std::max(a02, a03));
                        float b00 = fetch(baseNC, y1,     x1 - 1), b01 = fetch(baseNC, y1,     x1    ),
                              b02 = fetch(baseNC, y1,     x1 + 1), b03 = fetch(baseNC, y1,     x1 + 2);
                        float rowv1 = b00 * wx[0] + b01 * wx[1] + b02 * wx[2] + b03 * wx[3];
                        minv = std::min(minv, std::min(std::min(b00, b01), std::min(b02, b03)));
                        maxv = std::max(maxv, std::max(std::max(b00, b01), std::max(b02, b03)));
                        float c00 = fetch(baseNC, y1 + 1, x1 - 1), c01 = fetch(baseNC, y1 + 1, x1    ),
                              c02 = fetch(baseNC, y1 + 1, x1 + 1), c03 = fetch(baseNC, y1 + 1, x1 + 2);
                        float rowv2 = c00 * wx[0] + c01 * wx[1] + c02 * wx[2] + c03 * wx[3];
                        minv = std::min(minv, std::min(std::min(c00, c01), std::min(c02, c03)));
                        maxv = std::max(maxv, std::max(std::max(c00, c01), std::max(c02, c03)));
                        float d00 = fetch(baseNC, y1 + 2, x1 - 1), d01 = fetch(baseNC, y1 + 2, x1    ),
                              d02 = fetch(baseNC, y1 + 2, x1 + 1), d03 = fetch(baseNC, y1 + 2, x1 + 2);
                        float rowv3 = d00 * wx[0] + d01 * wx[1] + d02 * wx[2] + d03 * wx[3];
                        minv = std::min(minv, std::min(std::min(d00, d01), std::min(d02, d03)));
                        maxv = std::max(maxv, std::max(std::max(d00, d01), std::max(d02, d03)));
                        outv = rowv0 * wy[0] + rowv1 * wy[1] + rowv2 * wy[2] + rowv3 * wy[3];
                        outv = std::max(minv, std::min(outv, maxv));
                    }
                }
                Yptr[yRowBase + w] = saturate_cast<T>(outv);
            }
        }
    });
}

template<typename T>
static inline void gridSampleDispatch(
        const T* Xptr,
        const float* Gptr,
        T* Yptr,
        int N, int C, int H, int W,
        int Hout, int Wout,
        bool align_corners,
        int mode, int padding,
        float cubic_alpha)
{
    if (mode == M_NEAREST) {
        if (padding == P_ZEROS) gridSampleComputeRows<T, M_NEAREST, P_ZEROS>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
        else if (padding == P_BORDER) gridSampleComputeRows<T, M_NEAREST, P_BORDER>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
        else gridSampleComputeRows<T, M_NEAREST, P_REFLECTION>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
    } else if (mode == M_BILINEAR) {
        if (padding == P_ZEROS) gridSampleComputeRows<T, M_BILINEAR, P_ZEROS>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
        else if (padding == P_BORDER) gridSampleComputeRows<T, M_BILINEAR, P_BORDER>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
        else gridSampleComputeRows<T, M_BILINEAR, P_REFLECTION>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
    } else {
        if (padding == P_ZEROS) gridSampleComputeRows<T, M_BICUBIC, P_ZEROS>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
        else if (padding == P_BORDER) gridSampleComputeRows<T, M_BICUBIC, P_BORDER>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
        else gridSampleComputeRows<T, M_BICUBIC, P_REFLECTION>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, cubic_alpha);
    }
}

template<typename T, int MODE, int PAD>
static inline void gridSampleCompute3D(
        const T* Xptr,
        const float* Gptr,
        T* Yptr,
        int N, int C, int D, int H, int W,
        int Dout, int Hout, int Wout,
        bool align_corners)
{
    size_t xNStride = (size_t)C * D * H * W;
    size_t xCStride = (size_t)D * H * W;
    size_t xDStride = (size_t)H * W;
    size_t xHStride = (size_t)W;

    size_t gNStride = (size_t)Dout * Hout * Wout * 3;
    size_t gDStride = (size_t)Hout * Wout * 3;
    size_t gHStride = (size_t)Wout * 3;
    size_t gWStride = 3;

    size_t yNStride = (size_t)C * Dout * Hout * Wout;
    size_t yCStride = (size_t)Dout * Hout * Wout;
    size_t yDStride = (size_t)Hout * Wout;
    size_t yHStride = (size_t)Wout;

    auto fetch3D = [&](const T* baseNC, int zz, int yy, int xx)->float {
        int px = xx, py = yy, pz = zz;
        if (PAD == P_BORDER) {
            px = saturate_cast<int>(std::min(float(W - 1), std::max(0.f, float(px))));
            py = saturate_cast<int>(std::min(float(H - 1), std::max(0.f, float(py))));
            pz = saturate_cast<int>(std::min(float(D - 1), std::max(0.f, float(pz))));
        } else if (PAD == P_REFLECTION) {
            px = saturate_cast<int>(std::floor(reflectCoord((float)px, W, align_corners) + 0.5f));
            py = saturate_cast<int>(std::floor(reflectCoord((float)py, H, align_corners) + 0.5f));
            pz = saturate_cast<int>(std::floor(reflectCoord((float)pz, D, align_corners) + 0.5f));
        }
        if (px < 0 || py < 0 || pz < 0 || px >= W || py >= H || pz >= D) return 0.f;
        const T* p = baseNC + (size_t)pz * xDStride + (size_t)py * xHStride + (size_t)px;
        return (float)(*p);
    };

    int nstripes = Dout * Hout * C * N;
    parallel_for_(Range(0, nstripes), [&](const Range& range) {
        for (int slice = range.start; slice < range.end; slice++) {
            int n = slice / (Dout * Hout * C);
            int c = (slice - n * Dout * Hout * C) / (Dout * Hout);
            int d = (slice - n * Dout * Hout * C - c * Dout * Hout) / Hout;
            int h = slice % Hout;

            const T* baseNC = Xptr + n * xNStride + c * xCStride;
            size_t yRowBase = n * yNStride + c * yCStride + d * yDStride + h * yHStride;
            size_t gRowBase = n * gNStride + d * gDStride + h * gHStride;

            float xscale, yscale, zscale, xdelta, ydelta, zdelta;
            computeNormToPixParams3D(W, H, D, align_corners, xscale, yscale, zscale, xdelta, ydelta, zdelta);
            for (int w = 0; w < Wout; w++) {
                float nx = Gptr[gRowBase + w * gWStride + 0];
                float ny = Gptr[gRowBase + w * gWStride + 1];
                float nz = Gptr[gRowBase + w * gWStride + 2];
                float xf, yf, zf; normToPix3D(nx, ny, nz, xscale, yscale, zscale, xdelta, ydelta, zdelta, xf, yf, zf);

                float outv = 0.f;
                if (MODE == M_NEAREST) {
                    int px = cvRound(xf);
                    int py = cvRound(yf);
                    int pz = cvRound(zf);
                    outv = fetch3D(baseNC, pz, py, px);
                } else { // trilinear
                    int x0 = saturate_cast<int>(floorf(xf));
                    int y0 = saturate_cast<int>(floorf(yf));
                    int z0 = saturate_cast<int>(floorf(zf));
                    float dx = xf - x0, dy = yf - y0, dz = zf - z0;

                    if (x0 >= 0 && y0 >= 0 && z0 >= 0 && x0 + 1 < W && y0 + 1 < H && z0 + 1 < D)
                    {
                        const T* pz0 = baseNC + (size_t)z0 * xDStride;
                        const T* pz1 = baseNC + (size_t)(z0 + 1) * xDStride;
                        const T* p00 = pz0 + (size_t)y0 * xHStride;
                        const T* p01 = pz0 + (size_t)(y0 + 1) * xHStride;
                        const T* p10 = pz1 + (size_t)y0 * xHStride;
                        const T* p11 = pz1 + (size_t)(y0 + 1) * xHStride;
                        float v000 = (float)p00[x0];
                        float v001 = (float)p00[x0 + 1];
                        float v010 = (float)p01[x0];
                        float v011 = (float)p01[x0 + 1];
                        float v100 = (float)p10[x0];
                        float v101 = (float)p10[x0 + 1];
                        float v110 = (float)p11[x0];
                        float v111 = (float)p11[x0 + 1];

                        float vx00 = v000 * (1.f - dx) + v001 * dx;
                        float vx01 = v010 * (1.f - dx) + v011 * dx;
                        float vx10 = v100 * (1.f - dx) + v101 * dx;
                        float vx11 = v110 * (1.f - dx) + v111 * dx;

                        float vxy0 = vx00 * (1.f - dy) + vx01 * dy;
                        float vxy1 = vx10 * (1.f - dy) + vx11 * dy;

                        outv = vxy0 * (1.f - dz) + vxy1 * dz;
                    }
                    else
                    {
                        float v000 = fetch3D(baseNC, z0,     y0,     x0);
                        float v001 = fetch3D(baseNC, z0,     y0,     x0 + 1);
                        float v010 = fetch3D(baseNC, z0,     y0 + 1, x0);
                        float v011 = fetch3D(baseNC, z0,     y0 + 1, x0 + 1);
                        float v100 = fetch3D(baseNC, z0 + 1, y0,     x0);
                        float v101 = fetch3D(baseNC, z0 + 1, y0,     x0 + 1);
                        float v110 = fetch3D(baseNC, z0 + 1, y0 + 1, x0);
                        float v111 = fetch3D(baseNC, z0 + 1, y0 + 1, x0 + 1);

                        float vx00 = v000 * (1.f - dx) + v001 * dx;
                        float vx01 = v010 * (1.f - dx) + v011 * dx;
                        float vx10 = v100 * (1.f - dx) + v101 * dx;
                        float vx11 = v110 * (1.f - dx) + v111 * dx;

                        float vxy0 = vx00 * (1.f - dy) + vx01 * dy;
                        float vxy1 = vx10 * (1.f - dy) + vx11 * dy;

                        outv = vxy0 * (1.f - dz) + vxy1 * dz;
                    }
                }
                Yptr[yRowBase + w] = saturate_cast<T>(outv);
            }
        }
    });
}

template<typename T>
static inline void gridSampleDispatch3D(
        const T* Xptr,
        const float* Gptr,
        T* Yptr,
        int N, int C, int D, int H, int W,
        int Dout, int Hout, int Wout,
        bool align_corners,
        int mode, int padding)
{
    if (mode == M_NEAREST) {
        if (padding == P_ZEROS) gridSampleCompute3D<T, M_NEAREST, P_ZEROS>(Xptr, Gptr, Yptr, N, C, D, H, W, Dout, Hout, Wout, align_corners);
        else if (padding == P_BORDER) gridSampleCompute3D<T, M_NEAREST, P_BORDER>(Xptr, Gptr, Yptr, N, C, D, H, W, Dout, Hout, Wout, align_corners);
        else gridSampleCompute3D<T, M_NEAREST, P_REFLECTION>(Xptr, Gptr, Yptr, N, C, D, H, W, Dout, Hout, Wout, align_corners);
    } else if (mode == M_BILINEAR) {
        if (padding == P_ZEROS) gridSampleCompute3D<T, M_BILINEAR, P_ZEROS>(Xptr, Gptr, Yptr, N, C, D, H, W, Dout, Hout, Wout, align_corners);
        else if (padding == P_BORDER) gridSampleCompute3D<T, M_BILINEAR, P_BORDER>(Xptr, Gptr, Yptr, N, C, D, H, W, Dout, Hout, Wout, align_corners);
        else gridSampleCompute3D<T, M_BILINEAR, P_REFLECTION>(Xptr, Gptr, Yptr, N, C, D, H, W, Dout, Hout, Wout, align_corners);
    } else {
        CV_Error(Error::StsNotImplemented, "GridSample bicubic mode is not supported for 5D inputs");
    }
}

class GridSampleLayerImpl CV_FINAL : public GridSampleLayer
{
public:
    int mode = M_BILINEAR;
    int padding = P_ZEROS;
    bool align_corners = false;
    float cubic_alpha = -0.75f;

    GridSampleLayerImpl(const LayerParams& params) {
        setParamsFrom(params);
        String m = params.get<String>("mode", "bilinear");
        String p = params.get<String>("padding_mode", "zeros");
        align_corners = params.get<bool>("align_corners", false);
        cubic_alpha = params.get<float>("cubic_coeff_a", -0.75f);

        if (m == "nearest") mode = M_NEAREST;
        else if (m == "bicubic") mode = M_BICUBIC;
        else mode = M_BILINEAR;

        if (p == "border") padding = P_BORDER;
        else if (p == "reflection") padding = P_REFLECTION;
        else padding = P_ZEROS;
    }

    bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        const MatShape& x = inputs[0];
        const MatShape& g = inputs[1];
        if (x.size() == 4) {
            CV_Assert(g.size() == 4 && g[3] == 2);
            outputs.assign(1, MatShape({x[0], x[1], g[1], g[2]}));
        } else if (x.size() == 5) {
            CV_Assert(g.size() == 5 && g[4] == 3);
            outputs.assign(1, MatShape({x[0], x[1], g[1], g[2], g[3]}));
        } else {
            CV_Assert(x.size() == 4 || x.size() == 5);
        }
        return false;
    }

    void getTypes(const std::vector<MatType>& inTypes,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outTypes,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inTypes.empty());
        outTypes.assign(requiredOutputs, inTypes[0]);
        internals.assign(requiredInternals, inTypes[0]);
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& X = inputs[0];
        const Mat& G = inputs[1];

        CV_Assert((X.dims == 4 && G.dims == 4 && G.size[3] == 2) || (X.dims == 5 && G.dims == 5 && G.size[4] == 3));
        const int N = X.size[0], C = X.size[1];
        const bool is3D = (X.dims == 5);
        const int D = is3D ? X.size[2] : 1;
        const int H = is3D ? X.size[3] : X.size[2];
        const int W = is3D ? X.size[4] : X.size[3];
        const int Dout = is3D ? G.size[1] : 1;
        const int Hout = is3D ? G.size[2] : G.size[1];
        const int Wout = is3D ? G.size[3] : G.size[2];

        Mat Gf = G;
        if (G.depth() != CV_32F) G.convertTo(Gf, CV_32F);

        const float* Gptr = (const float*)Gf.data;

        CV_Assert(!outputs.empty());
        int outType  = outputs[0].type();

        Mat Ytmp = is3D ? Mat({N, C, Dout, Hout, Wout}, outType)
                        : Mat({N, C, Hout, Wout}, outType);
        Ytmp.setTo(Scalar(0));

        switch (X.depth()) {
        case CV_8U: {
            const uchar* Xptr = X.ptr<uchar>();
            uchar* Yptr = Ytmp.ptr<uchar>();
            if (is3D) gridSampleDispatch3D<uchar>(Xptr, Gptr, Yptr, N, C, D, H, W, Dout, Hout, Wout, align_corners, mode, padding);
            else gridSampleDispatch<uchar>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, mode, padding, cubic_alpha);
            break;
        }
        case CV_16F: {
            const hfloat* Xptr = X.ptr<hfloat>();
            hfloat* Yptr = Ytmp.ptr<hfloat>();
            if (is3D) gridSampleDispatch3D<hfloat>(Xptr, Gptr, Yptr, N, C, D, H, W, Dout, Hout, Wout, align_corners, mode, padding);
            else gridSampleDispatch<hfloat>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, mode, padding, cubic_alpha);
            break;
        }
#ifdef CV_16BF
        case CV_16BF: {
            const bfloat* Xptr = X.ptr<bfloat>();
            bfloat* Yptr = Ytmp.ptr<bfloat>();
            if (is3D) gridSampleDispatch3D<bfloat>(Xptr, Gptr, Yptr, N, C, D, H, W, Dout, Hout, Wout, align_corners, mode, padding);
            else gridSampleDispatch<bfloat>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, mode, padding, cubic_alpha);
            break;
        }
#endif
        case CV_32F: {
            const float* Xptr = X.ptr<float>();
            float* Yptr = Ytmp.ptr<float>();
            if (is3D) gridSampleDispatch3D<float>(Xptr, Gptr, Yptr, N, C, D, H, W, Dout, Hout, Wout, align_corners, mode, padding);
            else gridSampleDispatch<float>(Xptr, Gptr, Yptr, N, C, H, W, Hout, Wout, align_corners, mode, padding, cubic_alpha);
            break;
        }
        default:
            CV_Error(Error::StsNotImplemented, "Unsupported input depth for GridSample");
        }

        Ytmp.copyTo(outputs[0]);
    }
};

Ptr<GridSampleLayer> GridSampleLayer::create(const LayerParams& params) {
    return Ptr<GridSampleLayer>(new GridSampleLayerImpl(params));
}

}}
