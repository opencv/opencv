// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

// Fused MergeMertens stages, compiled once per dispatch mode (baseline and AVX512_ICL).
//   mertensConvertGray_: CV_8UC3 * (1/255) and COLOR_RGB2GRAY in one pass.
//   mertensPyrUpSub_:    cv::pyrUp (BORDER_REFLECT_101) subtracted from the fine image.
//   mertensBlendAcc_:    interleaved img * weight, stored or added into the accumulator.

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void mertensConvertGray_(const uchar* src, size_t srcStep, float* img, size_t imgStep,
                         float* gray, size_t grayStep, int rows, int cols);
void mertensPyrUpSub_(const float* src, size_t srcStep, float* dst, size_t dstStep,
                      int srcRows, int srcCols, int dstRows, int dstCols, int channels);
void mertensBlendAcc_(const float* img, size_t imgStep, const float* weight, size_t weightStep,
                      float* acc, size_t accStep, int rows, int cols, int channels, bool accumulate);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

static inline const float* mertensRow(const float* base, size_t step, int y)
{
    return reinterpret_cast<const float*>(reinterpret_cast<const uchar*>(base) + (size_t)y * step);
}
static inline float* mertensRow(float* base, size_t step, int y)
{
    return reinterpret_cast<float*>(reinterpret_cast<uchar*>(base) + (size_t)y * step);
}

void mertensConvertGray_(const uchar* src, size_t srcStep, float* img, size_t imgStep,
                         float* gray, size_t grayStep, int rows, int cols)
{
    const float scale = static_cast<float>(1.0 / 255.0);
    const float kr = 0.299f, kg = 0.587f, kb = 0.114f;

    for (int y = 0; y < rows; y++)
    {
        const uchar* s = src + (size_t)y * srcStep;
        float* d = mertensRow(img, imgStep, y);
        float* g = mertensRow(gray, grayStep, y);
        int x = 0;
#if CV_SIMD || CV_SIMD_SCALABLE
        const int n = VTraits<v_float32>::vlanes();
        const int n8 = VTraits<v_uint8>::vlanes();
        if (n8 == 4 * n)
        {
            const v_float32 vscale = vx_setall_f32(scale);
            const v_float32 vkr = vx_setall_f32(kr);
            const v_float32 vkg = vx_setall_f32(kg);
            const v_float32 vkb = vx_setall_f32(kb);
            for (; x <= cols - n8; x += n8)
            {
                v_uint8 c0, c1, c2;
                v_load_deinterleave(s + x * 3, c0, c1, c2);
                v_uint16 c0a, c0b, c1a, c1b, c2a, c2b;
                v_expand(c0, c0a, c0b);
                v_expand(c1, c1a, c1b);
                v_expand(c2, c2a, c2b);
                v_uint32 c0_0, c0_1, c0_2, c0_3;
                v_uint32 c1_0, c1_1, c1_2, c1_3;
                v_uint32 c2_0, c2_1, c2_2, c2_3;
                v_expand(c0a, c0_0, c0_1);
                v_expand(c0b, c0_2, c0_3);
                v_expand(c1a, c1_0, c1_1);
                v_expand(c1b, c1_2, c1_3);
                v_expand(c2a, c2_0, c2_1);
                v_expand(c2b, c2_2, c2_3);
                v_float32 f0[4] = {
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c0_0)), vscale),
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c0_1)), vscale),
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c0_2)), vscale),
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c0_3)), vscale)
                };
                v_float32 f1[4] = {
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c1_0)), vscale),
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c1_1)), vscale),
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c1_2)), vscale),
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c1_3)), vscale)
                };
                v_float32 f2[4] = {
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c2_0)), vscale),
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c2_1)), vscale),
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c2_2)), vscale),
                    v_mul(v_cvt_f32(v_reinterpret_as_s32(c2_3)), vscale)
                };
                for (int k = 0; k < 4; k++)
                {
                    const int xk = x + k * n;
                    v_store_interleave(d + xk * 3, f0[k], f1[k], f2[k]);
                    v_store(g + xk, v_fma(f2[k], vkb, v_fma(f1[k], vkg, v_mul(f0[k], vkr))));
                }
            }
        }
#endif
        for (; x < cols; x++)
        {
            const float c0 = s[x * 3] * scale;
            const float c1 = s[x * 3 + 1] * scale;
            const float c2 = s[x * 3 + 2] * scale;
            d[x * 3] = c0;
            d[x * 3 + 1] = c1;
            d[x * 3 + 2] = c2;
            g[x] = c0 * kr + c1 * kg + c2 * kb;
        }
    }
}

void mertensBlendAcc_(const float* img, size_t imgStep, const float* weight, size_t weightStep,
                      float* acc, size_t accStep, int rows, int cols, int channels, bool accumulate)
{
    for (int y = 0; y < rows; y++)
    {
        const float* s = mertensRow(img, imgStep, y);
        const float* w = mertensRow(weight, weightStep, y);
        float* d = mertensRow(acc, accStep, y);
        int x = 0;
#if CV_SIMD || CV_SIMD_SCALABLE
        const int n = VTraits<v_float32>::vlanes();
        if (channels == 3)
        {
            for (; x <= cols - n; x += n)
            {
                v_float32 c0, c1, c2;
                v_load_deinterleave(s + x * 3, c0, c1, c2);
                const v_float32 vw = vx_load(w + x);
                c0 = v_mul(c0, vw);
                c1 = v_mul(c1, vw);
                c2 = v_mul(c2, vw);
                if (accumulate)
                {
                    v_float32 a0, a1, a2;
                    v_load_deinterleave(d + x * 3, a0, a1, a2);
                    c0 = v_add(c0, a0);
                    c1 = v_add(c1, a1);
                    c2 = v_add(c2, a2);
                }
                v_store_interleave(d + x * 3, c0, c1, c2);
            }
        }
        else
        {
            for (; x <= cols - n; x += n)
            {
                v_float32 v = v_mul(vx_load(s + x), vx_load(w + x));
                if (accumulate)
                    v = v_add(v, vx_load(d + x));
                v_store(d + x, v);
            }
        }
#endif
        if (channels == 3)
        {
            for (; x < cols; x++)
            {
                const float wv = w[x];
                float c0 = s[x * 3] * wv;
                float c1 = s[x * 3 + 1] * wv;
                float c2 = s[x * 3 + 2] * wv;
                if (accumulate)
                {
                    c0 += d[x * 3];
                    c1 += d[x * 3 + 1];
                    c2 += d[x * 3 + 2];
                }
                d[x * 3] = c0;
                d[x * 3 + 1] = c1;
                d[x * 3 + 2] = c2;
            }
        }
        else
        {
            for (; x < cols; x++)
            {
                float v = s[x] * w[x];
                if (accumulate)
                    v += d[x];
                d[x] = v;
            }
        }
    }
}

// Horizontal pyrUp convolution. Matches cv::pyrUp_ for float, cn = 1 or 3.
static void mertensPyrUpHorizontal(const float* src, float* row, int srcCols, int dstCols, int cn)
{
    const int sw = srcCols * cn;
    const int dw = dstCols * cn;
    if (srcCols == 1)
    {
        for (int c = 0; c < cn; c++)
            row[c] = row[c + cn] = src[c] * 8.f;
        return;
    }

    for (int c = 0; c < cn; c++)
    {
        const float t0 = src[c] * 6.f + src[c + cn] * 2.f;
        const float t1 = (src[c] + src[c + cn]) * 4.f;
        row[c] = t0;
        row[c + cn] = t1;

        const int sx = sw - cn + c;
        const int dx = (srcCols - 1) * 2 * cn + c;
        const float u0 = src[sx - cn] + src[sx] * 7.f;
        const float u1 = src[sx] * 8.f;
        row[dx] = u0;
        row[dx + cn] = u1;
        if (dstCols > srcCols * 2)
            row[(dstCols - 1) * cn + c] = u1;
    }

    int p = 1;
#if CV_SIMD || CV_SIMD_SCALABLE
    const int n = VTraits<v_float32>::vlanes();
    const v_float32 six = vx_setall_f32(6.f);
    const v_float32 four = vx_setall_f32(4.f);
    if (cn == 1)
    {
        for (; p <= srcCols - n - 1; p += n)
        {
            const v_float32 c = vx_load(src + p);
            const v_float32 r = vx_load(src + p + 1);
            const v_float32 l = vx_load(src + p - 1);
            const v_float32 t0 = v_add(v_add(l, r), v_mul(c, six));
            const v_float32 t1 = v_mul(v_add(c, r), four);
            v_store_interleave(row + p * 2, t0, t1);
        }
    }
    else if (cn == 3 && n <= 32)
    {
        AutoBuffer<float> even(n * 3), odd(n * 3);
        for (; p <= srcCols - n - 1; p += n)
        {
            v_float32 l0, l1, l2, c0, c1, c2, r0, r1, r2;
            v_load_deinterleave(src + (p - 1) * 3, l0, l1, l2);
            v_load_deinterleave(src + p * 3, c0, c1, c2);
            v_load_deinterleave(src + (p + 1) * 3, r0, r1, r2);
            v_float32 t00 = v_add(v_add(l0, r0), v_mul(c0, six));
            v_float32 t01 = v_add(v_add(l1, r1), v_mul(c1, six));
            v_float32 t02 = v_add(v_add(l2, r2), v_mul(c2, six));
            v_float32 t10 = v_mul(v_add(c0, r0), four);
            v_float32 t11 = v_mul(v_add(c1, r1), four);
            v_float32 t12 = v_mul(v_add(c2, r2), four);
            v_store_interleave(even.data(), t00, t01, t02);
            v_store_interleave(odd.data(), t10, t11, t12);
            for (int i = 0; i < n; i++)
            {
                float* o = row + (p + i) * 6;
                o[0] = even[i * 3];
                o[1] = even[i * 3 + 1];
                o[2] = even[i * 3 + 2];
                o[3] = odd[i * 3];
                o[4] = odd[i * 3 + 1];
                o[5] = odd[i * 3 + 2];
            }
        }
    }
#endif
    for (; p < srcCols - 1; p++)
    {
        for (int c = 0; c < cn; c++)
        {
            const int x = p * cn + c;
            const float t0 = src[x - cn] + src[x] * 6.f + src[x + cn];
            const float t1 = (src[x] + src[x + cn]) * 4.f;
            row[p * 2 * cn + c] = t0;
            row[p * 2 * cn + cn + c] = t1;
        }
    }
    CV_UNUSED(dw);
}

static void mertensPyrUpVertical(const float* row0, const float* row1, const float* row2,
                                 float* dst0, float* dst1, int width)
{
    const float scale = static_cast<float>(1.0 / 64.0);
    int x = 0;
    if (dst0 != dst1)
    {
#if CV_SIMD || CV_SIMD_SCALABLE
        const int n = VTraits<v_float32>::vlanes();
        const v_float32 six = vx_setall_f32(6.f);
        const v_float32 four = vx_setall_f32(4.f);
        const v_float32 vscale = vx_setall_f32(scale);
        for (; x <= width - n; x += n)
        {
            const v_float32 r0 = vx_load(row0 + x);
            const v_float32 r1 = vx_load(row1 + x);
            const v_float32 r2 = vx_load(row2 + x);
            const v_float32 t0 = v_mul(v_add(v_add(r0, r2), v_mul(r1, six)), vscale);
            const v_float32 t1 = v_mul(v_mul(v_add(r1, r2), four), vscale);
            v_store(dst0 + x, v_sub(vx_load(dst0 + x), t0));
            v_store(dst1 + x, v_sub(vx_load(dst1 + x), t1));
        }
#endif
        for (; x < width; x++)
        {
            const float t0 = (row0[x] + row1[x] * 6.f + row2[x]) * scale;
            const float t1 = (row1[x] + row2[x]) * 4.f * scale;
            dst0[x] -= t0;
            dst1[x] -= t1;
        }
    }
    else
    {
#if CV_SIMD || CV_SIMD_SCALABLE
        const int n = VTraits<v_float32>::vlanes();
        const v_float32 six = vx_setall_f32(6.f);
        const v_float32 vscale = vx_setall_f32(scale);
        for (; x <= width - n; x += n)
        {
            const v_float32 r0 = vx_load(row0 + x);
            const v_float32 r1 = vx_load(row1 + x);
            const v_float32 r2 = vx_load(row2 + x);
            const v_float32 t0 = v_mul(v_add(v_add(r0, r2), v_mul(r1, six)), vscale);
            v_store(dst0 + x, v_sub(vx_load(dst0 + x), t0));
        }
#endif
        for (; x < width; x++)
            dst0[x] -= (row0[x] + row1[x] * 6.f + row2[x]) * scale;
    }
}

void mertensPyrUpSub_(const float* src, size_t srcStep, float* dst, size_t dstStep,
                      int srcRows, int srcCols, int dstRows, int dstCols, int channels)
{
    const int cn = channels;
    const int width = dstCols * cn;
    const int bufstep = (int)alignSize((dstCols + 1) * cn, 16);
    AutoBuffer<float> storage(bufstep * 3 + 16);
    float* buf = alignPtr(storage.data(), 16);
    float* rows[3];

    int sy = -1;
    for (int y = 0; y < srcRows; y++)
    {
        for (; sy <= y + 1; sy++)
        {
            float* row = buf + ((sy + 1) % 3) * bufstep;
            const int syb = borderInterpolate(sy * 2, srcRows * 2, BORDER_REFLECT_101) / 2;
            mertensPyrUpHorizontal(mertensRow(src, srcStep, syb), row, srcCols, dstCols, cn);
        }
        for (int k = 0; k < 3; k++)
            rows[k] = buf + ((y - 1 + k + 1) % 3) * bufstep;

        float* dst0 = mertensRow(dst, dstStep, y * 2);
        float* dst1 = mertensRow(dst, dstStep, std::min(y * 2 + 1, dstRows - 1));
        mertensPyrUpVertical(rows[0], rows[1], rows[2], dst0, dst1, width);
    }

    if (dstRows > srcRows * 2)
    {
        const float* from = mertensRow(dst, dstStep, srcRows * 2 - 2);
        float* to = mertensRow(dst, dstStep, srcRows * 2);
        for (int x = 0; x < width; x++)
            to[x] = from[x];
    }
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
