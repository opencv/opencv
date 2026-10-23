// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

// MergeMertens weight row, compiled once per dispatch mode (baseline and AVX512_ICL).
// Fuses cv::Laplacian ksize=1 (BORDER_REFLECT_101), abs, and the weight:
//   weight = pow(|lap|, wcon) * pow(saturation, wsat) * pow(well-exposedness, wexp) + 1e-12.
// grayUp/grayMid/grayDn are the three gray rows; the caller reflects the vertical border.
// Horizontal borders are applied here. Saturation is applied only for 3-channel images.
// (wcon, wsat, wexp) == (1, 1, 0) skips pow and the exposure exp.

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void mertensWeightRow_(const float* grayUp, const float* grayMid, const float* grayDn,
                       const float* src, float* dst, int cols, int channels,
                       float wcon, float wsat, float wexp, float* sum);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// |g[y-1][x] + g[y+1][x] + g[y][x-1] + g[y][x+1] - 4*g[y][x]|, with reflect-101 in x.
static inline float mertensContrastAt(const float* up, const float* mid, const float* dn, int x, int cols)
{
    const int xm = (x == 0) ? (cols > 1 ? 1 : 0) : x - 1;
    const int xp = (x + 1 == cols) ? (cols > 1 ? cols - 2 : 0) : x + 1;
    const float lap = up[x] + dn[x] + mid[xm] + mid[xp] - 4.f * mid[x];
    return std::fabs(lap);
}

#if CV_SIMD || CV_SIMD_SCALABLE
// Interior only: x >= 1 and x + vlanes <= cols - 1, so both neighbors are in range.
static inline v_float32 mertensContrast(const float* up, const float* mid, const float* dn, int x)
{
    const v_float32 center = vx_load(mid + x);
    v_float32 lap = v_add(vx_load(mid + x - 1), vx_load(mid + x + 1));
    lap = v_add(lap, vx_load(up + x));
    lap = v_add(lap, vx_load(dn + x));
    lap = v_sub(lap, v_mul(center, vx_setall_f32(4.f)));
    return v_abs(lap);
}
#endif

static inline float mertensPowf(float x, float p)
{
    if (p == 0.0f)
        return 1.0f;
    if (p == 1.0f)
        return x;
    if (p == 2.0f)
        return x * x;
    return std::pow(x, p);
}

#if CV_SIMD || CV_SIMD_SCALABLE
static inline v_float32 mertensPow(const v_float32& x, float p)
{
    if (p == 0.0f)
        return vx_setall_f32(1.0f);
    if (p == 1.0f)
        return x;
    if (p == 2.0f)
        return v_mul(x, x);
    return v_exp(v_mul(vx_setall_f32(p), v_log(x)));
}
#endif

// sum may be null. When set, the stored weight is added into that row (weight_sum).
static inline void mertensStore1(float* dst, float* sum, int i, float w)
{
    dst[i] = w;
    if (sum)
        sum[i] += w;
}

#if CV_SIMD || CV_SIMD_SCALABLE
static inline void mertensStoreN(float* dst, float* sum, int i, const v_float32& w)
{
    v_store(dst + i, w);
    if (sum)
        v_store(sum + i, v_add(vx_load(sum + i), w));
}
#endif

void mertensWeightRow_(const float* grayUp, const float* grayMid, const float* grayDn,
                       const float* src, float* dst, int cols, int channels,
                       float wcon, float wsat, float wexp, float* sum)
{
    const float eps = 1e-12f;
    const float inv3 = static_cast<float>(1.0 / 3.0);
    int x = 0;

    // createMergeMertens() defaults. pow(x, 1) is a copy and pow(x, 0) is 1.
    if (channels == 3 && wcon == 1.0f && wsat == 1.0f && wexp == 0.0f)
    {
        if (cols > 0)
        {
            const float c0 = src[0], c1 = src[1], c2 = src[2];
            const float mean = (c0 + c1 + c2) * inv3;
            const float d0 = c0 - mean, d1 = c1 - mean, d2 = c2 - mean;
            const float sat = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
            mertensStore1(dst, sum, 0, mertensContrastAt(grayUp, grayMid, grayDn, 0, cols) * sat + eps);
            x = 1;
        }
#if CV_SIMD || CV_SIMD_SCALABLE
        const int n = VTraits<v_float32>::vlanes();
        const v_float32 vinv3 = vx_setall_f32(inv3);
        const v_float32 veps = vx_setall_f32(eps);
        for (; x <= cols - n - 1; x += n)
        {
            v_float32 c0, c1, c2;
            v_load_deinterleave(src + x * 3, c0, c1, c2);
            const v_float32 mean = v_mul(v_add(v_add(c0, c1), c2), vinv3);
            const v_float32 d0 = v_sub(c0, mean);
            const v_float32 d1 = v_sub(c1, mean);
            const v_float32 d2 = v_sub(c2, mean);
            const v_float32 sat = v_sqrt(v_add(v_add(v_mul(d0, d0), v_mul(d1, d1)), v_mul(d2, d2)));
            const v_float32 contrast = mertensContrast(grayUp, grayMid, grayDn, x);
            mertensStoreN(dst, sum, x, v_add(v_mul(contrast, sat), veps));
        }
#endif
        for (; x < cols; x++)
        {
            const float c0 = src[x * 3], c1 = src[x * 3 + 1], c2 = src[x * 3 + 2];
            const float mean = (c0 + c1 + c2) * inv3;
            const float d0 = c0 - mean, d1 = c1 - mean, d2 = c2 - mean;
            const float sat = std::sqrt(d0 * d0 + d1 * d1 + d2 * d2);
            mertensStore1(dst, sum, x, mertensContrastAt(grayUp, grayMid, grayDn, x, cols) * sat + eps);
        }
        return;
    }

    const bool applySat = channels == 3 && wsat != 0.0f;
    const bool applyExp = wexp != 0.0f;
    const float expScale = -1.0f / 0.08f;

    auto storeGeneric = [&](int i)
    {
        float weight = mertensPowf(mertensContrastAt(grayUp, grayMid, grayDn, i, cols), wcon);
        if (channels == 3)
        {
            const float c0 = src[i * 3], c1 = src[i * 3 + 1], c2 = src[i * 3 + 2];
            if (applySat)
            {
                const float mean = (c0 + c1 + c2) * inv3;
                const float d0 = c0 - mean, d1 = c1 - mean, d2 = c2 - mean;
                weight *= mertensPowf(std::sqrt(d0 * d0 + d1 * d1 + d2 * d2), wsat);
            }
            if (applyExp)
            {
                const float e0 = c0 - 0.5f, e1 = c1 - 0.5f, e2 = c2 - 0.5f;
                const float wellexp = std::exp(e0 * e0 * expScale) * std::exp(e1 * e1 * expScale) * std::exp(e2 * e2 * expScale);
                weight *= mertensPowf(wellexp, wexp);
            }
        }
        else if (applyExp)
        {
            const float e0 = src[i] - 0.5f;
            weight *= mertensPowf(std::exp(e0 * e0 * expScale), wexp);
        }
        mertensStore1(dst, sum, i, weight + eps);
    };

    // Column 0 is the reflect-101 border, so the vector loop starts at column 1.
    if (cols > 0)
    {
        storeGeneric(0);
        x = 1;
    }

#if CV_SIMD || CV_SIMD_SCALABLE
    const int n = VTraits<v_float32>::vlanes();
    const v_float32 veps = vx_setall_f32(eps);
    const v_float32 vhalf = vx_setall_f32(0.5f);
    const v_float32 vexpScale = vx_setall_f32(expScale);
    if (channels == 3)
    {
        const v_float32 vinv3 = vx_setall_f32(inv3);
        for (; x <= cols - n - 1; x += n)
        {
            v_float32 c0, c1, c2;
            v_load_deinterleave(src + x * 3, c0, c1, c2);
            v_float32 weight = mertensPow(mertensContrast(grayUp, grayMid, grayDn, x), wcon);
            if (applySat)
            {
                const v_float32 mean = v_mul(v_add(v_add(c0, c1), c2), vinv3);
                const v_float32 d0 = v_sub(c0, mean);
                const v_float32 d1 = v_sub(c1, mean);
                const v_float32 d2 = v_sub(c2, mean);
                const v_float32 sat = v_sqrt(v_add(v_add(v_mul(d0, d0), v_mul(d1, d1)), v_mul(d2, d2)));
                weight = v_mul(weight, mertensPow(sat, wsat));
            }
            if (applyExp)
            {
                v_float32 e0 = v_sub(c0, vhalf);
                v_float32 e1 = v_sub(c1, vhalf);
                v_float32 e2 = v_sub(c2, vhalf);
                e0 = v_exp(v_mul(v_mul(e0, e0), vexpScale));
                e1 = v_exp(v_mul(v_mul(e1, e1), vexpScale));
                e2 = v_exp(v_mul(v_mul(e2, e2), vexpScale));
                weight = v_mul(weight, mertensPow(v_mul(v_mul(e0, e1), e2), wexp));
            }
            mertensStoreN(dst, sum, x, v_add(weight, veps));
        }
    }
    else
    {
        for (; x <= cols - n - 1; x += n)
        {
            const v_float32 c0 = vx_load(src + x);
            v_float32 weight = mertensPow(mertensContrast(grayUp, grayMid, grayDn, x), wcon);
            if (applyExp)
            {
                const v_float32 e0 = v_sub(c0, vhalf);
                const v_float32 wellexp = v_exp(v_mul(v_mul(e0, e0), vexpScale));
                weight = v_mul(weight, mertensPow(wellexp, wexp));
            }
            mertensStoreN(dst, sum, x, v_add(weight, veps));
        }
    }
#endif

    for (; x < cols; x++)
        storeGeneric(x);
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
