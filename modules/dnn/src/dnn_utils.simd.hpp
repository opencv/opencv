// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void blobFromImage32F_(const float* src, size_t srcstep, float* const dst[4],
                       int rows, int cols, int nch,
                       const float mean[4], const float scale[4], bool normalize);
void blobFromImage8U32F_(const uchar* src, size_t srcstep, float* const dst[4],
                         int rows, int cols, int nch,
                         const float mean[4], const float scale[4], bool normalize);
void blobFromImage8U8U_(const uchar* src, size_t srcstep, uchar* const dst[4],
                        int rows, int cols, int nch);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

#if (CV_SIMD || CV_SIMD_SCALABLE)
static inline v_float32 normalizeBlobVector(const v_float32& value,
                                            const v_float32& mean,
                                            const v_float32& scale, bool normalize)
{
    return normalize ? v_mul(v_sub(value, mean), scale) : value;
}

static inline void storeBlob8UAs32F(const v_uint8& value, float* dst,
                                    const v_float32& mean, const v_float32& scale,
                                    bool normalize)
{
    const int lanes = VTraits<v_float32>::vlanes();
    v_uint16 lo, hi;
    v_uint32 v0, v1, v2, v3;
    v_expand(value, lo, hi);
    v_expand(lo, v0, v1);
    v_expand(hi, v2, v3);
    vx_store(dst, normalizeBlobVector(v_cvt_f32(v_reinterpret_as_s32(v0)), mean, scale, normalize));
    vx_store(dst + lanes, normalizeBlobVector(v_cvt_f32(v_reinterpret_as_s32(v1)), mean, scale, normalize));
    vx_store(dst + lanes * 2, normalizeBlobVector(v_cvt_f32(v_reinterpret_as_s32(v2)), mean, scale, normalize));
    vx_store(dst + lanes * 3, normalizeBlobVector(v_cvt_f32(v_reinterpret_as_s32(v3)), mean, scale, normalize));
}
#endif

void blobFromImage32F_(const float* src, size_t srcstep, float* const dst[4],
                       int rows, int cols, int nch,
                       const float mean[4], const float scale[4], bool normalize)
{
    CV_DbgAssert(nch == 1 || nch == 3 || nch == 4);
    for (int row = 0; row < rows; ++row)
    {
        const float* srcRow = src + row * srcstep;
        float* dstRow[4] = {
            dst[0] + row * cols,
            nch > 1 ? dst[1] + row * cols : 0,
            nch > 2 ? dst[2] + row * cols : 0,
            nch > 3 ? dst[3] + row * cols : 0
        };
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int lanes = VTraits<v_float32>::vlanes();
        v_float32 vmean[4], vscale[4];
        for (int c = 0; c < nch; ++c)
        {
            vmean[c] = vx_setall_f32(mean[c]);
            vscale[c] = vx_setall_f32(scale[c]);
        }
        if (nch == 1)
        {
            for (; x <= cols - lanes; x += lanes)
                vx_store(dstRow[0] + x, normalizeBlobVector(vx_load(srcRow + x), vmean[0], vscale[0], normalize));
        }
        else if (nch == 3)
        {
            for (; x <= cols - lanes; x += lanes)
            {
                v_float32 c0, c1, c2;
                v_load_deinterleave(srcRow + x * 3, c0, c1, c2);
                vx_store(dstRow[0] + x, normalizeBlobVector(c0, vmean[0], vscale[0], normalize));
                vx_store(dstRow[1] + x, normalizeBlobVector(c1, vmean[1], vscale[1], normalize));
                vx_store(dstRow[2] + x, normalizeBlobVector(c2, vmean[2], vscale[2], normalize));
            }
        }
        else
        {
            for (; x <= cols - lanes; x += lanes)
            {
                v_float32 c0, c1, c2, c3;
                v_load_deinterleave(srcRow + x * 4, c0, c1, c2, c3);
                vx_store(dstRow[0] + x, normalizeBlobVector(c0, vmean[0], vscale[0], normalize));
                vx_store(dstRow[1] + x, normalizeBlobVector(c1, vmean[1], vscale[1], normalize));
                vx_store(dstRow[2] + x, normalizeBlobVector(c2, vmean[2], vscale[2], normalize));
                vx_store(dstRow[3] + x, normalizeBlobVector(c3, vmean[3], vscale[3], normalize));
            }
        }
#endif
        for (; x < cols; ++x)
            for (int c = 0; c < nch; ++c)
                dstRow[c][x] = normalize ? (srcRow[x * nch + c] - mean[c]) * scale[c]
                                         : srcRow[x * nch + c];
    }
    vx_cleanup();
}

void blobFromImage8U32F_(const uchar* src, size_t srcstep, float* const dst[4],
                         int rows, int cols, int nch,
                         const float mean[4], const float scale[4], bool normalize)
{
    CV_DbgAssert(nch == 1 || nch == 3 || nch == 4);
    for (int row = 0; row < rows; ++row)
    {
        const uchar* srcRow = src + row * srcstep;
        float* dstRow[4] = {
            dst[0] + row * cols,
            nch > 1 ? dst[1] + row * cols : 0,
            nch > 2 ? dst[2] + row * cols : 0,
            nch > 3 ? dst[3] + row * cols : 0
        };
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int lanes8 = VTraits<v_uint8>::vlanes();
        v_float32 vmean[4], vscale[4];
        for (int c = 0; c < nch; ++c)
        {
            vmean[c] = vx_setall_f32(mean[c]);
            vscale[c] = vx_setall_f32(scale[c]);
        }
        if (nch == 1)
        {
            for (; x <= cols - lanes8; x += lanes8)
                storeBlob8UAs32F(vx_load(srcRow + x), dstRow[0] + x, vmean[0], vscale[0], normalize);
        }
        else if (nch == 3)
        {
            for (; x <= cols - lanes8; x += lanes8)
            {
                v_uint8 c0, c1, c2;
                v_load_deinterleave(srcRow + x * 3, c0, c1, c2);
                storeBlob8UAs32F(c0, dstRow[0] + x, vmean[0], vscale[0], normalize);
                storeBlob8UAs32F(c1, dstRow[1] + x, vmean[1], vscale[1], normalize);
                storeBlob8UAs32F(c2, dstRow[2] + x, vmean[2], vscale[2], normalize);
            }
        }
        else
        {
            for (; x <= cols - lanes8; x += lanes8)
            {
                v_uint8 c0, c1, c2, c3;
                v_load_deinterleave(srcRow + x * 4, c0, c1, c2, c3);
                storeBlob8UAs32F(c0, dstRow[0] + x, vmean[0], vscale[0], normalize);
                storeBlob8UAs32F(c1, dstRow[1] + x, vmean[1], vscale[1], normalize);
                storeBlob8UAs32F(c2, dstRow[2] + x, vmean[2], vscale[2], normalize);
                storeBlob8UAs32F(c3, dstRow[3] + x, vmean[3], vscale[3], normalize);
            }
        }
#endif
        for (; x < cols; ++x)
            for (int c = 0; c < nch; ++c)
                dstRow[c][x] = normalize ? (srcRow[x * nch + c] - mean[c]) * scale[c]
                                         : (float)srcRow[x * nch + c];
    }
    vx_cleanup();
}

void blobFromImage8U8U_(const uchar* src, size_t srcstep, uchar* const dst[4],
                        int rows, int cols, int nch)
{
    CV_DbgAssert(nch == 1 || nch == 3 || nch == 4);
    for (int row = 0; row < rows; ++row)
    {
        const uchar* srcRow = src + row * srcstep;
        uchar* dstRow[4] = {
            dst[0] + row * cols,
            nch > 1 ? dst[1] + row * cols : 0,
            nch > 2 ? dst[2] + row * cols : 0,
            nch > 3 ? dst[3] + row * cols : 0
        };
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int lanes = VTraits<v_uint8>::vlanes();
        if (nch == 1)
        {
            for (; x <= cols - lanes; x += lanes)
                vx_store(dstRow[0] + x, vx_load(srcRow + x));
        }
        else if (nch == 3)
        {
            for (; x <= cols - lanes; x += lanes)
            {
                v_uint8 c0, c1, c2;
                v_load_deinterleave(srcRow + x * 3, c0, c1, c2);
                vx_store(dstRow[0] + x, c0);
                vx_store(dstRow[1] + x, c1);
                vx_store(dstRow[2] + x, c2);
            }
        }
        else
        {
            for (; x <= cols - lanes; x += lanes)
            {
                v_uint8 c0, c1, c2, c3;
                v_load_deinterleave(srcRow + x * 4, c0, c1, c2, c3);
                vx_store(dstRow[0] + x, c0);
                vx_store(dstRow[1] + x, c1);
                vx_store(dstRow[2] + x, c2);
                vx_store(dstRow[3] + x, c3);
            }
        }
#endif
        for (; x < cols; ++x)
            for (int c = 0; c < nch; ++c)
                dstRow[c][x] = srcRow[x * nch + c];
    }
    vx_cleanup();
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#endif
