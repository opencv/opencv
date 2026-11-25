// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// srcstep is in elements of the source type.
void blobFromImage32F_(const float* src, size_t srcstep, float* const dst[4],
                       int rows, int cols, int nch,
                       const float mean[4], const float scale[4]);
void blobFromImage8U32F_(const uchar* src, size_t srcstep, float* const dst[4],
                         int rows, int cols, int nch,
                         const float mean[4], const float scale[4]);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

#if (CV_SIMD || CV_SIMD_SCALABLE)
// Source vector type per source element type. Only scalar types are used as template
// arguments because scalable RVV vector types are sizeless.
template<typename T> struct BlobSrcVec;
template<> struct BlobSrcVec<float> { typedef v_float32 type; };
template<> struct BlobSrcVec<uchar> { typedef v_uint8 type; };

// Stores one source vector as (value - mean) * scale; lanes of value and dst advance together.
static inline void storeBlobVector(const v_float32& value, float* dst,
                                   const v_float32& mean, const v_float32& scale)
{
    vx_store(dst, v_mul(v_sub(value, mean), scale));
}

// A uint8 vector expands to four float vectors.
static inline void storeBlobVector(const v_uint8& value, float* dst,
                                   const v_float32& mean, const v_float32& scale)
{
    const int lanes = VTraits<v_float32>::vlanes();
    v_uint16 lo, hi;
    v_uint32 v0, v1, v2, v3;
    v_expand(value, lo, hi);
    v_expand(lo, v0, v1);
    v_expand(hi, v2, v3);
    vx_store(dst, v_mul(v_sub(v_cvt_f32(v_reinterpret_as_s32(v0)), mean), scale));
    vx_store(dst + lanes, v_mul(v_sub(v_cvt_f32(v_reinterpret_as_s32(v1)), mean), scale));
    vx_store(dst + lanes * 2, v_mul(v_sub(v_cvt_f32(v_reinterpret_as_s32(v2)), mean), scale));
    vx_store(dst + lanes * 3, v_mul(v_sub(v_cvt_f32(v_reinterpret_as_s32(v3)), mean), scale));
}
#endif

// Deinterleaves an HWC image of nch (1, 3 or 4) channels into planar float dst[c]
// (each rows*cols, contiguous), applying (x - mean[c]) * scale[c].
template<typename T>
static void blobFromImage_(const T* src, size_t srcstep, float* const dst[4],
                           int rows, int cols, int nch,
                           const float mean[4], const float scale[4])
{
    CV_DbgAssert(nch == 1 || nch == 3 || nch == 4);
#if (CV_SIMD || CV_SIMD_SCALABLE)
    typedef typename BlobSrcVec<T>::type vec;
    const int lanes = VTraits<vec>::vlanes();
    // Scalable RVV types are sizeless and cannot be array elements.
    const v_float32 vmean0 = vx_setall_f32(mean[0]);
    const v_float32 vmean1 = vx_setall_f32(mean[1]);
    const v_float32 vmean2 = vx_setall_f32(mean[2]);
    const v_float32 vmean3 = vx_setall_f32(mean[3]);
    const v_float32 vscale0 = vx_setall_f32(scale[0]);
    const v_float32 vscale1 = vx_setall_f32(scale[1]);
    const v_float32 vscale2 = vx_setall_f32(scale[2]);
    const v_float32 vscale3 = vx_setall_f32(scale[3]);
#endif
    for (int row = 0; row < rows; ++row)
    {
        const T* srcRow = src + row * srcstep;
        float* dstRow[4] = {
            dst[0] + row * cols,
            nch > 1 ? dst[1] + row * cols : 0,
            nch > 2 ? dst[2] + row * cols : 0,
            nch > 3 ? dst[3] + row * cols : 0
        };
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if (nch == 1)
        {
            for (; x <= cols - lanes; x += lanes)
                storeBlobVector(vx_load(srcRow + x), dstRow[0] + x, vmean0, vscale0);
        }
        else if (nch == 3)
        {
            for (; x <= cols - lanes; x += lanes)
            {
                vec c0, c1, c2;
                v_load_deinterleave(srcRow + x * 3, c0, c1, c2);
                storeBlobVector(c0, dstRow[0] + x, vmean0, vscale0);
                storeBlobVector(c1, dstRow[1] + x, vmean1, vscale1);
                storeBlobVector(c2, dstRow[2] + x, vmean2, vscale2);
            }
        }
        else
        {
            for (; x <= cols - lanes; x += lanes)
            {
                vec c0, c1, c2, c3;
                v_load_deinterleave(srcRow + x * 4, c0, c1, c2, c3);
                storeBlobVector(c0, dstRow[0] + x, vmean0, vscale0);
                storeBlobVector(c1, dstRow[1] + x, vmean1, vscale1);
                storeBlobVector(c2, dstRow[2] + x, vmean2, vscale2);
                storeBlobVector(c3, dstRow[3] + x, vmean3, vscale3);
            }
        }
#endif
        for (; x < cols; ++x)
            for (int c = 0; c < nch; ++c)
                dstRow[c][x] = ((float)srcRow[x * nch + c] - mean[c]) * scale[c];
    }
    vx_cleanup();
}

void blobFromImage32F_(const float* src, size_t srcstep, float* const dst[4],
                       int rows, int cols, int nch,
                       const float mean[4], const float scale[4])
{
    blobFromImage_<float>(src, srcstep, dst, rows, cols, nch, mean, scale);
}

void blobFromImage8U32F_(const uchar* src, size_t srcstep, float* const dst[4],
                         int rows, int cols, int nch,
                         const float mean[4], const float scale[4])
{
    blobFromImage_<uchar>(src, srcstep, dst, rows, cols, nch, mean, scale);
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#endif
