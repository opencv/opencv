/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// Nearest-neighbour resize of dst rows [range) for 2- and 4-byte pixels. x_ofs holds a source byte
// offset per dst column, y_ofs a source row per dst row. Returns false when this target has no
// vector path, and the caller then runs the generic scalar loop.
bool resizeNN2or4(const Mat& src, Mat& dst, int* x_ofs, const int* y_ofs, const Range& range);
int vresizeLanczos4_32f16u(const float** src, ushort* dst, const float* beta, int width);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// Nearest neighbour is a pure gather, so only targets with a gather instruction get a vector path:
// the 256- and 512-bit ones and RVV. On 128-bit targets the vector would be assembled from the
// same scalar loads the plain loop does.
#if (CV_SIMD || CV_SIMD_SCALABLE) && CV_SIMD_WIDTH >= 32

static void resizeNN4(const Mat& src, Mat& dst, const int* x_ofs, const int* y_ofs, const Range& range)
{
    const int width = dst.cols, VECSZ = VTraits<v_int32>::vlanes();
    for (int y = range.start; y < range.end; y++)
    {
        const int* S = (const int*)(src.data + src.step*y_ofs[y]);
        int* D = (int*)(dst.data + dst.step*y);
        int x = 0;
        // x_ofs is in bytes; v_lut indexes 32-bit elements
        for (; x <= width - VECSZ; x += VECSZ)
            v_store(D + x, v_lut(S, v_shr<2>(vx_load(x_ofs + x))));
        for (; x < width; x++)
            D[x] = S[x_ofs[x] >> 2];
    }
    vx_cleanup();
}

// There is no 16-bit gather, so this fetches the 32-bit word that ends with each pixel, keeps its
// upper half and packs two such vectors into one. The word starts 2 bytes before the pixel, which
// leaves the pixels taken from source column 0 to the scalar loop; x_ofs never decreases, so they
// are a prefix of every row.
static void resizeNN2(const Mat& src, Mat& dst, const int* x_ofs, const int* y_ofs, const Range& range)
{
    const int width = dst.cols;
    const int VECSZ = VTraits<v_uint16>::vlanes(), HALF = VTraits<v_uint32>::vlanes();
    int x0 = 0;
    while (x0 < width && x_ofs[x0] < 2)
        x0++;

    for (int y = range.start; y < range.end; y++)
    {
        const uchar* S = src.data + src.step*y_ofs[y];
        ushort* D = (ushort*)(dst.data + dst.step*y);
        int x = 0;
        for (; x < x0; x++)
            D[x] = *(const ushort*)(S + x_ofs[x]);
        for (; x <= width - VECSZ; x += VECSZ)
        {
            v_uint32 a = v_shr<16>(v_reinterpret_as_u32(vx_lut_quads((const schar*)(S - 2), x_ofs + x)));
            v_uint32 b = v_shr<16>(v_reinterpret_as_u32(vx_lut_quads((const schar*)(S - 2), x_ofs + x + HALF)));
            v_store(D + x, v_pack_u(v_reinterpret_as_s32(a), v_reinterpret_as_s32(b)));
        }
        for (; x < width; x++)
            D[x] = *(const ushort*)(S + x_ofs[x]);
    }
    vx_cleanup();
}

#endif

bool resizeNN2or4(const Mat& src, Mat& dst, int* x_ofs, const int* y_ofs, const Range& range)
{
#if (CV_SIMD || CV_SIMD_SCALABLE) && CV_SIMD_WIDTH >= 32
    const size_t pix_size = src.elemSize();
    if (pix_size == 4)
    {
        resizeNN4(src, dst, x_ofs, y_ofs, range);
        return true;
    }
    if (pix_size == 2)
    {
        resizeNN2(src, dst, x_ofs, y_ofs, range);
        return true;
    }
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(x_ofs); CV_UNUSED(y_ofs); CV_UNUSED(range);
#endif
    return false;
}

// Vertical Lanczos4 pass, float rows to ushort. Returns how many elements it wrote; the caller
// finishes the row.
int vresizeLanczos4_32f16u(const float** src, ushort* dst, const float* beta, int width)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
    const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
                *S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
    const v_float32 b0 = vx_setall_f32(beta[0]), b1 = vx_setall_f32(beta[1]),
                    b2 = vx_setall_f32(beta[2]), b3 = vx_setall_f32(beta[3]),
                    b4 = vx_setall_f32(beta[4]), b5 = vx_setall_f32(beta[5]),
                    b6 = vx_setall_f32(beta[6]), b7 = vx_setall_f32(beta[7]);
    int x = 0;
    for (; x <= width - 2*VECSZ; x += 2*VECSZ)
    {
        const int x1 = x + VECSZ;
        v_float32 d0 = v_mul(vx_load(S0 + x), b0), d1 = v_mul(vx_load(S0 + x1), b0);
        d0 = v_muladd(vx_load(S1 + x), b1, d0); d1 = v_muladd(vx_load(S1 + x1), b1, d1);
        d0 = v_muladd(vx_load(S2 + x), b2, d0); d1 = v_muladd(vx_load(S2 + x1), b2, d1);
        d0 = v_muladd(vx_load(S3 + x), b3, d0); d1 = v_muladd(vx_load(S3 + x1), b3, d1);
        d0 = v_muladd(vx_load(S4 + x), b4, d0); d1 = v_muladd(vx_load(S4 + x1), b4, d1);
        d0 = v_muladd(vx_load(S5 + x), b5, d0); d1 = v_muladd(vx_load(S5 + x1), b5, d1);
        d0 = v_muladd(vx_load(S6 + x), b6, d0); d1 = v_muladd(vx_load(S6 + x1), b6, d1);
        d0 = v_muladd(vx_load(S7 + x), b7, d0); d1 = v_muladd(vx_load(S7 + x1), b7, d1);
        v_store(dst + x, v_pack_u(v_round(d0), v_round(d1)));
    }
    vx_cleanup();
    return x;
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(beta); CV_UNUSED(width);
    return 0;
#endif
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
