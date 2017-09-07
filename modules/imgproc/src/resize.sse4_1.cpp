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

/* ////////////////////////////////////////////////////////////////////
//
//  Geometrical transforms on images and matrices: rotation, zoom etc.
//
// */

#include "precomp.hpp"
#include "resize.hpp"

namespace cv
{
namespace opt_SSE4_1
{

class resizeNNInvokerSSE2 :
    public ParallelLoopBody
{
public:
    resizeNNInvokerSSE2(const Mat& _src, Mat &_dst, int *_x_ofs, int _pix_size4, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs), pix_size4(_pix_size4),
        ify(_ify)
    {
    }

#if defined(__INTEL_COMPILER)
#pragma optimization_parameter target_arch=SSE4.2
#endif
    virtual void operator() (const Range& range) const
    {
        Size ssize = src.size(), dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int sseWidth = width - (width & 0x7);
        for(y = range.start; y < range.end; y++)
        {
            uchar* D = dst.data + dst.step*y;
            uchar* Dstart = D;
            int sy = std::min(cvFloor(y*ify), ssize.height-1);
            const uchar* S = src.data + sy*src.step;
            __m128i CV_DECL_ALIGNED(64) pixels = _mm_set1_epi16(0);
            for(x = 0; x < sseWidth; x += 8)
            {
                ushort imm = *(ushort*)(S + x_ofs[x + 0]);
                pixels = _mm_insert_epi16(pixels, imm, 0);
                imm = *(ushort*)(S + x_ofs[x + 1]);
                pixels = _mm_insert_epi16(pixels, imm, 1);
                imm = *(ushort*)(S + x_ofs[x + 2]);
                pixels = _mm_insert_epi16(pixels, imm, 2);
                imm = *(ushort*)(S + x_ofs[x + 3]);
                pixels = _mm_insert_epi16(pixels, imm, 3);
                imm = *(ushort*)(S + x_ofs[x + 4]);
                pixels = _mm_insert_epi16(pixels, imm, 4);
                imm = *(ushort*)(S + x_ofs[x + 5]);
                pixels = _mm_insert_epi16(pixels, imm, 5);
                imm = *(ushort*)(S + x_ofs[x + 6]);
                pixels = _mm_insert_epi16(pixels, imm, 6);
                imm = *(ushort*)(S + x_ofs[x + 7]);
                pixels = _mm_insert_epi16(pixels, imm, 7);
                _mm_storeu_si128((__m128i*)D, pixels);
                D += 16;
            }
            for(; x < width; x++)
            {
                *(ushort*)(Dstart + x*2) = *(ushort*)(S + x_ofs[x]);
            }
        }
    }

private:
    const Mat src;
    Mat dst;
    int* x_ofs, pix_size4;
    double ify;

    resizeNNInvokerSSE2(const resizeNNInvokerSSE2&);
    resizeNNInvokerSSE2& operator=(const resizeNNInvokerSSE2&);
};

class resizeNNInvokerSSE4 :
    public ParallelLoopBody
{
public:
    resizeNNInvokerSSE4(const Mat& _src, Mat &_dst, int *_x_ofs, int _pix_size4, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs), pix_size4(_pix_size4),
        ify(_ify)
    {
    }
#if defined(__INTEL_COMPILER)
#pragma optimization_parameter target_arch=SSE4.2
#endif
    virtual void operator() (const Range& range) const
    {
        Size ssize = src.size(), dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int sseWidth = width - (width & 0x3);
        for(y = range.start; y < range.end; y++)
        {
            uchar* D = dst.data + dst.step*y;
            uchar* Dstart = D;
            int sy = std::min(cvFloor(y*ify), ssize.height-1);
            const uchar* S = src.data + sy*src.step;
            __m128i CV_DECL_ALIGNED(64) pixels = _mm_set1_epi16(0);
            for(x = 0; x < sseWidth; x += 4)
            {
                int imm = *(int*)(S + x_ofs[x + 0]);
                pixels = _mm_insert_epi32(pixels, imm, 0);
                imm = *(int*)(S + x_ofs[x + 1]);
                pixels = _mm_insert_epi32(pixels, imm, 1);
                imm = *(int*)(S + x_ofs[x + 2]);
                pixels = _mm_insert_epi32(pixels, imm, 2);
                imm = *(int*)(S + x_ofs[x + 3]);
                pixels = _mm_insert_epi32(pixels, imm, 3);
                _mm_storeu_si128((__m128i*)D, pixels);
                D += 16;
            }
            for(; x < width; x++)
            {
                *(int*)(Dstart + x*4) = *(int*)(S + x_ofs[x]);
            }
        }
    }

private:
    const Mat src;
    Mat dst;
    int* x_ofs, pix_size4;
    double ify;

    resizeNNInvokerSSE4(const resizeNNInvokerSSE4&);
    resizeNNInvokerSSE4& operator=(const resizeNNInvokerSSE4&);
};

void resizeNN2_SSE4_1(const Range& range, const Mat& src, Mat &dst, int *x_ofs, int pix_size4, double ify)
{
    resizeNNInvokerSSE2 invoker(src, dst, x_ofs, pix_size4, ify);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

void resizeNN4_SSE4_1(const Range& range, const Mat& src, Mat &dst, int *x_ofs, int pix_size4, double ify)
{
    resizeNNInvokerSSE4 invoker(src, dst, x_ofs, pix_size4, ify);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

int VResizeLanczos4Vec_32f16u_SSE41(const uchar** _src, uchar* _dst, const uchar* _beta, int width)
{
    const float** src = (const float**)_src;
    const float* beta = (const float*)_beta;
    const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
        *S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
    short * dst = (short*)_dst;
    int x = 0;
    __m128 v_b0 = _mm_set1_ps(beta[0]), v_b1 = _mm_set1_ps(beta[1]),
        v_b2 = _mm_set1_ps(beta[2]), v_b3 = _mm_set1_ps(beta[3]),
        v_b4 = _mm_set1_ps(beta[4]), v_b5 = _mm_set1_ps(beta[5]),
        v_b6 = _mm_set1_ps(beta[6]), v_b7 = _mm_set1_ps(beta[7]);

    for (; x <= width - 8; x += 8)
    {
        __m128 v_dst0 = _mm_mul_ps(v_b0, _mm_loadu_ps(S0 + x));
        v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b1, _mm_loadu_ps(S1 + x)));
        v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b2, _mm_loadu_ps(S2 + x)));
        v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b3, _mm_loadu_ps(S3 + x)));
        v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b4, _mm_loadu_ps(S4 + x)));
        v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b5, _mm_loadu_ps(S5 + x)));
        v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b6, _mm_loadu_ps(S6 + x)));
        v_dst0 = _mm_add_ps(v_dst0, _mm_mul_ps(v_b7, _mm_loadu_ps(S7 + x)));

        __m128 v_dst1 = _mm_mul_ps(v_b0, _mm_loadu_ps(S0 + x + 4));
        v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b1, _mm_loadu_ps(S1 + x + 4)));
        v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b2, _mm_loadu_ps(S2 + x + 4)));
        v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b3, _mm_loadu_ps(S3 + x + 4)));
        v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b4, _mm_loadu_ps(S4 + x + 4)));
        v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b5, _mm_loadu_ps(S5 + x + 4)));
        v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b6, _mm_loadu_ps(S6 + x + 4)));
        v_dst1 = _mm_add_ps(v_dst1, _mm_mul_ps(v_b7, _mm_loadu_ps(S7 + x + 4)));

        __m128i v_dsti0 = _mm_cvtps_epi32(v_dst0);
        __m128i v_dsti1 = _mm_cvtps_epi32(v_dst1);

        _mm_storeu_si128((__m128i *)(dst + x), _mm_packus_epi32(v_dsti0, v_dsti1));
    }

    return x;
}

}
}
/* End of file. */
