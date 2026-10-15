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

// Nearest-neighbor resize of rows [range) for 2- and 4-byte pixels; returns false if no vectorized path applies.
bool resizeNN2or4(const Mat& src, Mat& dst, int* x_ofs, const int* y_ofs, const Range& range);
int vresizeLanczos4_32f16u(const float** src, ushort* dst, const float* beta, int width);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

#if CV_AVX2

class resizeNNInvokerAVX4 CV_FINAL :
    public ParallelLoopBody
{
public:
    resizeNNInvokerAVX4(const Mat& _src, Mat &_dst, int *_x_ofs, const int *_y_ofs) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
        y_ofs(_y_ofs)
    {
    }

#if defined(__INTEL_COMPILER)
#pragma optimization_parameter target_arch=AVX
#endif
    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int avxWidth = width - (width & 0x7);
        const __m256i CV_DECL_ALIGNED(64) mask = _mm256_set1_epi32(-1);
        if(((int64)(dst.data + dst.step) & 0x1f) == 0)
        {
            for(y = range.start; y < range.end; y++)
            {
                uchar* D = dst.data + dst.step*y;
                uchar* Dstart = D;
                int sy = y_ofs[y];
                const uchar* S = src.data + sy*src.step;
#ifdef CV_ICC
#pragma unroll(4)
#endif
                for(x = 0; x < avxWidth; x += 8)
                {
                    const __m256i CV_DECL_ALIGNED(64) *addr = (__m256i*)(x_ofs + x);
                    __m256i CV_DECL_ALIGNED(64) indices = _mm256_lddqu_si256(addr);
                    __m256i CV_DECL_ALIGNED(64) pixels = _mm256_i32gather_epi32((const int*)S, indices, 1);
                    _mm256_maskstore_epi32((int*)D, mask, pixels);
                    D += 32;
                }
                for(; x < width; x++)
                {
                    *(int*)(Dstart + x*4) = *(int*)(S + x_ofs[x]);
                }
            }
        }
        else
        {
            for(y = range.start; y < range.end; y++)
            {
                uchar* D = dst.data + dst.step*y;
                uchar* Dstart = D;
                int sy = y_ofs[y];
                const uchar* S = src.data + sy*src.step;
#ifdef CV_ICC
#pragma unroll(4)
#endif
                for(x = 0; x < avxWidth; x += 8)
                {
                    const __m256i CV_DECL_ALIGNED(64) *addr = (__m256i*)(x_ofs + x);
                    __m256i CV_DECL_ALIGNED(64) indices = _mm256_lddqu_si256(addr);
                    __m256i CV_DECL_ALIGNED(64) pixels = _mm256_i32gather_epi32((const int*)S, indices, 1);
                    _mm256_storeu_si256((__m256i*)D, pixels);
                    D += 32;
                }
                for(; x < width; x++)
                {
                    *(int*)(Dstart + x*4) = *(int*)(S + x_ofs[x]);
                }
            }
        }
        _mm256_zeroupper();
    }

private:
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    const int* y_ofs;

    resizeNNInvokerAVX4(const resizeNNInvokerAVX4&);
    resizeNNInvokerAVX4& operator=(const resizeNNInvokerAVX4&);
};

class resizeNNInvokerAVX2 CV_FINAL :
    public ParallelLoopBody
{
public:
    resizeNNInvokerAVX2(const Mat& _src, Mat &_dst, int *_x_ofs, const int *_y_ofs) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
        y_ofs(_y_ofs)
    {
    }

#if defined(__INTEL_COMPILER)
#pragma optimization_parameter target_arch=AVX
#endif
    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size dsize = dst.size();
        int y, x;
        int width = dsize.width;
        //int avxWidth = (width - 1) - ((width - 1) & 0x7);
        int avxWidth = width - (width & 0xf);
        const __m256i CV_DECL_ALIGNED(64) mask = _mm256_set1_epi32(-1);
        const __m256i CV_DECL_ALIGNED(64) shuffle_mask = _mm256_set_epi8(15,14,11,10,13,12,9,8,7,6,3,2,5,4,1,0,
                                                                         15,14,11,10,13,12,9,8,7,6,3,2,5,4,1,0);
        const __m256i CV_DECL_ALIGNED(64) permute_mask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
        //const __m256i CV_DECL_ALIGNED(64) shift_shuffle_mask = _mm256_set_epi8(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2,
        //                                                                       13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2);
        if(((int64)(dst.data + dst.step) & 0x1f) == 0)
        {
            for(y = range.start; y < range.end; y++)
            {
                uchar* D = dst.data + dst.step*y;
                uchar* Dstart = D;
                int sy = y_ofs[y];
                const uchar* S = src.data + sy*src.step;
                const uchar* S2 = S - 2;
#ifdef CV_ICC
#pragma unroll(4)
#endif
                for(x = 0; x < avxWidth; x += 16)
                {
                    const __m256i CV_DECL_ALIGNED(64) *addr = (__m256i*)(x_ofs + x);
                    __m256i CV_DECL_ALIGNED(64) indices = _mm256_lddqu_si256(addr);
                    __m256i CV_DECL_ALIGNED(64) pixels1 = _mm256_i32gather_epi32((const int*)S, indices, 1);
                    const __m256i CV_DECL_ALIGNED(64) *addr2 = (__m256i*)(x_ofs + x + 8);
                    __m256i CV_DECL_ALIGNED(64) indices2 = _mm256_lddqu_si256(addr2);
                    __m256i CV_DECL_ALIGNED(64) pixels2 = _mm256_i32gather_epi32((const int*)S2, indices2, 1);
                    __m256i CV_DECL_ALIGNED(64) unpacked = _mm256_blend_epi16(pixels1, pixels2, 0xaa);

                    __m256i CV_DECL_ALIGNED(64) bytes_shuffled = _mm256_shuffle_epi8(unpacked, shuffle_mask);
                    __m256i CV_DECL_ALIGNED(64) ints_permuted = _mm256_permutevar8x32_epi32(bytes_shuffled, permute_mask);
                    _mm256_maskstore_epi32((int*)D, mask, ints_permuted);
                    D += 32;
                }
                for(; x < width; x++)
                {
                    *(ushort*)(Dstart + x*2) = *(ushort*)(S + x_ofs[x]);
                }

            }
        }
        else
        {
            for(y = range.start; y < range.end; y++)
            {
                uchar* D = dst.data + dst.step*y;
                uchar* Dstart = D;
                int sy = y_ofs[y];
                const uchar* S = src.data + sy*src.step;
                const uchar* S2 = S - 2;
#ifdef CV_ICC
#pragma unroll(4)
#endif
                for(x = 0; x < avxWidth; x += 16)
                {
                    const __m256i CV_DECL_ALIGNED(64) *addr = (__m256i*)(x_ofs + x);
                    __m256i CV_DECL_ALIGNED(64) indices = _mm256_lddqu_si256(addr);
                    __m256i CV_DECL_ALIGNED(64) pixels1 = _mm256_i32gather_epi32((const int*)S, indices, 1);
                    const __m256i CV_DECL_ALIGNED(64) *addr2 = (__m256i*)(x_ofs + x + 8);
                    __m256i CV_DECL_ALIGNED(64) indices2 = _mm256_lddqu_si256(addr2);
                    __m256i CV_DECL_ALIGNED(64) pixels2 = _mm256_i32gather_epi32((const int*)S2, indices2, 1);
                    __m256i CV_DECL_ALIGNED(64) unpacked = _mm256_blend_epi16(pixels1, pixels2, 0xaa);

                    __m256i CV_DECL_ALIGNED(64) bytes_shuffled = _mm256_shuffle_epi8(unpacked, shuffle_mask);
                    __m256i CV_DECL_ALIGNED(64) ints_permuted = _mm256_permutevar8x32_epi32(bytes_shuffled, permute_mask);
                    _mm256_storeu_si256((__m256i*)D, ints_permuted);
                    D += 32;
                }
                for(; x < width; x++)
                {
                    *(ushort*)(Dstart + x*2) = *(ushort*)(S + x_ofs[x]);
                }
            }
        }
        _mm256_zeroupper();
    }

private:
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    const int* y_ofs;

    resizeNNInvokerAVX2(const resizeNNInvokerAVX2&);
    resizeNNInvokerAVX2& operator=(const resizeNNInvokerAVX2&);
};

#elif CV_SSE4_1

class resizeNNInvokerSSE2 :
    public ParallelLoopBody
{
public:
    resizeNNInvokerSSE2(const Mat& _src, Mat &_dst, int *_x_ofs, const int *_y_ofs) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
        y_ofs(_y_ofs)
    {
    }

#if defined(__INTEL_COMPILER)
#pragma optimization_parameter target_arch=SSE4.2
#endif
    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int sseWidth = width - (width & 0x7);
        for(y = range.start; y < range.end; y++)
        {
            uchar* D = dst.data + dst.step*y;
            uchar* Dstart = D;
            int sy = y_ofs[y];
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
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    const int* y_ofs;

    resizeNNInvokerSSE2(const resizeNNInvokerSSE2&);
    resizeNNInvokerSSE2& operator=(const resizeNNInvokerSSE2&);
};

class resizeNNInvokerSSE4 :
    public ParallelLoopBody
{
public:
    resizeNNInvokerSSE4(const Mat& _src, Mat &_dst, int *_x_ofs, const int *_y_ofs) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
        y_ofs(_y_ofs)
    {
    }
#if defined(__INTEL_COMPILER)
#pragma optimization_parameter target_arch=SSE4.2
#endif
    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int sseWidth = width - (width & 0x3);
        for(y = range.start; y < range.end; y++)
        {
            uchar* D = dst.data + dst.step*y;
            uchar* Dstart = D;
            int sy = y_ofs[y];
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
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    const int* y_ofs;

    resizeNNInvokerSSE4(const resizeNNInvokerSSE4&);
    resizeNNInvokerSSE4& operator=(const resizeNNInvokerSSE4&);
};

#elif CV_LASX

class resizeNNInvokerLASX4 CV_FINAL :
    public ParallelLoopBody
{
public:
    resizeNNInvokerLASX4(const Mat& _src, Mat &_dst, int *_x_ofs, const int *_y_ofs) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
        y_ofs(_y_ofs)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int avxWidth = width - (width & 0x7);
        if(((int64)(dst.data + dst.step) & 0x1f) == 0)
        {
            for(y = range.start; y < range.end; y++)
            {
                uchar* D = dst.data + dst.step*y;
                uchar* Dstart = D;
                int sy = y_ofs[y];
                const uchar* S = src.data + sy*src.step;
#ifdef CV_ICC
#pragma unroll(4)
#endif
                for(x = 0; x < avxWidth; x += 8)
                {
                    const __m256i CV_DECL_ALIGNED(64) *addr = (__m256i*)(x_ofs + x);
                    __m256i CV_DECL_ALIGNED(64) pixels = v256_lut_quads((schar *)S, (int *)addr).val;
                    __lasx_xvst(pixels, (int*)D, 0);
                    D += 32;
                }
                for(; x < width; x++)
                {
                    *(int*)(Dstart + x*4) = *(int*)(S + x_ofs[x]);
                }
            }
        }
        else
        {
            for(y = range.start; y < range.end; y++)
            {
                uchar* D = dst.data + dst.step*y;
                uchar* Dstart = D;
                int sy = y_ofs[y];
                const uchar* S = src.data + sy*src.step;
#ifdef CV_ICC
#pragma unroll(4)
#endif
                for(x = 0; x < avxWidth; x += 8)
                {
                    const __m256i CV_DECL_ALIGNED(64) *addr = (__m256i*)(x_ofs + x);
                    __m256i CV_DECL_ALIGNED(64) pixels = v256_lut_quads((schar *)S, (int *)addr).val;
                    __lasx_xvst(pixels, (int*)D, 0);
                    D += 32;
                }
                for(; x < width; x++)
                {
                    *(int*)(Dstart + x*4) = *(int*)(S + x_ofs[x]);
                }
            }
        }
    }

private:
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    const int* y_ofs;

    resizeNNInvokerLASX4(const resizeNNInvokerLASX4&);
    resizeNNInvokerLASX4& operator=(const resizeNNInvokerLASX4&);
};

class resizeNNInvokerLASX2 CV_FINAL :
    public ParallelLoopBody
{
public:
    resizeNNInvokerLASX2(const Mat& _src, Mat &_dst, int *_x_ofs, const int *_y_ofs) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
        y_ofs(_y_ofs)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int avxWidth = width - (width & 0xf);
        const __m256i CV_DECL_ALIGNED(64) shuffle_mask = _v256_set_b(15,14,11,10,13,12,9,8,7,6,3,2,5,4,1,0,
                                                                     15,14,11,10,13,12,9,8,7,6,3,2,5,4,1,0);
        const __m256i CV_DECL_ALIGNED(64) permute_mask = _v256_set_w(7, 5, 3, 1, 6, 4, 2, 0);
        if(((int64)(dst.data + dst.step) & 0x1f) == 0)
        {
            for(y = range.start; y < range.end; y++)
            {
                uchar* D = dst.data + dst.step*y;
                uchar* Dstart = D;
                int sy = y_ofs[y];
                const uchar* S = src.data + sy*src.step;
                const uchar* S2 = S - 2;
#ifdef CV_ICC
#pragma unroll(4)
#endif
                for(x = 0; x < avxWidth; x += 16)
                {
                    const __m256i CV_DECL_ALIGNED(64) *addr = (__m256i*)(x_ofs + x);
                    __m256i CV_DECL_ALIGNED(64) pixels1 = v256_lut_quads((schar *)S, (int *)addr).val;

                    const __m256i CV_DECL_ALIGNED(64) *addr2 = (__m256i*)(x_ofs + x + 8);
                    __m256i CV_DECL_ALIGNED(64) pixels2 = v256_lut_quads((schar *)S2, (int *)addr2).val;

                    const __m256i h_mask = __lasx_xvreplgr2vr_w(0xFFFF0000);
                    __m256i CV_DECL_ALIGNED(64) unpacked = __lasx_xvbitsel_v(pixels1, pixels2, h_mask);

                    __m256i CV_DECL_ALIGNED(64) bytes_shuffled = __lasx_xvshuf_b(unpacked, unpacked, shuffle_mask);
                    __m256i CV_DECL_ALIGNED(64) ints_permuted = __lasx_xvperm_w(bytes_shuffled, permute_mask);
                    __lasx_xvst(ints_permuted, (int*)D, 0);
                    D += 32;
                }
                for(; x < width; x++)
                {
                    *(ushort*)(Dstart + x*2) = *(ushort*)(S + x_ofs[x]);
                }

            }
        }
        else
        {
            for(y = range.start; y < range.end; y++)
            {
                uchar* D = dst.data + dst.step*y;
                uchar* Dstart = D;
                int sy = y_ofs[y];
                const uchar* S = src.data + sy*src.step;
                const uchar* S2 = S - 2;
#ifdef CV_ICC
#pragma unroll(4)
#endif
                for(x = 0; x < avxWidth; x += 16)
                {
                    const __m256i CV_DECL_ALIGNED(64) *addr = (__m256i*)(x_ofs + x);
                    __m256i CV_DECL_ALIGNED(64) pixels1 = v256_lut_quads((schar *)S, (int *)addr).val;

                    const __m256i CV_DECL_ALIGNED(64) *addr2 = (__m256i*)(x_ofs + x + 8);
                    __m256i CV_DECL_ALIGNED(64) pixels2 = v256_lut_quads((schar *)S2, (int *)addr2).val;

                    const __m256i h_mask = __lasx_xvreplgr2vr_w(0xFFFF0000);
                    __m256i CV_DECL_ALIGNED(64) unpacked = __lasx_xvbitsel_v(pixels1, pixels2, h_mask);

                    __m256i CV_DECL_ALIGNED(64) bytes_shuffled = __lasx_xvshuf_b(unpacked, unpacked, shuffle_mask);
                    __m256i CV_DECL_ALIGNED(64) ints_permuted = __lasx_xvperm_w(bytes_shuffled, permute_mask);
                    __lasx_xvst(ints_permuted, (int*)D, 0);
                    D += 32;
                }
                for(; x < width; x++)
                {
                    *(ushort*)(Dstart + x*2) = *(ushort*)(S + x_ofs[x]);
                }
            }
        }
    }

private:
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    const int* y_ofs;

    resizeNNInvokerLASX2(const resizeNNInvokerLASX2&);
    resizeNNInvokerLASX2& operator=(const resizeNNInvokerLASX2&);
};

#endif

bool resizeNN2or4(const Mat& src, Mat& dst, int* x_ofs, const int* y_ofs, const Range& range)
{
    int pix_size = (int)src.elemSize();
#if CV_AVX2
    if (pix_size == 2)
    {
        resizeNNInvokerAVX2 invoker(src, dst, x_ofs, y_ofs);
        invoker(range);
        return true;
    }
    if (pix_size == 4)
    {
        resizeNNInvokerAVX4 invoker(src, dst, x_ofs, y_ofs);
        invoker(range);
        return true;
    }
#elif CV_SSE4_1
    if (pix_size == 2)
    {
        resizeNNInvokerSSE2 invoker(src, dst, x_ofs, y_ofs);
        invoker(range);
        return true;
    }
    if (pix_size == 4)
    {
        resizeNNInvokerSSE4 invoker(src, dst, x_ofs, y_ofs);
        invoker(range);
        return true;
    }
#elif CV_LASX
    if (pix_size == 2)
    {
        resizeNNInvokerLASX2 invoker(src, dst, x_ofs, y_ofs);
        invoker(range);
        return true;
    }
    if (pix_size == 4)
    {
        resizeNNInvokerLASX4 invoker(src, dst, x_ofs, y_ofs);
        invoker(range);
        return true;
    }
#else
    CV_UNUSED(pix_size); CV_UNUSED(dst); CV_UNUSED(x_ofs); CV_UNUSED(y_ofs); CV_UNUSED(range);
#endif
    return false;
}

int vresizeLanczos4_32f16u(const float** src, ushort* dst, const float* beta, int width)
{
#if CV_SSE4_1
    const float *S0 = src[0], *S1 = src[1], *S2 = src[2], *S3 = src[3],
        *S4 = src[4], *S5 = src[5], *S6 = src[6], *S7 = src[7];
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
#else
    CV_UNUSED(src); CV_UNUSED(dst); CV_UNUSED(beta); CV_UNUSED(width);
    return 0;
#endif
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
