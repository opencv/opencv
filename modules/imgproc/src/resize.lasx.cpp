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
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{
namespace opt_LASX
{

class resizeNNInvokerLASX4 CV_FINAL :
    public ParallelLoopBody
{
public:
    resizeNNInvokerLASX4(const Mat& _src, Mat &_dst, int *_x_ofs, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
        ify(_ify)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int avxWidth = width - (width & 0x7);
        if(((int64)(dst.data + dst.step) & 0x1f) == 0)
        {
            for(y = range.start; y < range.end; y++)
            {
                uchar* D = dst.data + dst.step*y;
                uchar* Dstart = D;
                int sy = std::min(cvFloor(y*ify), ssize.height-1);
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
                int sy = std::min(cvFloor(y*ify), ssize.height-1);
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
    double ify;

    resizeNNInvokerLASX4(const resizeNNInvokerLASX4&);
    resizeNNInvokerLASX4& operator=(const resizeNNInvokerLASX4&);
};

class resizeNNInvokerLASX2 CV_FINAL :
    public ParallelLoopBody
{
public:
    resizeNNInvokerLASX2(const Mat& _src, Mat &_dst, int *_x_ofs, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs),
        ify(_ify)
    {
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
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
                int sy = std::min(cvFloor(y*ify), ssize.height-1);
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
                int sy = std::min(cvFloor(y*ify), ssize.height-1);
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
    double ify;

    resizeNNInvokerLASX2(const resizeNNInvokerLASX2&);
    resizeNNInvokerLASX2& operator=(const resizeNNInvokerLASX2&);
};

void resizeNN2_LASX(const Range& range, const Mat& src, Mat &dst, int *x_ofs, double ify)
{
    resizeNNInvokerLASX2 invoker(src, dst, x_ofs, ify);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

void resizeNN4_LASX(const Range& range, const Mat& src, Mat &dst, int *x_ofs, double ify)
{
    resizeNNInvokerLASX4 invoker(src, dst, x_ofs, ify);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

}
}
/* End of file. */
