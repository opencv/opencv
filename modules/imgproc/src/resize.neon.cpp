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

#if CV_NEON
#include <arm_neon.h>

namespace cv
{
namespace opt_NEON
{

class resizeNNInvokerNEON2 CV_FINAL :
    public ParallelLoopBody
{
public:
    resizeNNInvokerNEON2(const Mat& _src, Mat& _dst, int* _x_ofs, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs), ify(_ify)
    {
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int neonWidth = width - (width & 0x7);

        for (y = range.start; y < range.end; y++)
        {
            uchar* D = dst.data + dst.step * y;
            uchar* Dstart = D;
            int sy = std::min(cvFloor(y * ify), ssize.height - 1);
            const uchar* S = src.data + sy * src.step;

            uint16x8_t pixels = vdupq_n_u16(0);
            for (x = 0; x < neonWidth; x += 8)
            {
                pixels = vsetq_lane_u16(*(const ushort*)(S + x_ofs[x + 0]), pixels, 0);
                pixels = vsetq_lane_u16(*(const ushort*)(S + x_ofs[x + 1]), pixels, 1);
                pixels = vsetq_lane_u16(*(const ushort*)(S + x_ofs[x + 2]), pixels, 2);
                pixels = vsetq_lane_u16(*(const ushort*)(S + x_ofs[x + 3]), pixels, 3);
                pixels = vsetq_lane_u16(*(const ushort*)(S + x_ofs[x + 4]), pixels, 4);
                pixels = vsetq_lane_u16(*(const ushort*)(S + x_ofs[x + 5]), pixels, 5);
                pixels = vsetq_lane_u16(*(const ushort*)(S + x_ofs[x + 6]), pixels, 6);
                pixels = vsetq_lane_u16(*(const ushort*)(S + x_ofs[x + 7]), pixels, 7);
                vst1q_u16((ushort*)D, pixels);
                D += 16;
            }
            for (; x < width; x++)
                *(ushort*)(Dstart + x * 2) = *(const ushort*)(S + x_ofs[x]);
        }
    }

private:
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    double ify;

    resizeNNInvokerNEON2(const resizeNNInvokerNEON2&);
    resizeNNInvokerNEON2& operator=(const resizeNNInvokerNEON2&);
};

class resizeNNInvokerNEON4 CV_FINAL :
    public ParallelLoopBody
{
public:
    resizeNNInvokerNEON4(const Mat& _src, Mat& _dst, int* _x_ofs, double _ify) :
        ParallelLoopBody(), src(_src), dst(_dst), x_ofs(_x_ofs), ify(_ify)
    {
    }

    virtual void operator()(const Range& range) const CV_OVERRIDE
    {
        Size ssize = src.size(), dsize = dst.size();
        int y, x;
        int width = dsize.width;
        int neonWidth = width - (width & 0x3);

        for (y = range.start; y < range.end; y++)
        {
            uchar* D = dst.data + dst.step * y;
            uchar* Dstart = D;
            int sy = std::min(cvFloor(y * ify), ssize.height - 1);
            const uchar* S = src.data + sy * src.step;

            uint32x4_t pixels = vdupq_n_u32(0);
            for (x = 0; x < neonWidth; x += 4)
            {
                pixels = vsetq_lane_u32(*(const uint32_t*)(S + x_ofs[x + 0]), pixels, 0);
                pixels = vsetq_lane_u32(*(const uint32_t*)(S + x_ofs[x + 1]), pixels, 1);
                pixels = vsetq_lane_u32(*(const uint32_t*)(S + x_ofs[x + 2]), pixels, 2);
                pixels = vsetq_lane_u32(*(const uint32_t*)(S + x_ofs[x + 3]), pixels, 3);
                vst1q_u32((uint32_t*)D, pixels);
                D += 16;
            }
            for (; x < width; x++)
                *(uint32_t*)(Dstart + x * 4) = *(const uint32_t*)(S + x_ofs[x]);
        }
    }

private:
    const Mat& src;
    Mat& dst;
    int* x_ofs;
    double ify;

    resizeNNInvokerNEON4(const resizeNNInvokerNEON4&);
    resizeNNInvokerNEON4& operator=(const resizeNNInvokerNEON4&);
};

void resizeNN2_NEON(const Range& range, const Mat& src, Mat& dst, int* x_ofs, double ify)
{
    resizeNNInvokerNEON2 invoker(src, dst, x_ofs, ify);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

void resizeNN4_NEON(const Range& range, const Mat& src, Mat& dst, int* x_ofs, double ify)
{
    resizeNNInvokerNEON4 invoker(src, dst, x_ofs, ify);
    parallel_for_(range, invoker, dst.total() / (double)(1 << 16));
}

} // opt_NEON
} // cv

#endif // CV_NEON
/* End of file. */
