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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if kercn != 3
#define loadpix(addr) *(__global const T *)(addr)
#define storepix(val, addr)  *(__global T *)(addr) = val
#define storepix_2(val0, val1, addr0, addr1) \
    *(__global T *)(addr0) = val0; *(__global T *)(addr1) = val1
#define TSIZE (int)sizeof(T)
#else
#define loadpix(addr) vload3(0, (__global const T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#if DEPTH == 2 || DEPTH == 3
#define storepix_2(val0, val1, addr0, addr1) \
    ((__global T1 *)(addr0))[0] = val0.x; \
    ((__global T1 *)(addr1))[0] = val1.x; \
    ((__global T1 *)(addr0))[1] = val0.y; \
    ((__global T1 *)(addr1))[1] = val1.y; \
    ((__global T1 *)(addr0))[2] = val0.z; \
    ((__global T1 *)(addr1))[2] = val1.z
#else
#define storepix_2(val0, val1, addr0, addr1) \
    storepix(val0, addr0); \
    storepix(val1, addr1)
#endif
#define TSIZE ((int)sizeof(T1)*3)
#endif

__kernel void arithm_flip_rows(__global const uchar * srcptr, int src_step, int src_offset,
                               __global uchar * dstptr, int dst_step, int dst_offset,
                               int rows, int cols, int thread_rows, int thread_cols)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index0 = mad24(y0, src_step, mad24(x, TSIZE, src_offset));
        int src_index1 = mad24(rows - y0 - 1, src_step, mad24(x, TSIZE, src_offset));
        int dst_index0 = mad24(y0, dst_step, mad24(x, TSIZE, dst_offset));
        int dst_index1 = mad24(rows - y0 - 1, dst_step, mad24(x, TSIZE, dst_offset));

        #pragma unroll
        for (int y = y0, y1 = min(thread_rows, y0 + PIX_PER_WI_Y); y < y1; ++y)
        {
            T src0 = loadpix(srcptr + src_index0);
            T src1 = loadpix(srcptr + src_index1);

            storepix_2(src1, src0, dstptr + dst_index0, dstptr + dst_index1);

            src_index0 += src_step;
            src_index1 -= src_step;
            dst_index0 += dst_step;
            dst_index1 -= dst_step;
        }
    }
}

__kernel void arithm_flip_rows_cols(__global const uchar * srcptr, int src_step, int src_offset,
                                    __global uchar * dstptr, int dst_step, int dst_offset,
                                    int rows, int cols, int thread_rows, int thread_cols)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1)*PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index0 = mad24(y0, src_step, mad24(x, TSIZE, src_offset));
        int src_index1 = mad24(rows - y0 - 1, src_step, mad24(cols - x - 1, TSIZE, src_offset));
        int dst_index0 = mad24(y0, dst_step, mad24(x, TSIZE, dst_offset));
        int dst_index1 = mad24(rows - y0 - 1, dst_step, mad24(cols - x - 1, TSIZE, dst_offset));

        #pragma unroll
        for (int y = y0, y1 = min(thread_rows, y0 + PIX_PER_WI_Y); y < y1; ++y)
        {
            T src0 = loadpix(srcptr + src_index0);
            T src1 = loadpix(srcptr + src_index1);

#if kercn == 2
#if cn == 1
            src0 = src0.s10;
            src1 = src1.s10;
#endif
#elif kercn == 4
#if cn == 1
            src0 = src0.s3210;
            src1 = src1.s3210;
#elif cn == 2
            src0 = src0.s2301;
            src1 = src1.s2301;
#endif
#endif

            storepix_2(src1, src0, dstptr + dst_index0, dstptr + dst_index1);

            src_index0 += src_step;
            src_index1 -= src_step;
            dst_index0 += dst_step;
            dst_index1 -= dst_step;
        }
    }
}

__kernel void arithm_flip_cols(__global const uchar * srcptr, int src_step, int src_offset,
                               __global uchar * dstptr, int dst_step, int dst_offset,
                               int rows, int cols, int thread_rows, int thread_cols)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1)*PIX_PER_WI_Y;

    if (x < thread_cols)
    {
        int src_index0 = mad24(y0, src_step, mad24(x, TSIZE, src_offset));
        int src_index1 = mad24(y0, src_step, mad24(cols - x - 1, TSIZE, src_offset));
        int dst_index0 = mad24(y0, dst_step, mad24(x, TSIZE, dst_offset));
        int dst_index1 = mad24(y0, dst_step, mad24(cols - x - 1, TSIZE, dst_offset));

        #pragma unroll
        for (int y = y0, y1 = min(rows, y0 + PIX_PER_WI_Y); y < y1; ++y)
        {
            T src0 = loadpix(srcptr + src_index0);
            T src1 = loadpix(srcptr + src_index1);

#if kercn == 2
#if cn == 1
            src0 = src0.s10;
            src1 = src1.s10;
#endif
#elif kercn == 4
#if cn == 1
            src0 = src0.s3210;
            src1 = src1.s3210;
#elif cn == 2
            src0 = src0.s2301;
            src1 = src1.s2301;
#endif
#endif

            storepix_2(src1, src0, dstptr + dst_index0, dstptr + dst_index1);

            src_index0 += src_step;
            src_index1 += src_step;
            dst_index0 += dst_step;
            dst_index1 += dst_step;
        }
    }
}
