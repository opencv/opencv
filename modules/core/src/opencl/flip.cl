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

#if cn != 3
#define loadpix(addr) *(__global const T *)(addr)
#define storepix(val, addr)  *(__global T *)(addr) = val
#define TSIZE (int)sizeof(T)
#else
#define loadpix(addr) vload3(0, (__global const T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#define TSIZE ((int)sizeof(T1)*3)
#endif

__kernel void arithm_flip_rows(__global const uchar * srcptr, int src_step, int src_offset,
                               __global uchar * dstptr, int dst_step, int dst_offset,
                               int rows, int cols, int thread_rows, int thread_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < thread_rows)
    {
        T src0 = loadpix(srcptr + mad24(y, src_step, mad24(x, TSIZE, src_offset)));
        T src1 = loadpix(srcptr + mad24(rows - y - 1, src_step, mad24(x, TSIZE, src_offset)));

        storepix(src1, dstptr + mad24(y, dst_step, mad24(x, TSIZE, dst_offset)));
        storepix(src0, dstptr + mad24(rows - y - 1, dst_step, mad24(x, TSIZE, dst_offset)));
    }
}

__kernel void arithm_flip_rows_cols(__global const uchar * srcptr, int src_step, int src_offset,
                                    __global uchar * dstptr, int dst_step, int dst_offset,
                                    int rows, int cols, int thread_rows, int thread_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < thread_rows)
    {
        int x1 = cols - x - 1;
        T src0 = loadpix(srcptr + mad24(y, src_step, mad24(x, TSIZE, src_offset)));
        T src1 = loadpix(srcptr + mad24(rows - y - 1, src_step, mad24(x1, TSIZE, src_offset)));

        storepix(src0, dstptr + mad24(rows - y - 1, dst_step, mad24(x1, TSIZE, dst_offset)));
        storepix(src1, dstptr + mad24(y, dst_step, mad24(x, TSIZE, dst_offset)));
    }
}

__kernel void arithm_flip_cols(__global const uchar * srcptr, int src_step, int src_offset,
                               __global uchar * dstptr, int dst_step, int dst_offset,
                               int rows, int cols, int thread_rows, int thread_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < thread_cols && y < rows)
    {
        int x1 = cols - x - 1;
        T src0 = loadpix(srcptr + mad24(y, src_step, mad24(x, TSIZE, src_offset)));
        T src1 = loadpix(srcptr + mad24(y, src_step, mad24(x1, TSIZE, src_offset)));

        storepix(src0, dstptr + mad24(y, dst_step, mad24(x1, TSIZE, dst_offset)));
        storepix(src1, dstptr + mad24(y, dst_step, mad24(x, TSIZE, dst_offset)));
    }
}
