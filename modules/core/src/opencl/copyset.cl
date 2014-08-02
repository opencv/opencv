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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
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
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifdef COPY_TO_MASK

#define DEFINE_DATA \
    int src_index = mad24(y, src_step, mad24(x, (int)sizeof(T1) * scn, src_offset)); \
    int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(T1) * scn, dst_offset)); \
     \
    __global const T1 * src = (__global const T1 *)(srcptr + src_index); \
    __global T1 * dst = (__global T1 *)(dstptr + dst_index)

__kernel void copyToMask(__global const uchar * srcptr, int src_step, int src_offset,
                         __global const uchar * mask, int mask_step, int mask_offset,
                         __global uchar * dstptr, int dst_step, int dst_offset,
                         int dst_rows, int dst_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        mask += mad24(y, mask_step, mad24(x, mcn, mask_offset));

#if mcn == 1
        if (mask[0])
        {
            DEFINE_DATA;

            #pragma unroll
            for (int c = 0; c < scn; ++c)
                dst[c] = src[c];
        }
#ifdef HAVE_DST_UNINIT
        else
        {
            DEFINE_DATA;

            #pragma unroll
            for (int c = 0; c < scn; ++c)
                dst[c] = (T1)(0);
        }
#endif
#elif scn == mcn
        DEFINE_DATA;

        #pragma unroll
        for (int c = 0; c < scn; ++c)
            if (mask[c])
                dst[c] = src[c];
#ifdef HAVE_DST_UNINIT
            else
                dst[c] = (T1)(0);
#endif
#else
#error "(mcn == 1 || mcn == scn) should be true"
#endif
    }
}

#else

#ifndef dstST
#define dstST dstT
#endif

#if cn != 3
#define value value_
#define storedst(val) *(__global dstT *)(dstptr + dst_index) = val
#else
#define value (dstT)(value_.x, value_.y, value_.z)
#define storedst(val) vstore3(val, 0, (__global dstT1 *)(dstptr + dst_index))
#endif

__kernel void setMask(__global const uchar* mask, int maskstep, int maskoffset,
                      __global uchar* dstptr, int dststep, int dstoffset,
                      int rows, int cols, dstST value_)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * rowsPerWI;

    if (x < cols)
    {
        int mask_index = mad24(y0, maskstep, x + maskoffset);
        int dst_index  = mad24(y0, dststep, mad24(x, (int)sizeof(dstT1) * cn, dstoffset));

        for (int y = y0, y1 = min(rows, y0 + rowsPerWI); y < y1; ++y)
        {
            if( mask[mask_index] )
                storedst(value);

            mask_index += maskstep;
            dst_index += dststep;
        }
    }
}

__kernel void set(__global uchar* dstptr, int dststep, int dstoffset,
                  int rows, int cols, dstST value_)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * rowsPerWI;

    if (x < cols)
    {
        int dst_index  = mad24(y0, dststep, mad24(x, (int)sizeof(dstT1) * cn, dstoffset));

        for (int y = y0, y1 = min(rows, y0 + rowsPerWI); y < y1; ++y, dst_index += dststep)
            storedst(value);
    }
}

#endif
