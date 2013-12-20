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
    int src_index = mad24(y, src_step, x*(int)sizeof(T)*scn + src_offset); \
    int dst_index = mad24(y, dst_step, x*(int)sizeof(T)*scn + dst_offset); \
     \
    __global const T * src = (__global const T *)(srcptr + src_index); \
    __global T * dst = (__global T *)(dstptr + dst_index)

__kernel void copyToMask(__global const uchar * srcptr, int src_step, int src_offset,
                         __global const uchar * maskptr, int mask_step, int mask_offset,
                         __global uchar * dstptr, int dst_step, int dst_offset,
                         int dst_rows, int dst_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int mask_index = mad24(y, mask_step, x * mcn + mask_offset);
        __global const uchar * mask = (__global const uchar *)(maskptr + mask_index);

#if mcn == 1
        if (mask[0])
        {
            DEFINE_DATA;

            #pragma unroll
            for (int c = 0; c < scn; ++c)
                dst[c] = src[c];
        }
#elif scn == mcn
        DEFINE_DATA;

        #pragma unroll
        for (int c = 0; c < scn; ++c)
            if (mask[c])
                dst[c] = src[c];
#else
#error "(mcn == 1 || mcn == scn) should be true"
#endif
    }
}

#else

__kernel void setMask(__global const uchar* mask, int maskstep, int maskoffset,
                      __global uchar* dstptr, int dststep, int dstoffset,
                      int rows, int cols, dstT value )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int mask_index = mad24(y, maskstep, x + maskoffset);
        if( mask[mask_index] )
        {
            int dst_index  = mad24(y, dststep, x*(int)sizeof(dstT) + dstoffset);
            *(__global dstT*)(dstptr + dst_index) = value;
        }
    }
}

__kernel void set(__global uchar* dstptr, int dststep, int dstoffset,
                  int rows, int cols, dstT value )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int dst_index  = mad24(y, dststep, x*(int)sizeof(dstT) + dstoffset);
        *(__global dstT*)(dstptr + dst_index) = value;
    }
}

#endif
