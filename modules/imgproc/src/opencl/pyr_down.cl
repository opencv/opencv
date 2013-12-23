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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Dachuan Zhao, dachuan@multicorewareinc.com
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
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define noconvert

inline int idx_row_low(int y, int last_row)
{
    return abs(y) % (last_row + 1);
}

inline int idx_row_high(int y, int last_row)
{
    return abs(last_row - (int)abs(last_row - y)) % (last_row + 1);
}

inline int idx_row(int y, int last_row)
{
    return idx_row_low(idx_row_high(y, last_row), last_row);
}

inline int idx_col_low(int x, int last_col)
{
    return abs(x) % (last_col + 1);
}

inline int idx_col_high(int x, int last_col)
{
    return abs(last_col - (int)abs(last_col - x)) % (last_col + 1);
}

inline int idx_col(int x, int last_col)
{
    return idx_col_low(idx_col_high(x, last_col), last_col);
}

__kernel void pyrDown(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                         __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    const int x = get_global_id(0);
    const int y = get_group_id(1);

    __local FT smem[256 + 4];
    __global T * dstData = (__global T *)(dst + dst_offset);
    __global const uchar * srcData = (__global const uchar*)(src + src_offset);

    FT sum;
    FT co1 = 0.375f;
    FT co2 = 0.25f;
    FT co3 = 0.0625f;

    const int src_y = 2*y;
    const int last_row = src_rows - 1;
    const int last_col = src_cols - 1;

    if (src_y >= 2 && src_y < src_rows - 2 && x >= 2 && x < src_cols - 2)
    {
        sum =       co3 * convertToFT(((__global T*)(srcData + (src_y - 2) * src_step))[x]);
        sum = sum + co2 * convertToFT(((__global T*)(srcData + (src_y - 1) * src_step))[x]);
        sum = sum + co1 * convertToFT(((__global T*)(srcData + (src_y    ) * src_step))[x]);
        sum = sum + co2 * convertToFT(((__global T*)(srcData + (src_y + 1) * src_step))[x]);
        sum = sum + co3 * convertToFT(((__global T*)(srcData + (src_y + 2) * src_step))[x]);

        smem[2 + get_local_id(0)] = sum;

        if (get_local_id(0) < 2)
        {
            const int left_x = x - 2;

            sum =       co3 * convertToFT(((__global T*)(srcData + (src_y - 2) * src_step))[left_x]);
            sum = sum + co2 * convertToFT(((__global T*)(srcData + (src_y - 1) * src_step))[left_x]);
            sum = sum + co1 * convertToFT(((__global T*)(srcData + (src_y    ) * src_step))[left_x]);
            sum = sum + co2 * convertToFT(((__global T*)(srcData + (src_y + 1) * src_step))[left_x]);
            sum = sum + co3 * convertToFT(((__global T*)(srcData + (src_y + 2) * src_step))[left_x]);

            smem[get_local_id(0)] = sum;
        }

        if (get_local_id(0) > 253)
        {
            const int right_x = x + 2;

            sum =       co3 * convertToFT(((__global T*)(srcData + (src_y - 2) * src_step))[right_x]);
            sum = sum + co2 * convertToFT(((__global T*)(srcData + (src_y - 1) * src_step))[right_x]);
            sum = sum + co1 * convertToFT(((__global T*)(srcData + (src_y    ) * src_step))[right_x]);
            sum = sum + co2 * convertToFT(((__global T*)(srcData + (src_y + 1) * src_step))[right_x]);
            sum = sum + co3 * convertToFT(((__global T*)(srcData + (src_y + 2) * src_step))[right_x]);

            smem[4 + get_local_id(0)] = sum;
        }
    }
    else
    {
        int col = idx_col(x, last_col);

        sum =       co3 * convertToFT(((__global T*)(srcData + idx_row(src_y - 2, last_row) * src_step))[col]);
        sum = sum + co2 * convertToFT(((__global T*)(srcData + idx_row(src_y - 1, last_row) * src_step))[col]);
        sum = sum + co1 * convertToFT(((__global T*)(srcData + idx_row(src_y    , last_row) * src_step))[col]);
        sum = sum + co2 * convertToFT(((__global T*)(srcData + idx_row(src_y + 1, last_row) * src_step))[col]);
        sum = sum + co3 * convertToFT(((__global T*)(srcData + idx_row(src_y + 2, last_row) * src_step))[col]);

        smem[2 + get_local_id(0)] = sum;

        if (get_local_id(0) < 2)
        {
            const int left_x = x - 2;

            col = idx_col(left_x, last_col);

            sum =       co3 * convertToFT(((__global T*)(srcData + idx_row(src_y - 2, last_row) * src_step))[col]);
            sum = sum + co2 * convertToFT(((__global T*)(srcData + idx_row(src_y - 1, last_row) * src_step))[col]);
            sum = sum + co1 * convertToFT(((__global T*)(srcData + idx_row(src_y    , last_row) * src_step))[col]);
            sum = sum + co2 * convertToFT(((__global T*)(srcData + idx_row(src_y + 1, last_row) * src_step))[col]);
            sum = sum + co3 * convertToFT(((__global T*)(srcData + idx_row(src_y + 2, last_row) * src_step))[col]);

            smem[get_local_id(0)] = sum;
        }

        if (get_local_id(0) > 253)
        {
            const int right_x = x + 2;

            col = idx_col(right_x, last_col);

            sum =       co3 * convertToFT(((__global T*)(srcData + idx_row(src_y - 2, last_row) * src_step))[col]);
            sum = sum + co2 * convertToFT(((__global T*)(srcData + idx_row(src_y - 1, last_row) * src_step))[col]);
            sum = sum + co1 * convertToFT(((__global T*)(srcData + idx_row(src_y    , last_row) * src_step))[col]);
            sum = sum + co2 * convertToFT(((__global T*)(srcData + idx_row(src_y + 1, last_row) * src_step))[col]);
            sum = sum + co3 * convertToFT(((__global T*)(srcData + idx_row(src_y + 2, last_row) * src_step))[col]);

            smem[4 + get_local_id(0)] = sum;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < 128)
    {
        const int tid2 = get_local_id(0) * 2;

        sum =       co3 * smem[2 + tid2 - 2];
        sum = sum + co2 * smem[2 + tid2 - 1];
        sum = sum + co1 * smem[2 + tid2    ];
        sum = sum + co2 * smem[2 + tid2 + 1];
        sum = sum + co3 * smem[2 + tid2 + 2];

        const int dst_x = (get_group_id(0) * get_local_size(0) + tid2) / 2;

        if (dst_x < dst_cols)
            dstData[y * dst_step / ((int)sizeof(T)) + dst_x] = convertToT(sum);
    }

}
