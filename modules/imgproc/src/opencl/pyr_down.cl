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

#if defined BORDER_REPLICATE
// aaaaaa|abcdefgh|hhhhhhh
#define EXTRAPOLATE(x, maxV) clamp(x, 0, maxV-1)
#elif defined BORDER_WRAP
// cdefgh|abcdefgh|abcdefg
#define EXTRAPOLATE(x, maxV) ( (x) + (maxV) ) % (maxV)
#elif defined BORDER_REFLECT
// fedcba|abcdefgh|hgfedcb
#define EXTRAPOLATE(x, maxV) min(((maxV)-1)*2-(x)+1, max((x),-(x)-1) )
#elif defined BORDER_REFLECT_101 || defined BORDER_REFLECT101
// gfedcb|abcdefgh|gfedcba
#define EXTRAPOLATE(x, maxV) min(((maxV)-1)*2-(x), max((x),-(x)) )
#else
#error No extrapolation method
#endif

#if cn != 3
#define loadpix(addr)  *(__global const T*)(addr)
#define storepix(val, addr)  *(__global T*)(addr) = (val)
#define PIXSIZE ((int)sizeof(T))
#else
#define loadpix(addr)  vload3(0, (__global const T1*)(addr))
#define storepix(val, addr) vstore3((val), 0, (__global T1*)(addr))
#define PIXSIZE ((int)sizeof(T1)*3)
#endif

#define SRC(_x,_y) convertToFT(loadpix(srcData + mad24(_y, src_step, PIXSIZE * _x)))

#define noconvert

__kernel void pyrDown(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                         __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    const int x = get_global_id(0);
    const int y = get_group_id(1);

    __local FT smem[LOCAL_SIZE + 4];
    __global uchar * dstData = dst + dst_offset;
    __global const uchar * srcData = src + src_offset;

    FT sum;
    FT co1 = 0.375f;
    FT co2 = 0.25f;
    FT co3 = 0.0625f;

    const int src_y = 2*y;

    if (src_y >= 2 && src_y < src_rows - 2 && x >= 2 && x < src_cols - 2)
    {
        sum =       co3 * SRC(x, src_y - 2);
        sum = sum + co2 * SRC(x, src_y - 1);
        sum = sum + co1 * SRC(x, src_y    );
        sum = sum + co2 * SRC(x, src_y + 1);
        sum = sum + co3 * SRC(x, src_y + 2);

        smem[2 + get_local_id(0)] = sum;

        if (get_local_id(0) < 2)
        {
            const int left_x = x - 2;

            sum =       co3 * SRC(left_x, src_y - 2);
            sum = sum + co2 * SRC(left_x, src_y - 1);
            sum = sum + co1 * SRC(left_x, src_y    );
            sum = sum + co2 * SRC(left_x, src_y + 1);
            sum = sum + co3 * SRC(left_x, src_y + 2);

            smem[get_local_id(0)] = sum;
        }

        if (get_local_id(0) > LOCAL_SIZE - 3)
        {
            const int right_x = x + 2;

            sum =       co3 * SRC(right_x, src_y - 2);
            sum = sum + co2 * SRC(right_x, src_y - 1);
            sum = sum + co1 * SRC(right_x, src_y    );
            sum = sum + co2 * SRC(right_x, src_y + 1);
            sum = sum + co3 * SRC(right_x, src_y + 2);

            smem[4 + get_local_id(0)] = sum;
        }
    }
    else
    {
        int col = EXTRAPOLATE(x, src_cols);

        sum =       co3 * SRC(col, EXTRAPOLATE(src_y - 2, src_rows));
        sum = sum + co2 * SRC(col, EXTRAPOLATE(src_y - 1, src_rows));
        sum = sum + co1 * SRC(col, EXTRAPOLATE(src_y    , src_rows));
        sum = sum + co2 * SRC(col, EXTRAPOLATE(src_y + 1, src_rows));
        sum = sum + co3 * SRC(col, EXTRAPOLATE(src_y + 2, src_rows));

        smem[2 + get_local_id(0)] = sum;

        if (get_local_id(0) < 2)
        {
            col = EXTRAPOLATE(x - 2, src_cols);

            sum =       co3 * SRC(col, EXTRAPOLATE(src_y - 2, src_rows));
            sum = sum + co2 * SRC(col, EXTRAPOLATE(src_y - 1, src_rows));
            sum = sum + co1 * SRC(col, EXTRAPOLATE(src_y    , src_rows));
            sum = sum + co2 * SRC(col, EXTRAPOLATE(src_y + 1, src_rows));
            sum = sum + co3 * SRC(col, EXTRAPOLATE(src_y + 2, src_rows));

            smem[get_local_id(0)] = sum;
        }

        if (get_local_id(0) > LOCAL_SIZE - 3)
        {
            col = EXTRAPOLATE(x + 2, src_cols);

            sum =       co3 * SRC(col, EXTRAPOLATE(src_y - 2, src_rows));
            sum = sum + co2 * SRC(col, EXTRAPOLATE(src_y - 1, src_rows));
            sum = sum + co1 * SRC(col, EXTRAPOLATE(src_y    , src_rows));
            sum = sum + co2 * SRC(col, EXTRAPOLATE(src_y + 1, src_rows));
            sum = sum + co3 * SRC(col, EXTRAPOLATE(src_y + 2, src_rows));

            smem[4 + get_local_id(0)] = sum;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (get_local_id(0) < LOCAL_SIZE / 2)
    {
        const int tid2 = get_local_id(0) * 2;

        sum =       co3 * smem[2 + tid2 - 2];
        sum = sum + co2 * smem[2 + tid2 - 1];
        sum = sum + co1 * smem[2 + tid2    ];
        sum = sum + co2 * smem[2 + tid2 + 1];
        sum = sum + co3 * smem[2 + tid2 + 2];

        const int dst_x = (get_group_id(0) * get_local_size(0) + tid2) / 2;

        if (dst_x < dst_cols)
            storepix(convertToT(sum), dstData + y * dst_step + dst_x * PIXSIZE);
    }

}
