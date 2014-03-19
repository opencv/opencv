//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
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
//

#define READ_TIMES_COL ((2*(RADIUSY+LSIZE1)-1)/LSIZE1)
#define RADIUS 1

#define noconvert

/**********************************************************************************
These kernels are written for separable filters such as Sobel, Scharr, GaussianBlur.
Now(6/29/2011) the kernels only support 8U data type and the anchor of the convovle
kernel must be in the center. ROI is not supported either.
Each kernels read 4 elements(not 4 pixels), save them to LDS and read the data needed
from LDS to calculate the result.
The length of the convovle kernel supported is only related to the MAX size of LDS,
which is HW related.
Niko
6/29/2011
The info above maybe obsolete.
***********************************************************************************/

#define DIG(a) a,
__constant float mat_kernel[] = { COEFF };

__kernel void col_filter(__global const srcT * src, int src_step_in_pixel, int src_whole_cols, int src_whole_rows,
                         __global dstT * dst, int dst_offset_in_pixel, int dst_step_in_pixel, int dst_cols, int dst_rows)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int l_x = get_local_id(0);
    int l_y = get_local_id(1);

    int start_addr = mad24(y, src_step_in_pixel, x);
    int end_addr = mad24(src_whole_rows - 1, src_step_in_pixel, src_whole_cols);

    srcT sum, temp[READ_TIMES_COL];
    __local srcT LDS_DAT[LSIZE1 * READ_TIMES_COL][LSIZE0 + 1];

    // read pixels from src
    for (int i = 0; i < READ_TIMES_COL; ++i)
    {
        int current_addr = mad24(i, LSIZE1 * src_step_in_pixel, start_addr);
        current_addr = current_addr < end_addr ? current_addr : 0;
        temp[i] = src[current_addr];
    }

    // save pixels to lds
    for (int i = 0; i < READ_TIMES_COL; ++i)
        LDS_DAT[mad24(i, LSIZE1, l_y)][l_x] = temp[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // read pixels from lds and calculate the result
    sum = LDS_DAT[l_y + RADIUSY][l_x] * mat_kernel[RADIUSY];
    for (int i = 1; i <= RADIUSY; ++i)
    {
        temp[0] = LDS_DAT[l_y + RADIUSY - i][l_x];
        temp[1] = LDS_DAT[l_y + RADIUSY + i][l_x];
        sum += mad(temp[0], mat_kernel[RADIUSY - i], temp[1] * mat_kernel[RADIUSY + i]);
    }

    // write the result to dst
    if (x < dst_cols && y < dst_rows)
    {
        start_addr = mad24(y, dst_step_in_pixel, x + dst_offset_in_pixel);
        dst[start_addr] = convertToDstT(sum);
    }
}
