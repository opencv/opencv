//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2014, Itseez, Inc, all rights reserved.
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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define READ_TIMES_COL ((2*(RADIUSY+LSIZE1)-1)/LSIZE1)
#define RADIUS 1

#define noconvert

#if CN != 3
#define loadpix(addr) *(__global const srcT *)(addr)
#define storepix(val, addr)  *(__global dstT *)(addr) = val
#define SRCSIZE (int)sizeof(srcT)
#define DSTSIZE (int)sizeof(dstT)
#else
#define loadpix(addr)  vload3(0, (__global const srcT1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global dstT1 *)(addr))
#define SRCSIZE (int)sizeof(srcT1)*3
#define DSTSIZE (int)sizeof(dstT1)*3
#endif

#define DIG(a) a,
__constant srcT1 mat_kernel[] = { COEFF };

__kernel void col_filter(__global const uchar * src, int src_step, int src_offset, int src_whole_rows, int src_whole_cols,
                         __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols, float delta)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    int l_x = get_local_id(0);
    int l_y = get_local_id(1);

    int start_addr = mad24(y, src_step, x * SRCSIZE);
    int end_addr = mad24(src_whole_rows - 1, src_step, src_whole_cols * SRCSIZE);

    srcT sum, temp[READ_TIMES_COL];
    __local srcT LDS_DAT[LSIZE1 * READ_TIMES_COL][LSIZE0 + 1];

    // read pixels from src
    for (int i = 0; i < READ_TIMES_COL; ++i)
    {
        int current_addr = mad24(i, LSIZE1 * src_step, start_addr);
        current_addr = current_addr < end_addr ? current_addr : 0;
        temp[i] = loadpix(src + current_addr);
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
#if (defined(INTEGER_ARITHMETIC) && !INTEL_DEVICE)
        sum += mad24(temp[0],mat_kernel[RADIUSY - i], temp[1] * mat_kernel[RADIUSY + i]);
#else
        sum += mad(temp[0], mat_kernel[RADIUSY - i], temp[1] * mat_kernel[RADIUSY + i]);
#endif
    }

#ifdef INTEGER_ARITHMETIC
#ifdef INTEL_DEVICE
    sum = (sum + (1 << (SHIFT_BITS-1))) / (1 << SHIFT_BITS);
#else
    sum = (sum + (1 << (SHIFT_BITS-1))) >> SHIFT_BITS;
#endif
#endif

    // write the result to dst
    if (x < dst_cols && y < dst_rows)
    {
        start_addr = mad24(y, dst_step, mad24(DSTSIZE, x, dst_offset));
        storepix(convertToDstT(sum + (srcT)(delta)), dst + start_addr);
    }
}
