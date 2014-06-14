//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Zero Lin, zero.lin@amd.com
//    Yao Wang, bitwangyaoyao@gmail.com
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

#if cn != 3
#define loadpix(addr) *(__global const T *)(addr)
#define storepix(val, addr)  *(__global T *)(addr) = val
#define TSIZE (int)sizeof(T)
#else
#define loadpix(addr) vload3(0, (__global const T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#define TSIZE ((int)sizeof(T1)*3)
#endif

#ifdef DEPTH_0
#ifdef ERODE
#define VAL 255
#endif
#ifdef DILATE
#define VAL 0
#endif
#elif defined DEPTH_5
#ifdef ERODE
#define VAL FLT_MAX
#endif
#ifdef DILATE
#define VAL -FLT_MAX
#endif
#elif defined DEPTH_6
#ifdef ERODE
#define VAL DBL_MAX
#endif
#ifdef DILATE
#define VAL -DBL_MAX
#endif
#endif

#ifdef ERODE
#if defined(INTEL_DEVICE) && (DEPTH_0)
// workaround for bug in Intel HD graphics drivers (10.18.10.3496 or older)
#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)
#define WA_CONVERT_1 CAT(convert_uint, cn)
#define WA_CONVERT_2 CAT(convert_, T)
#define convert_uint1 convert_uint
#define MORPH_OP(A,B) WA_CONVERT_2(min(WA_CONVERT_1(A),WA_CONVERT_1(B)))
#else
#define MORPH_OP(A,B) min((A),(B))
#endif
#endif
#ifdef DILATE
#define MORPH_OP(A,B) max((A),(B))
#endif

// BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii
#define ELEM(i, l_edge, r_edge, elem1, elem2) (i) < (l_edge) | (i) >= (r_edge) ? (elem1) : (elem2)

__kernel void morph(__global const uchar * srcptr, int src_step, int src_offset,
                    __global uchar * dstptr, int dst_step, int dst_offset,
                    int src_offset_x, int src_offset_y, int cols, int rows,
                    __constant uchar * mat_kernel, int src_whole_cols, int src_whole_rows)
{
    int gidx = get_global_id(0), gidy = get_global_id(1);
    int l_x = get_local_id(0), l_y = get_local_id(1);
    int x = get_group_id(0) * LSIZE0, y = get_group_id(1) * LSIZE1;
    int start_x = x + src_offset_x - RADIUSX;
    int end_x = x + src_offset_x + LSIZE0 + RADIUSX;
    int width = end_x - (x + src_offset_x - RADIUSX) + 1;
    int start_y = y + src_offset_y - RADIUSY;
    int point1 = mad24(l_y, LSIZE0, l_x);
    int point2 = point1 + LSIZE0 * LSIZE1;
    int tl_x = point1 % width, tl_y = point1 / width;
    int tl_x2 = point2 % width, tl_y2 = point2 / width;
    int cur_x = start_x + tl_x, cur_y = start_y + tl_y;
    int cur_x2 = start_x + tl_x2, cur_y2 = start_y + tl_y2;
    int start_addr = mad24(cur_y, src_step, cur_x * TSIZE);
    int start_addr2 = mad24(cur_y2, src_step, cur_x2 * TSIZE);

    __local T LDS_DAT[2*LSIZE1*LSIZE0];

    // read pixels from src
    int end_addr = mad24(src_whole_rows - 1, src_step, src_whole_cols * TSIZE);
    start_addr = start_addr < end_addr && start_addr > 0 ? start_addr : 0;
    start_addr2 = start_addr2 < end_addr && start_addr2 > 0 ? start_addr2 : 0;

    T temp0 = loadpix(srcptr + start_addr);
    T temp1 = loadpix(srcptr + start_addr2);

    // judge if read out of boundary
    temp0 = ELEM(cur_x, 0, src_whole_cols, (T)(VAL),temp0);
    temp0 = ELEM(cur_y, 0, src_whole_rows, (T)(VAL),temp0);

    temp1 = ELEM(cur_x2, 0, src_whole_cols, (T)(VAL), temp1);
    temp1 = ELEM(cur_y2, 0, src_whole_rows, (T)(VAL), temp1);

    LDS_DAT[point1] = temp0;
    LDS_DAT[point2] = temp1;
    barrier(CLK_LOCAL_MEM_FENCE);

    T res = (T)(VAL);
    for (int i = 0, sizey = 2 * RADIUSY + 1; i < sizey; i++)
        for (int j = 0, sizex = 2 * RADIUSX + 1; j < sizex; j++)
        {
            res =
#ifndef RECTKERNEL
                mat_kernel[i*(2*RADIUSX+1)+j] ?
#endif
                MORPH_OP(res, LDS_DAT[mad24(l_y + i, width, l_x + j)])
#ifndef RECTKERNEL
                : res
#endif
                ;
        }

    if (gidx < cols && gidy < rows)
    {
        int dst_index = mad24(gidy, dst_step, mad24(gidx, TSIZE, dst_offset));
        storepix(res, dstptr + dst_index);
    }
}
