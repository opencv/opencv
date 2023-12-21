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
//    Zhang Chunpeng	chunpeng@multicorewareinc.com
//    Dachuan Zhao, dachuan@multicorewareinc.com
//    Yao Wang, yao@multicorewareinc.com
//    Peng Xiao, pengxiao@outlook.com
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

///////////////////////////////////////////////////////////////////////
////////////////////////  Generic PyrUp  //////////////////////////////
///////////////////////////////////////////////////////////////////////

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#if CN != 3
#define loadpix(addr)  *(__global const T*)(addr)
#define storepix(val, addr)  *(__global T*)(addr) = (val)
#define PIXSIZE ((int)sizeof(T))
#else
#define loadpix(addr)  vload3(0, (__global const T1*)(addr))
#define storepix(val, addr) vstore3((val), 0, (__global T1*)(addr))
#define PIXSIZE ((int)sizeof(T1)*3)
#endif

#define EXTRAPOLATE(x, maxV) min(maxV - 1, (int) abs(x))

#define noconvert

__kernel void pyrUp(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                         __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int tidx = get_local_id(0);
    const int tidy = get_local_id(1);

    __local FT s_srcPatch[LOCAL_SIZE/2 + 2][LOCAL_SIZE/2 + 2];
    __local FT s_dstPatch[LOCAL_SIZE/2 + 2][LOCAL_SIZE];

    __global uchar * dstData = dst + dst_offset;
    __global const uchar * srcData = src + src_offset;

    if( tidx < (LOCAL_SIZE/2 + 2) && tidy < LOCAL_SIZE/2 + 2 )
    {
        int srcx = EXTRAPOLATE(mad24((int)get_group_id(0), LOCAL_SIZE/2, tidx) - 1, src_cols);
        int srcy = EXTRAPOLATE(mad24((int)get_group_id(1), LOCAL_SIZE/2, tidy) - 1, src_rows);

        s_srcPatch[tidy][tidx] = CONVERT_TO_FT(loadpix(srcData + srcy * src_step + srcx * PIXSIZE));
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    FT sum = 0.f;

    const FT co1 = 0.75f;
    const FT co2 = 0.5f;
    const FT co3 = 0.125f;

    const FT coef1 = (tidx & 1) == 0 ? co1 : (FT) 0;
    const FT coef2 = (tidx & 1) == 0 ? co3 : co2;
    const FT coefy1 = (tidy & 1) == 0 ? co1 : (FT) 0;
    const FT coefy2 = (tidy & 1) == 0 ? co3 : co2;

    if(tidy < LOCAL_SIZE/2 + 2)
    {
        sum =     coef2* s_srcPatch[tidy][1 + ((tidx - 1) >> 1)];
        sum = mad(coef1, s_srcPatch[tidy][1 + ((tidx    ) >> 1)], sum);
        sum = mad(coef2, s_srcPatch[tidy][1 + ((tidx + 2) >> 1)], sum);

        s_dstPatch[tidy][tidx] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    sum =     coefy2* s_dstPatch[1 + ((tidy - 1) >> 1)][tidx];
    sum = mad(coefy1, s_dstPatch[1 + ((tidy    ) >> 1)][tidx], sum);
    sum = mad(coefy2, s_dstPatch[1 + ((tidy + 2) >> 1)][tidx], sum);

    if ((x < dst_cols) && (y < dst_rows))
        storepix(CONVERT_TO_T(sum), dstData + y * dst_step + x * PIXSIZE);
}


__kernel void pyrUp_unrolled(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                         __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    const int lx = 2*get_local_id(0);
    const int ly = 2*get_local_id(1);

    __local FT s_srcPatch[LOCAL_SIZE+2][LOCAL_SIZE+2];
    __local FT s_dstPatch[LOCAL_SIZE+2][2*LOCAL_SIZE];

    __global uchar * dstData = dst + dst_offset;
    __global const uchar * srcData = src + src_offset;

    if( lx < (LOCAL_SIZE+2) && ly < (LOCAL_SIZE+2) )
    {
        int srcx = mad24((int)get_group_id(0), LOCAL_SIZE, lx) - 1;
        int srcy = mad24((int)get_group_id(1), LOCAL_SIZE, ly) - 1;

        int srcx1 = EXTRAPOLATE(srcx, src_cols);
        int srcx2 = EXTRAPOLATE(srcx+1, src_cols);
        int srcy1 = EXTRAPOLATE(srcy, src_rows);
        int srcy2 = EXTRAPOLATE(srcy+1, src_rows);
        s_srcPatch[ly][lx] = CONVERT_TO_FT(loadpix(srcData + srcy1 * src_step + srcx1 * PIXSIZE));
        s_srcPatch[ly+1][lx] = CONVERT_TO_FT(loadpix(srcData + srcy2 * src_step + srcx1 * PIXSIZE));
        s_srcPatch[ly][lx+1] = CONVERT_TO_FT(loadpix(srcData + srcy1 * src_step + srcx2 * PIXSIZE));
        s_srcPatch[ly+1][lx+1] = CONVERT_TO_FT(loadpix(srcData + srcy2 * src_step + srcx2 * PIXSIZE));
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    FT sum;

    const FT co1 = 0.75f;
    const FT co2 = 0.5f;
    const FT co3 = 0.125f;

    // (x,y)
    sum =       co3 * s_srcPatch[1 + (ly >> 1)][1 + ((lx - 2) >> 1)];
    sum = mad(co1, s_srcPatch[1 + (ly >> 1)][1 + ((lx    ) >> 1)], sum);
    sum = mad(co3, s_srcPatch[1 + (ly >> 1)][1 + ((lx + 2) >> 1)], sum);

    s_dstPatch[1 + get_local_id(1)][lx] = sum;

    // (x+1,y)
    sum =       co2 * s_srcPatch[1 + (ly >> 1)][1 + ((lx + 1 - 1) >> 1)];
    sum = mad(co2, s_srcPatch[1 + (ly >> 1)][1 + ((lx + 1 + 1) >> 1)], sum);
    s_dstPatch[1 + get_local_id(1)][lx+1] = sum;

    if (ly < 1)
    {
        // (x,y)
        sum =       co3 * s_srcPatch[0][1 + ((lx - 2) >> 1)];
        sum = mad(co1, s_srcPatch[0][1 + ((lx    ) >> 1)], sum);
        sum = mad(co3, s_srcPatch[0][1 + ((lx + 2) >> 1)], sum);
        s_dstPatch[0][lx] = sum;

        // (x+1,y)
        sum =       co2 * s_srcPatch[0][1 + ((lx + 1 - 1) >> 1)];
        sum = mad(co2, s_srcPatch[0][1 + ((lx + 1 + 1) >> 1)], sum);
        s_dstPatch[0][lx+1] = sum;
    }

    if (ly > 2*LOCAL_SIZE-3)
    {
        // (x,y)
        sum =       co3 * s_srcPatch[LOCAL_SIZE+1][1 + ((lx - 2) >> 1)];
        sum = mad(co1, s_srcPatch[LOCAL_SIZE+1][1 + ((lx    ) >> 1)], sum);
        sum = mad(co3, s_srcPatch[LOCAL_SIZE+1][1 + ((lx + 2) >> 1)], sum);
        s_dstPatch[LOCAL_SIZE+1][lx] = sum;

        // (x+1,y)
        sum =       co2 * s_srcPatch[LOCAL_SIZE+1][1 + ((lx + 1 - 1) >> 1)];
        sum = mad(co2, s_srcPatch[LOCAL_SIZE+1][1 + ((lx + 1 + 1) >> 1)], sum);
        s_dstPatch[LOCAL_SIZE+1][lx+1] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int dst_x = 2*get_global_id(0);
    int dst_y = 2*get_global_id(1);

    if ((dst_x < dst_cols) && (dst_y < dst_rows))
    {
        // (x,y)
        sum =       co3 * s_dstPatch[1 + get_local_id(1) - 1][lx];
        sum = mad(co1, s_dstPatch[1 + get_local_id(1)    ][lx], sum);
        sum = mad(co3, s_dstPatch[1 + get_local_id(1) + 1][lx], sum);
        storepix(CONVERT_TO_T(sum), dstData + dst_y * dst_step + dst_x * PIXSIZE);

        // (x+1,y)
        sum =       co3 * s_dstPatch[1 + get_local_id(1) - 1][lx+1];
        sum = mad(co1, s_dstPatch[1 + get_local_id(1)    ][lx+1], sum);
        sum = mad(co3, s_dstPatch[1 + get_local_id(1) + 1][lx+1], sum);
        storepix(CONVERT_TO_T(sum), dstData + dst_y * dst_step + (dst_x+1) * PIXSIZE);

        // (x,y+1)
        sum =       co2 * s_dstPatch[1 + get_local_id(1)    ][lx];
        sum = mad(co2, s_dstPatch[1 + get_local_id(1) + 1][lx], sum);
        storepix(CONVERT_TO_T(sum), dstData + (dst_y+1) * dst_step + dst_x * PIXSIZE);

        // (x+1,y+1)
        sum =       co2 * s_dstPatch[1 + get_local_id(1)    ][lx+1];
        sum = mad(co2, s_dstPatch[1 + get_local_id(1) + 1][lx+1], sum);
        storepix(CONVERT_TO_T(sum), dstData + (dst_y+1) * dst_step + (dst_x+1) * PIXSIZE);
    }
}
