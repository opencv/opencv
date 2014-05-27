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

#if cn != 3
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

    const int lsizex = get_local_size(0);
    const int lsizey = get_local_size(1);

    const int tidx = get_local_id(0);
    const int tidy = get_local_id(1);

    __local FT s_srcPatch[10][10];
    __local FT s_dstPatch[20][16];

    __global uchar * dstData = dst + dst_offset;
    __global const uchar * srcData = src + src_offset;

    if( tidx < 10 && tidy < 10 )
    {
        int srcx = mad24((int)get_group_id(0), lsizex>>1, tidx) - 1;
        int srcy = mad24((int)get_group_id(1), lsizey>>1, tidy) - 1;

        srcx = abs(srcx);
        srcx = min(src_cols - 1, srcx);

        srcy = abs(srcy);
        srcy = min(src_rows - 1, srcy);

        s_srcPatch[tidy][tidx] = convertToFT(loadpix(srcData + srcy * src_step + srcx * PIXSIZE));
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    FT sum = 0.f;
    const FT evenFlag = (FT)((tidx & 1) == 0);
    const FT  oddFlag = (FT)((tidx & 1) != 0);
    const bool  eveny = ((tidy & 1) == 0);

    const FT co1 = 0.75f;
    const FT co2 = 0.5f;
    const FT co3 = 0.125f;

    if(eveny)
    {
        sum =       ( evenFlag* co3 ) * s_srcPatch[1 + (tidy >> 1)][1 + ((tidx - 2) >> 1)];
        sum = sum + ( oddFlag * co2 ) * s_srcPatch[1 + (tidy >> 1)][1 + ((tidx - 1) >> 1)];
        sum = sum + ( evenFlag* co1 ) * s_srcPatch[1 + (tidy >> 1)][1 + ((tidx    ) >> 1)];
        sum = sum + ( oddFlag * co2 ) * s_srcPatch[1 + (tidy >> 1)][1 + ((tidx + 1) >> 1)];
        sum = sum + ( evenFlag* co3 ) * s_srcPatch[1 + (tidy >> 1)][1 + ((tidx + 2) >> 1)];
    }

    s_dstPatch[2 + tidy][tidx] = sum;

    if (tidy < 2)
    {
        sum = 0;

        if (eveny)
        {
            sum =       (evenFlag * co3 ) * s_srcPatch[lsizey-16][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * co2 ) * s_srcPatch[lsizey-16][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * co1 ) * s_srcPatch[lsizey-16][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * co2 ) * s_srcPatch[lsizey-16][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag * co3 ) * s_srcPatch[lsizey-16][1 + ((tidx + 2) >> 1)];
        }

        s_dstPatch[tidy][tidx] = sum;
    }

    if (tidy > 13)
    {
        sum = 0;

        if (eveny)
        {
            sum =       (evenFlag * co3) * s_srcPatch[lsizey-7][1 + ((tidx - 2) >> 1)];
            sum = sum + ( oddFlag * co2) * s_srcPatch[lsizey-7][1 + ((tidx - 1) >> 1)];
            sum = sum + (evenFlag * co1) * s_srcPatch[lsizey-7][1 + ((tidx    ) >> 1)];
            sum = sum + ( oddFlag * co2) * s_srcPatch[lsizey-7][1 + ((tidx + 1) >> 1)];
            sum = sum + (evenFlag * co3) * s_srcPatch[lsizey-7][1 + ((tidx + 2) >> 1)];
        }
        s_dstPatch[4 + tidy][tidx] = sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    sum =       co3 * s_dstPatch[2 + tidy - 2][tidx];
    sum = sum + co2 * s_dstPatch[2 + tidy - 1][tidx];
    sum = sum + co1 * s_dstPatch[2 + tidy    ][tidx];
    sum = sum + co2 * s_dstPatch[2 + tidy + 1][tidx];
    sum = sum + co3 * s_dstPatch[2 + tidy + 2][tidx];

    if ((x < dst_cols) && (y < dst_rows))
        storepix(convertToT(sum), dstData + y * dst_step + x * PIXSIZE);
}


__kernel void pyrUp_unrolled(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                         __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    const int lx = 2*get_local_id(0);
    const int ly = 2*get_local_id(1);

    __local FT s_srcPatch[LOCAL_SIZE+2][LOCAL_SIZE+2];
    __local FT s_dstPatch[2*LOCAL_SIZE+4][2*LOCAL_SIZE];

    __global uchar * dstData = dst + dst_offset;
    __global const uchar * srcData = src + src_offset;

    if( lx < (LOCAL_SIZE+2) && lx < (LOCAL_SIZE+2) )
    {
        int srcx = mad24((int)get_group_id(0), LOCAL_SIZE, lx) - 1;
        int srcy = mad24((int)get_group_id(1), LOCAL_SIZE, ly) - 1;

        int srcx1 = EXTRAPOLATE(srcx, src_cols);
        int srcx2 = EXTRAPOLATE(srcx+1, src_cols);
        int srcy1 = EXTRAPOLATE(srcy, src_rows);
        int srcy2 = EXTRAPOLATE(srcy+1, src_rows);
        s_srcPatch[ly][lx] = convertToFT(loadpix(srcData + srcy1 * src_step + srcx1 * PIXSIZE));
        s_srcPatch[ly+1][lx] = convertToFT(loadpix(srcData + srcy2 * src_step + srcx1 * PIXSIZE));
        s_srcPatch[ly][lx+1] = convertToFT(loadpix(srcData + srcy1 * src_step + srcx2 * PIXSIZE));
        s_srcPatch[ly+1][lx+1] = convertToFT(loadpix(srcData + srcy2 * src_step + srcx2 * PIXSIZE));
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    FT sum;

    const FT co1 = 0.75f;
    const FT co2 = 0.5f;
    const FT co3 = 0.125f;

    // (x,y)
    sum =       co3 * s_srcPatch[1 + (ly >> 1)][1 + ((lx - 2) >> 1)];
    sum = sum + co1 * s_srcPatch[1 + (ly >> 1)][1 + ((lx    ) >> 1)];
    sum = sum + co3 * s_srcPatch[1 + (ly >> 1)][1 + ((lx + 2) >> 1)];
    
    s_dstPatch[2 + ly][lx] = sum;

    // (x+1,y)
    sum =       co2 * s_srcPatch[1 + (ly >> 1)][1 + ((lx + 1 - 1) >> 1)];
    sum = sum + co2 * s_srcPatch[1 + (ly >> 1)][1 + ((lx + 1 + 1) >> 1)];
    s_dstPatch[2 + ly][lx+1] = sum;

    // (x, y+1) (x+1, y+1)
    s_dstPatch[2 + ly+1][lx] = 0.f;
    s_dstPatch[2 + ly+1][lx+1] = 0.f;

    if (ly < 1)
    {
        // (x,y)
        sum =       co3 * s_srcPatch[0][1 + ((lx - 2) >> 1)];
        sum = sum + co1 * s_srcPatch[0][1 + ((lx    ) >> 1)];
        sum = sum + co3 * s_srcPatch[0][1 + ((lx + 2) >> 1)];
        s_dstPatch[ly][lx] = sum;
        
        // (x+1,y)
        sum =       co2 * s_srcPatch[0][1 + ((lx + 1 - 1) >> 1)];
        sum = sum + co2 * s_srcPatch[0][1 + ((lx + 1 + 1) >> 1)];
        s_dstPatch[ly][lx+1] = sum;

        // (x, y+1) (x+1, y+1)
        s_dstPatch[ly+1][lx] = 0.f;
        s_dstPatch[ly+1][lx+1] = 0.f;
    }

    if (ly > 2*LOCAL_SIZE-3)
    {
        // (x,y)
        sum =       co3 * s_srcPatch[LOCAL_SIZE+1][1 + ((lx - 2) >> 1)];
        sum = sum + co1 * s_srcPatch[LOCAL_SIZE+1][1 + ((lx    ) >> 1)];
        sum = sum + co3 * s_srcPatch[LOCAL_SIZE+1][1 + ((lx + 2) >> 1)];
        s_dstPatch[4 + ly][lx] = sum;

        // (x+1,y)
        sum =       co2 * s_srcPatch[LOCAL_SIZE+1][1 + ((lx + 1 - 1) >> 1)];
        sum = sum + co2 * s_srcPatch[LOCAL_SIZE+1][1 + ((lx + 1 + 1) >> 1)];
        s_dstPatch[4 + ly][lx+1] = sum;

        // (x, y+1) (x+1, y+1)
        s_dstPatch[4 + ly+1][lx] = 0.f;
        s_dstPatch[4 + ly+1][lx+1] = 0.f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int dst_x = 2*get_global_id(0);
    int dst_y = 2*get_global_id(1);
    
    // (x,y)
    sum =       co3 * s_dstPatch[2 + ly - 2][lx];
    sum = sum + co2 * s_dstPatch[2 + ly - 1][lx];
    sum = sum + co1 * s_dstPatch[2 + ly    ][lx];
    sum = sum + co2 * s_dstPatch[2 + ly + 1][lx];
    sum = sum + co3 * s_dstPatch[2 + ly + 2][lx];

    if ((dst_x < dst_cols) && (dst_y < dst_rows))
        storepix(convertToT(sum), dstData + dst_y * dst_step + dst_x * PIXSIZE);

    // (x+1,y)
    sum =       co3 * s_dstPatch[2 + ly - 2][lx+1];
    sum = sum + co2 * s_dstPatch[2 + ly - 1][lx+1];
    sum = sum + co1 * s_dstPatch[2 + ly    ][lx+1];
    sum = sum + co2 * s_dstPatch[2 + ly + 1][lx+1];
    sum = sum + co3 * s_dstPatch[2 + ly + 2][lx+1];

    if ((dst_x+1 < dst_cols) && (dst_y < dst_rows))
        storepix(convertToT(sum), dstData + dst_y * dst_step + (dst_x+1) * PIXSIZE);

    // (x,y+1)
    sum =       co3 * s_dstPatch[2 + ly+1 - 2][lx];
    sum = sum + co2 * s_dstPatch[2 + ly+1 - 1][lx];
    sum = sum + co1 * s_dstPatch[2 + ly+1    ][lx];
    sum = sum + co2 * s_dstPatch[2 + ly+1 + 1][lx];
    sum = sum + co3 * s_dstPatch[2 + ly+1 + 2][lx];

    if ((dst_x < dst_cols) && (dst_y+1 < dst_rows))
        storepix(convertToT(sum), dstData + (dst_y+1) * dst_step + dst_x * PIXSIZE);

    // (x+1,y+1)
    sum =       co3 * s_dstPatch[2 + ly+1 - 2][lx+1];
    sum = sum + co2 * s_dstPatch[2 + ly+1 - 1][lx+1];
    sum = sum + co1 * s_dstPatch[2 + ly+1    ][lx+1];
    sum = sum + co2 * s_dstPatch[2 + ly+1 + 1][lx+1];
    sum = sum + co3 * s_dstPatch[2 + ly+1 + 2][lx+1];

    if ((dst_x+1 < dst_cols) && (dst_y+1 < dst_rows))
        storepix(convertToT(sum), dstData + (dst_y+1) * dst_step + (dst_x+1) * PIXSIZE);
}
