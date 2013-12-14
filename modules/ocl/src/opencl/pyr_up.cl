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

__kernel void pyrUp(__global Type* src, __global Type* dst,
                          int srcRows, int dstRows, int srcCols, int dstCols,
                          int srcOffset, int dstOffset, int srcStep, int dstStep)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int lsizex = get_local_size(0);
    const int lsizey = get_local_size(1);

    const int tidx = get_local_id(0);
    const int tidy = get_local_id(1);

    __local floatType s_srcPatch[10][10];
    __local floatType s_dstPatch[20][16];

    if( tidx < 10 && tidy < 10 )
    {
        int srcx = mad24((int)get_group_id(0), lsizex>>1, tidx) - 1;
        int srcy = mad24((int)get_group_id(1), lsizey>>1, tidy) - 1;

        srcx = abs(srcx);
        srcx = min(srcCols - 1,srcx);

        srcy = abs(srcy);
        srcy = min(srcRows -1 ,srcy);

        s_srcPatch[tidy][tidx] = convertToFloat(src[srcx + srcy * srcStep]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    floatType sum = (floatType)0;
    const floatType evenFlag = (floatType)((tidx & 1) == 0);
    const floatType oddFlag = (floatType)((tidx & 1) != 0);
    const bool  eveny = ((tidy & 1) == 0);

    const floatType co1 = (floatType)0.375f;
    const floatType co2 = (floatType)0.25f;
    const floatType co3 = (floatType)0.0625f;

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

    if ((x < dstCols) && (y < dstRows))
        dst[x + y * dstStep] = convertToType(4.0f * sum);
}
