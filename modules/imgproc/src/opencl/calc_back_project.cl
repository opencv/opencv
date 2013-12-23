//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Xu Pang, pangxu010@163.com
//    Wenju He, wenju@multicorewareinc.com
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

#if histdims == 1

#define OUT_OF_RANGE -1

__kernel void calcLUT(__global const uchar * histptr, int hist_step, int hist_offset, int hist_bins,
                      __global int * lut, float scale, __constant float * ranges)
{
    int x = get_global_id(0);
    float value = convert_float(x);

    if (value > ranges[1] || value < ranges[0])
        lut[x] = OUT_OF_RANGE;
    else
    {
        float lb = ranges[0], ub = ranges[1], gap = (ub - lb) / hist_bins;
        value -= lb;
        int bin = convert_int_sat_rtn(value / gap);

        if (bin >= hist_bins)
            lut[x] = OUT_OF_RANGE;
        else
        {
            int hist_index = mad24(hist_step, bin, hist_offset);
            __global const float * hist = (__global const float *)(histptr + hist_index);

            lut[x] = (int)convert_uchar_sat_rte(hist[0] * scale);
        }
    }
}

__kernel void LUT(__global const uchar * src, int src_step, int src_offset,
                  __global const int * lut,
                  __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int src_index = mad24(y, src_step, src_offset + x * scn);
        int dst_index = mad24(y, dst_step, dst_offset + x);

        int value = lut[src[src_index]];
        dst[dst_index] = value == OUT_OF_RANGE ? 0 : convert_uchar(value);
    }
}

#elif histdims == 2

#define OUT_OF_RANGES(i) ( (value##i > ranges[(i<<1)+1]) || (value##i < ranges[i<<1]) )
#define CALCULATE_BIN(i) \
    float lb##i = ranges[i<<1], ub##i = ranges[(i<<1)+1], gap##i = (ub##i - lb##i) / hist_bins##i; \
    value##i -= ranges[i<<1]; \
    int bin##i = convert_int_sat_rtn(value##i / gap##i)

__kernel void calcBackProject(__global const uchar * src0, int src0_step, int src0_offset,
                              __global const uchar * src1, int src1_step, int src1_offset,
                              __global const uchar * histptr, int hist_step, int hist_offset, int hist_bins0, int hist_bins1,
                              __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              float scale, __constant float * ranges)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int src0_index = mad24(src0_step, y, src0_offset + x * scn0);
        int src1_index = mad24(src1_step, y, src1_offset + x * scn1);
        int dst_index = mad24(dst_step, y, dst_offset + x);

        float value0 = convert_float(src0[src0_index]), value1 = convert_float(src1[src1_index]);
        if (OUT_OF_RANGES(0) || OUT_OF_RANGES(1))
            dst[dst_index] = 0;
        else
        {
            CALCULATE_BIN(0);
            CALCULATE_BIN(1);

            if (bin0 >= hist_bins0 || bin1 >= hist_bins1)
                dst[dst_index] = 0;
            else
            {
                int hist_index = mad24(hist_step, bin0, hist_offset + bin1 * (int)sizeof(float));
                __global const float * hist = (__global const float *)(histptr + hist_index);

                dst[dst_index] = convert_uchar_sat_rte(scale * hist[0]);
            }
        }
    }
}

#else
#error "(nimages <= 2) should be true"
#endif
