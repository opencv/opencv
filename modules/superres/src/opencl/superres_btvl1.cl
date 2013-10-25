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
//    Jin Ma jin@multicorewareinc.com
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

__kernel void buildMotionMapsKernel(__global float* forwardMotionX,
                                    __global float* forwardMotionY,
                                    __global float* backwardMotionX,
                                    __global float* backwardMotionY,
                                    __global float* forwardMapX,
                                    __global float* forwardMapY,
                                    __global float* backwardMapX,
                                    __global float* backwardMapY,
                                    int forwardMotionX_row,
                                    int forwardMotionX_col,
                                    int forwardMotionX_step,
                                    int forwardMotionY_step,
                                    int backwardMotionX_step,
                                    int backwardMotionY_step,
                                    int forwardMapX_step,
                                    int forwardMapY_step,
                                    int backwardMapX_step,
                                    int backwardMapY_step
                                   )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < forwardMotionX_col && y < forwardMotionX_row)
    {
        float fx = forwardMotionX[y * forwardMotionX_step + x];
        float fy = forwardMotionY[y * forwardMotionY_step + x];

        float bx = backwardMotionX[y * backwardMotionX_step + x];
        float by = backwardMotionY[y * backwardMotionY_step + x];

        forwardMapX[y * forwardMapX_step + x] = x + bx;
        forwardMapY[y * forwardMapY_step + x] = y + by;

        backwardMapX[y * backwardMapX_step + x] = x + fx;
        backwardMapY[y * backwardMapY_step + x] = y + fy;
    }
}

__kernel void upscaleKernel(__global float* src,
                            __global float* dst,
                            int src_step,
                            int dst_step,
                            int src_row,
                            int src_col,
                            int scale,
                            int channels
                           )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < src_col && y < src_row)
    {
        if(channels == 1)
        {
            dst[y * scale * dst_step + x * scale] = src[y * src_step + x];
        }
        else
        {
            vstore4(vload4(0, src + y * channels * src_step + 4 * x), 0, dst + y * channels * scale * dst_step + 4 * x * scale);
        }
    }
}


float diffSign(float a, float b)
{
    return a > b ? 1.0f : a < b ? -1.0f : 0.0f;
}

float4 diffSign4(float4 a, float4 b)
{
    float4 pos;
    pos.x = a.x > b.x ? 1.0f : a.x < b.x ? -1.0f : 0.0f;
    pos.y = a.y > b.y ? 1.0f : a.y < b.y ? -1.0f : 0.0f;
    pos.z = a.z > b.z ? 1.0f : a.z < b.z ? -1.0f : 0.0f;
    pos.w = 0.0f;
    return pos;
}

__kernel void diffSignKernel(__global float* src1,
                             __global float* src2,
                             __global float* dst,
                             int src1_row,
                             int src1_col,
                             int dst_step,
                             int src1_step,
                             int src2_step)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < src1_col && y < src1_row)
    {
        dst[y * dst_step + x] = diffSign(src1[y * src1_step + x], src2[y * src2_step + x]);
    }
}

__kernel void calcBtvRegularizationKernel(__global float* src,
        __global float* dst,
        int src_step,
        int dst_step,
        int src_row,
        int src_col,
        int ksize,
        int channels,
        __constant float* c_btvRegWeights
                                         )
{
    int x = get_global_id(0) + ksize;
    int y = get_global_id(1) + ksize;

    if ((y < src_row - ksize) && (x < src_col - ksize))
    {
        if(channels == 1)
        {
            const float srcVal = src[y * src_step + x];
            float dstVal = 0.0f;

            for (int m = 0, count = 0; m <= ksize; ++m)
            {
                for (int l = ksize; l + m >= 0; --l, ++count)
                {
                    dstVal = dstVal + c_btvRegWeights[count] * (diffSign(srcVal, src[(y + m) * src_step + (x + l)]) - diffSign(src[(y - m) * src_step + (x - l)], srcVal));
                }
            }
            dst[y * dst_step + x] = dstVal;
        }
        else
        {
            float4 srcVal = vload4(0, src + y * src_step + 4 * x);
            float4 dstVal = 0.f;

            for (int m = 0, count = 0; m <= ksize; ++m)
            {
                for (int l = ksize; l + m >= 0; --l, ++count)
                {
                    float4 src1;
                    src1.x = src[(y + m) * src_step + 4 * (x + l) + 0];
                    src1.y = src[(y + m) * src_step + 4 * (x + l) + 1];
                    src1.z = src[(y + m) * src_step + 4 * (x + l) + 2];
                    src1.w = src[(y + m) * src_step + 4 * (x + l) + 3];

                    float4 src2;
                    src2.x = src[(y - m) * src_step + 4 * (x - l) + 0];
                    src2.y = src[(y - m) * src_step + 4 * (x - l) + 1];
                    src2.z = src[(y - m) * src_step + 4 * (x - l) + 2];
                    src2.w = src[(y - m) * src_step + 4 * (x - l) + 3];

                    dstVal = dstVal + c_btvRegWeights[count] * (diffSign4(srcVal, src1) - diffSign4(src2, srcVal));
                }
            }
            vstore4(dstVal, 0, dst + y * dst_step + 4 * x);
        }
    }
}
