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

#ifndef cn
#define cn 1
#endif

#define sz (int)sizeof(float)
#define src_elem_at(_src, y, step, x) *(__global const float *)(_src + mad24(y, step, (x) * sz))
#define dst_elem_at(_dst, y, step, x) *(__global float *)(_dst + mad24(y, step, (x) * sz))

__kernel void buildMotionMaps(__global const uchar * forwardMotionPtr, int forwardMotion_step, int forwardMotion_offset,
                              __global const uchar * backwardMotionPtr, int backwardMotion_step, int backwardMotion_offset,
                              __global const uchar * forwardMapPtr, int forwardMap_step, int forwardMap_offset,
                              __global const uchar * backwardMapPtr, int backwardMap_step, int backwardMap_offset,
                              int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int forwardMotion_index = mad24(forwardMotion_step, y, (int)sizeof(float2) * x + forwardMotion_offset);
        int backwardMotion_index = mad24(backwardMotion_step, y, (int)sizeof(float2) * x + backwardMotion_offset);
        int forwardMap_index = mad24(forwardMap_step, y, (int)sizeof(float2) * x + forwardMap_offset);
        int backwardMap_index = mad24(backwardMap_step, y, (int)sizeof(float2) * x + backwardMap_offset);

        float2 forwardMotion = *(__global const float2 *)(forwardMotionPtr + forwardMotion_index);
        float2 backwardMotion = *(__global const float2 *)(backwardMotionPtr + backwardMotion_index);
        __global float2 * forwardMap = (__global float2 *)(forwardMapPtr + forwardMap_index);
        __global float2 * backwardMap = (__global float2 *)(backwardMapPtr + backwardMap_index);

        float2 basePoint = (float2)(x, y);

        forwardMap[0] = basePoint + backwardMotion;
        backwardMap[0] = basePoint + forwardMotion;
    }
}

__kernel void upscale(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                      __global uchar * dstptr, int dst_step, int dst_offset, int scale)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < src_cols && y < src_rows)
    {
        int src_index = mad24(y, src_step, sz * x * cn + src_offset);
        int dst_index = mad24(y * scale, dst_step, sz * x * scale * cn + dst_offset);

        __global const float * src = (__global const float *)(srcptr + src_index);
        __global float * dst = (__global float *)(dstptr + dst_index);

        #pragma unroll
        for (int c = 0; c < cn; ++c)
            dst[c] = src[c];
    }
}


inline float diffSign1(float a, float b)
{
    return a > b ? 1.0f : a < b ? -1.0f : 0.0f;
}

inline float3 diffSign3(float3 a, float3 b)
{
    float3 pos;
    pos.x = a.x > b.x ? 1.0f : a.x < b.x ? -1.0f : 0.0f;
    pos.y = a.y > b.y ? 1.0f : a.y < b.y ? -1.0f : 0.0f;
    pos.z = a.z > b.z ? 1.0f : a.z < b.z ? -1.0f : 0.0f;
    return pos;
}

__kernel void diffSign(__global const uchar * src1, int src1_step, int src1_offset,
                       __global const uchar * src2, int src2_step, int src2_offset,
                       __global uchar * dst, int dst_step, int dst_offset, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
        *(__global float *)(dst + mad24(y, dst_step, sz * x + dst_offset)) =
            diffSign1(*(__global const float *)(src1 + mad24(y, src1_step, sz * x + src1_offset)),
                      *(__global const float *)(src2 + mad24(y, src2_step, sz * x + src2_offset)));
}

__kernel void calcBtvRegularization(__global const uchar * src, int src_step, int src_offset,
                                    __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                    int ksize, __constant float * c_btvRegWeights)
{
    int x = get_global_id(0) + ksize;
    int y = get_global_id(1) + ksize;

    if (y < dst_rows - ksize && x < dst_cols - ksize)
    {
        src += src_offset;

#if cn == 1
        const float srcVal = src_elem_at(src, y, src_step, x);
        float dstVal = 0.0f;

        for (int m = 0, count = 0; m <= ksize; ++m)
            for (int l = ksize; l + m >= 0; --l, ++count)
            {
                dstVal += c_btvRegWeights[count] * (diffSign1(srcVal, src_elem_at(src, y + m, src_step, x + l))
                    - diffSign1(src_elem_at(src, y - m, src_step, x - l), srcVal));
            }

        dst_elem_at(dst, y, dst_step, x) = dstVal;
#elif cn == 3
        __global const float * src0ptr = (__global const float *)(src + mad24(y, src_step, 3 * sz * x + src_offset));
        float3 srcVal = (float3)(src0ptr[0], src0ptr[1], src0ptr[2]), dstVal = 0.f;

        for (int m = 0, count = 0; m <= ksize; ++m)
        {
            for (int l = ksize; l + m >= 0; --l, ++count)
            {
                __global const float * src1ptr = (__global const float *)(src + mad24(y + m, src_step, 3 * sz * (x + l) + src_offset));
                __global const float * src2ptr = (__global const float *)(src + mad24(y - m, src_step, 3 * sz * (x - l) + src_offset));

                float3 src1 = (float3)(src1ptr[0], src1ptr[1], src1ptr[2]);
                float3 src2 = (float3)(src2ptr[0], src2ptr[1], src2ptr[2]);

                dstVal += c_btvRegWeights[count] * (diffSign3(srcVal, src1) - diffSign3(src2, srcVal));
            }
        }

        __global float * dstptr = (__global float *)(dst + mad24(y, dst_step, 3 * sz * x + dst_offset + 0));
        dstptr[0] = dstVal.x;
        dstptr[1] = dstVal.y;
        dstptr[2] = dstVal.z;
#else
#error "Number of channels should be either 1 of 3"
#endif
    }
}
