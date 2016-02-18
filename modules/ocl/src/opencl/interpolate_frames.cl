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
// Copyright (C) 2010,2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

// Image read mode
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

// atomic add for 32bit floating point
void atomic_addf(volatile __global float *source, const float operand) {
    union {
        unsigned int intVal;
        float floatVal;
    } newVal;
    union {
        unsigned int intVal;
        float floatVal;
    } prevVal;
    do {
        prevVal.floatVal = *source;
        newVal.floatVal = prevVal.floatVal + operand;
    } while (atomic_cmpxchg((volatile __global unsigned int *)source, prevVal.intVal, newVal.intVal) != prevVal.intVal);
}

__kernel void memsetKernel(
    float val,
    __global float * image,
    int width,
    int height,
    int step, // in element
    int offset
    )
{
    if(get_global_id(0) >= width || get_global_id(1) >= height)
    {
        return;
    }
    image += offset;
    image[get_global_id(0) + get_global_id(1) * step] = val;
}

__kernel void normalizeKernel(
    __global float * buffer,
    int width,
    int height,
    int step,
    int f_offset,
    int d_offset
    )
{
    __global float * factors = buffer + f_offset;
    __global float * dst     = buffer + d_offset;

    int j = get_global_id(0);
    int i = get_global_id(1);

    if(j >= width || i >= height)
    {
        return;
    }
    float scale = factors[step * i + j];
    float invScale = (scale == 0.0f) ? 1.0f : (1.0f / scale);

    dst[step * i + j] *= invScale;
}

__kernel void forwardWarpKernel(
    __global const float * src,
    __global float * buffer,
    __global const float * u,
    __global const float * v,
    const int w,
    const int h,
    const int flow_stride,
    const int image_stride,
    const int factor_offset,
    const int dst_offset,
    const float time_scale
    )
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    if (i >= h || j >= w) return;

    volatile __global float * normalization_factor = (volatile __global float *) buffer + factor_offset;
    volatile __global float * dst = (volatile __global float *)buffer + dst_offset;

    int flow_row_offset  = i * flow_stride;
    int image_row_offset = i * image_stride;

    //bottom left corner of a target pixel
    float cx = u[flow_row_offset + j] * time_scale + (float)j + 1.0f;
    float cy = v[flow_row_offset + j] * time_scale + (float)i + 1.0f;
    // pixel containing bottom left corner
    float px;
    float py;
    float dx = modf(cx, &px);
    float dy = modf(cy, &py);
    // target pixel integer coords
    int tx;
    int ty;
    tx = (int) px;
    ty = (int) py;
    float value = src[image_row_offset + j];
    float weight;
    // fill pixel containing bottom right corner
    if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
    {
        weight = dx * dy;
        atomic_addf(dst + ty * image_stride + tx, value * weight);
        atomic_addf(normalization_factor + ty * image_stride + tx, weight);
    }

    // fill pixel containing bottom left corner
    tx -= 1;
    if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
    {
        weight = (1.0f - dx) * dy;
        atomic_addf(dst + ty * image_stride + tx, value * weight);
        atomic_addf(normalization_factor + ty * image_stride + tx, weight);
    }

    // fill pixel containing upper left corner
    ty -= 1;
    if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
    {
        weight = (1.0f - dx) * (1.0f - dy);
        atomic_addf(dst + ty * image_stride + tx, value * weight);
        atomic_addf(normalization_factor + ty * image_stride + tx, weight);
    }

    // fill pixel containing upper right corner
    tx += 1;
    if (!((tx >= w) || (tx < 0) || (ty >= h) || (ty < 0)))
    {
        weight = dx * (1.0f - dy);
        atomic_addf(dst + ty * image_stride + tx, value * weight);
        atomic_addf(normalization_factor + ty * image_stride + tx, weight);
    }
}

// define buffer offsets
enum
{
    O0_OS = 0,
    O1_OS,
    U_OS,
    V_OS,
    UR_OS,
    VR_OS
};

__kernel void blendFramesKernel(
    image2d_t tex_src0,
    image2d_t tex_src1,
    __global float * buffer,
    __global float * out,
    int w,
    int h,
    int step,
    float theta
    )
{
    __global float * u  = buffer + h * step * U_OS;
    __global float * v  = buffer + h * step * V_OS;
    __global float * ur = buffer + h * step * UR_OS;
    __global float * vr = buffer + h * step * VR_OS;
    __global float * o0 = buffer + h * step * O0_OS;
    __global float * o1 = buffer + h * step * O1_OS;

    int ix = get_global_id(0);
    int iy = get_global_id(1);

    if(ix >= w || iy >= h) return;

    int pos = ix + step * iy;

    float _u  = u[pos];
    float _v  = v[pos];

    float _ur = ur[pos];
    float _vr = vr[pos];

    float x = (float)ix + 0.5f;
    float y = (float)iy + 0.5f;
    bool b0 = o0[pos] > 1e-4f;
    bool b1 = o1[pos] > 1e-4f;

    float2 coord0 = (float2)(x - _u * theta, y - _v * theta);
    float2 coord1 = (float2)(x + _u * (1.0f - theta), y + _v * (1.0f - theta));

    if (b0 && b1)
    {
        // pixel is visible on both frames
        out[pos] = read_imagef(tex_src0, sampler, coord0).x * (1.0f - theta) +
            read_imagef(tex_src1, sampler, coord1).x * theta;
    }
    else if (b0)
    {
        // visible on the first frame only
        out[pos] = read_imagef(tex_src0, sampler, coord0).x;
    }
    else
    {
        // visible on the second frame only
        out[pos] = read_imagef(tex_src1, sampler, coord1).x;
    }
}
