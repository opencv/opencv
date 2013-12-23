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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
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

/**************************************PUBLICFUNC*************************************/

#ifndef hscale
#define hscale 0
#endif

#ifndef hrange
#define hrange 0
#endif

#ifdef DEPTH_0
#define DATA_TYPE uchar
#define VECTOR2 uchar2
#define VECTOR4 uchar4
#define VECTOR8 uchar8
#define VECTOR16 uchar16
#define COEFF_TYPE int
#define MAX_NUM  255
#define HALF_MAX 128
#define SAT_CAST(num) convert_uchar_sat_rte(num)
#define SAT_CAST2(num) convert_uchar2_sat(num)
#define SAT_CAST4(num) convert_uchar4_sat(num)
#endif

#ifdef DEPTH_2
#define DATA_TYPE ushort
#define VECTOR2 ushort2
#define VECTOR4 ushort4
#define VECTOR8 ushort8
#define VECTOR16 ushort16
#define COEFF_TYPE int
#define MAX_NUM  65535
#define HALF_MAX 32768
#define SAT_CAST(num) convert_ushort_sat_rte(num)
#define SAT_CAST2(num) convert_ushort2_sat(num)
#define SAT_CAST4(num) convert_ushort4_sat(num)
#endif

#ifdef DEPTH_5
#define DATA_TYPE float
#define VECTOR2 float2
#define VECTOR4 float4
#define VECTOR8 float8
#define VECTOR16 float16
#define COEFF_TYPE float
#define MAX_NUM  1.0f
#define HALF_MAX 0.5f
#define SAT_CAST(num) (num)
#endif

#ifndef bidx
    #define bidx 0
#endif

#ifndef pixels_per_work_item
    #define pixels_per_work_item 1
#endif

#define CV_DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

enum
{
    yuv_shift  = 14,
    xyz_shift  = 12,
    hsv_shift  = 12,
    R2Y        = 4899,
    G2Y        = 9617,
    B2Y        = 1868,
    BLOCK_SIZE = 256
};

///////////////////////////////////// RGB <-> GRAY //////////////////////////////////////

__constant float c_RGB2GrayCoeffs_f[3]  = { 0.114f, 0.587f, 0.299f };
__constant int   c_RGB2GrayCoeffs_i[3]  = { B2Y, G2Y, R2Y };

__kernel void RGB2Gray(int cols, int rows, int src_step, int dst_step,
                       __global const DATA_TYPE* src, __global DATA_TYPE* dst,
                       int src_offset, int dst_offset)
{
    int x = get_global_id(0) * pixels_per_work_item;
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int src_idx = mad24(y, src_step, src_offset + (x << 2));
        int dst_idx = mad24(y, dst_step, dst_offset + x);

#ifndef INTEL_DEVICE

#ifdef DEPTH_5
        dst[dst_idx] = src[src_idx + bidx] * 0.114f + src[src_idx + 1] * 0.587f + src[src_idx + (bidx^2)] * 0.299f;
#else
        dst[dst_idx] = (DATA_TYPE)CV_DESCALE((src[src_idx + bidx] * B2Y + src[src_idx + 1] * G2Y + src[src_idx + (bidx^2)] * R2Y), yuv_shift);
#endif

#else   //INTEL_DEVICE
        global DATA_TYPE *src_ptr = (global DATA_TYPE *)(src + src_idx);
        global DATA_TYPE *dst_ptr = (global DATA_TYPE *)(dst + dst_idx);

#ifdef DEPTH_5
        __constant float * coeffs = c_RGB2GrayCoeffs_f;
#else
        __constant int * coeffs = c_RGB2GrayCoeffs_i;
#endif

#if (1 == pixels_per_work_item)
        {
#ifdef DEPTH_5
            *dst_ptr = src_ptr[bidx] * coeffs[0] + src_ptr[1] * coeffs[1] + src_ptr[(bidx^2)] *coeffs[2];
#else
            *dst_ptr = (DATA_TYPE)CV_DESCALE((src_ptr[bidx] * coeffs[0] + src_ptr[1] * coeffs[1] + src_ptr[(bidx^2)] * coeffs[2]), yuv_shift);
#endif
        }
#elif (2 == pixels_per_work_item)
        {
            const VECTOR8 r0 = vload8(0, src_ptr);

#ifdef DEPTH_5
            const float2 c0 = r0.s04;
            const float2 c1 = r0.s15;
            const float2 c2 = r0.s26;

            const float2 Y = c0 * coeffs[bidx] + c1 * coeffs[1] + c2 * coeffs[bidx^2];
#else
            const int2 c0 = convert_int2(r0.s04);
            const int2 c1 = convert_int2(r0.s15);
            const int2 c2 = convert_int2(r0.s26);

            const int2 yi = CV_DESCALE(c0 * coeffs[bidx] + c1 * coeffs[1] + c2 * coeffs[bidx^2], yuv_shift);
            const VECTOR2 Y = SAT_CAST2(yi);
#endif

            vstore2(Y, 0, dst_ptr);
        }
#elif (4 == pixels_per_work_item)
        {
#ifndef DEPTH_5
            const VECTOR16 r0 = vload16(0, src_ptr);

            const int4 c0 = convert_int4(r0.s048c);
            const int4 c1 = convert_int4(r0.s159d);
            const int4 c2 = convert_int4(r0.s26ae);
            const int4 Y = CV_DESCALE(c0 * coeffs[bidx] + c1 * coeffs[1] + c2 * coeffs[bidx^2], yuv_shift);

            vstore4(SAT_CAST4(Y), 0, dst_ptr);
#endif
        }
#endif //pixels_per_work_item
#endif //INTEL_DEVICE
    }
}

__kernel void Gray2RGB(int cols, int rows, int src_step, int dst_step,
                       __global const DATA_TYPE* src, __global DATA_TYPE* dst,
                       int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + (x << 2));

        DATA_TYPE val = src[src_idx];
        dst[dst_idx] = val;
        dst[dst_idx + 1] = val;
        dst[dst_idx + 2] = val;
#if dcn == 4
        dst[dst_idx + 3] = MAX_NUM;
#endif
    }
}

///////////////////////////////////// RGB <-> YUV //////////////////////////////////////

__constant float c_RGB2YUVCoeffs_f[5]  = { 0.114f, 0.587f, 0.299f, 0.492f, 0.877f };
__constant int   c_RGB2YUVCoeffs_i[5]  = { B2Y, G2Y, R2Y, 8061, 14369 };

__kernel void RGB2YUV(int cols, int rows, int src_step, int dst_step,
                      __global const DATA_TYPE* src, __global DATA_TYPE* dst,
                      int src_offset, int dst_offset)
{
    int x = get_global_id(0) * pixels_per_work_item;
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        global DATA_TYPE *src_ptr = (global DATA_TYPE *)(src + src_idx);
        global DATA_TYPE *dst_ptr = (global DATA_TYPE *)(dst + dst_idx);

#ifdef DEPTH_5
        __constant float * coeffs = c_RGB2YUVCoeffs_f;
#else
        __constant int * coeffs = c_RGB2YUVCoeffs_i;
        const int delta = HALF_MAX * (1 << yuv_shift);
#endif

#if (1 == pixels_per_work_item)
        {
            const DATA_TYPE rgb[] = {src_ptr[0], src_ptr[1], src_ptr[2]};

#ifdef DEPTH_5
            float Y = rgb[0] * coeffs[bidx^2] + rgb[1] * coeffs[1] + rgb[2] * coeffs[bidx];
            float U = (rgb[bidx^2] - Y) * coeffs[3] + HALF_MAX;
            float V = (rgb[bidx] - Y) * coeffs[4] + HALF_MAX;
#else
            int Y = CV_DESCALE(rgb[0] * coeffs[bidx^2] + rgb[1] * coeffs[1] + rgb[2] * coeffs[bidx], yuv_shift);
            int U = CV_DESCALE((rgb[bidx^2] - Y) * coeffs[3] + delta, yuv_shift);
            int V = CV_DESCALE((rgb[bidx] - Y) * coeffs[4] + delta, yuv_shift);
#endif

            dst_ptr[0] = SAT_CAST( Y );
            dst_ptr[1] = SAT_CAST( U );
            dst_ptr[2] = SAT_CAST( V );
        }
#elif (2 == pixels_per_work_item)
        {
            const VECTOR8 r0 = vload8(0, src_ptr);

#ifdef DEPTH_5
            const float2 c0 = r0.s04;
            const float2 c1 = r0.s15;
            const float2 c2 = r0.s26;

            const float2 Y = (bidx == 0) ? (c0 * coeffs[2] + c1 * coeffs[1] + c2 * coeffs[0]) : (c0 * coeffs[0] + c1 * coeffs[1] + c2 * coeffs[2]);
            const float2 U = (bidx == 0) ? ((c2 - Y) * coeffs[3] + HALF_MAX) : ((c0 - Y) * coeffs[3] + HALF_MAX);
            const float2 V = (bidx == 0) ? ((c0 - Y) * coeffs[4] + HALF_MAX) : ((c2 - Y) * coeffs[4] + HALF_MAX);
#else
            const int2 c0 = convert_int2(r0.s04);
            const int2 c1 = convert_int2(r0.s15);
            const int2 c2 = convert_int2(r0.s26);

            const int2 yi = (bidx == 0) ? CV_DESCALE(c0 * coeffs[2] + c1 * coeffs[1] + c2 * coeffs[0], yuv_shift) : CV_DESCALE(c0 * coeffs[0] + c1 * coeffs[1] + c2 * coeffs[2], yuv_shift);
            const int2 ui = (bidx == 0) ? CV_DESCALE((c2 - yi) * coeffs[3] + delta, yuv_shift) : CV_DESCALE((c0 - yi) * coeffs[3] + delta, yuv_shift);
            const int2 vi = (bidx == 0) ? CV_DESCALE((c0 - yi) * coeffs[4] + delta, yuv_shift) : CV_DESCALE((c2 - yi) * coeffs[4] + delta, yuv_shift);

            const VECTOR2 Y = SAT_CAST2(yi);
            const VECTOR2 U = SAT_CAST2(ui);
            const VECTOR2 V = SAT_CAST2(vi);
#endif

            vstore8((VECTOR8)(Y.s0, U.s0, V.s0, 0, Y.s1, U.s1, V.s1, 0), 0, dst_ptr);
        }
#elif (4 == pixels_per_work_item)
        {
#ifndef DEPTH_5
            const VECTOR16 r0 = vload16(0, src_ptr);

            const int4 c0 = convert_int4(r0.s048c);
            const int4 c1 = convert_int4(r0.s159d);
            const int4 c2 = convert_int4(r0.s26ae);

            const int4 yi = (bidx == 0) ? CV_DESCALE(c0 * coeffs[2] + c1 * coeffs[1] + c2 * coeffs[0], yuv_shift) : CV_DESCALE(c0 * coeffs[0] + c1 * coeffs[1] + c2 * coeffs[2], yuv_shift);
            const int4 ui = (bidx == 0) ? CV_DESCALE((c2 - yi) * coeffs[3] + delta, yuv_shift) : CV_DESCALE((c0 - yi) * coeffs[3] + delta, yuv_shift);
            const int4 vi = (bidx == 0) ? CV_DESCALE((c0 - yi) * coeffs[4] + delta, yuv_shift) : CV_DESCALE((c2 - yi) * coeffs[4] + delta, yuv_shift);

            const VECTOR4 Y = SAT_CAST4(yi);
            const VECTOR4 U = SAT_CAST4(ui);
            const VECTOR4 V = SAT_CAST4(vi);

            vstore16((VECTOR16)(Y.s0, U.s0, V.s0, 0, Y.s1, U.s1, V.s1, 0, Y.s2, U.s2, V.s2, 0, Y.s3, U.s3, V.s3, 0), 0, dst_ptr);
#endif
        }
#endif //pixels_per_work_item
    }
}

__constant float c_YUV2RGBCoeffs_f[5] = { 2.032f, -0.395f, -0.581f, 1.140f };
__constant int   c_YUV2RGBCoeffs_i[5] = { 33292, -6472, -9519, 18678 };

__kernel void YUV2RGB(int cols, int rows, int src_step, int dst_step,
                      __global const DATA_TYPE* src, __global DATA_TYPE* dst,
                      int src_offset, int dst_offset)
{
    int x = get_global_id(0) * pixels_per_work_item;
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        global DATA_TYPE *src_ptr = (global DATA_TYPE *)(src + src_idx);
        global DATA_TYPE *dst_ptr = (global DATA_TYPE *)(dst + dst_idx);

#ifdef DEPTH_5
        __constant float * coeffs = c_YUV2RGBCoeffs_f;
#else
        __constant int * coeffs = c_YUV2RGBCoeffs_i;
#endif

#if (1 == pixels_per_work_item)
        {
            const DATA_TYPE yuv[] = {src_ptr[0], src_ptr[1], src_ptr[2]};

#ifdef DEPTH_5
            float B = yuv[0] + (yuv[2] - HALF_MAX) * coeffs[3];
            float G = yuv[0] + (yuv[2] - HALF_MAX) * coeffs[2] + (yuv[1] - HALF_MAX) * coeffs[1];
            float R = yuv[0] + (yuv[1] - HALF_MAX) * coeffs[0];
#else
            int B = yuv[0] + CV_DESCALE((yuv[2] - HALF_MAX) * coeffs[3], yuv_shift);
            int G = yuv[0] + CV_DESCALE((yuv[2] - HALF_MAX) * coeffs[2] + (yuv[1] - HALF_MAX) * coeffs[1], yuv_shift);
            int R = yuv[0] + CV_DESCALE((yuv[1] - HALF_MAX) * coeffs[0], yuv_shift);
#endif

            dst_ptr[bidx]     = SAT_CAST( B );
            dst_ptr[1]        = SAT_CAST( G );
            dst_ptr[(bidx^2)] = SAT_CAST( R );
#if dcn == 4
            dst_ptr[3]         = MAX_NUM;
#endif
        }
#elif (2 == pixels_per_work_item)
        {
            const VECTOR8 r0 = vload8(0, src_ptr);

#ifdef DEPTH_5
            const float2 Y = r0.s04;
            const float2 U = r0.s15;
            const float2 V = r0.s26;

            const float2 c0 = (bidx == 0) ? (Y + (V - HALF_MAX) * coeffs[3]) : (Y + (U - HALF_MAX) * coeffs[0]);
            const float2 c1 = Y + (V - HALF_MAX) * coeffs[2] + (U - HALF_MAX) * coeffs[1];
            const float2 c2 = (bidx == 0) ? (Y + (U - HALF_MAX) * coeffs[0]) : (Y + (V - HALF_MAX) * coeffs[3]);
#else
            const int2 Y = convert_int2(r0.s04);
            const int2 U = convert_int2(r0.s15);
            const int2 V = convert_int2(r0.s26);

            const int2 c0i = (bidx == 0) ? (Y + CV_DESCALE((V - HALF_MAX) * coeffs[3], yuv_shift)) : (Y + CV_DESCALE((U - HALF_MAX) * coeffs[0], yuv_shift));
            const int2 c1i = Y + CV_DESCALE((V - HALF_MAX) * coeffs[2] + (U - HALF_MAX) * coeffs[1], yuv_shift);
            const int2 c2i = (bidx == 0) ? (Y + CV_DESCALE((U - HALF_MAX) * coeffs[0], yuv_shift)) : (Y + CV_DESCALE((V - HALF_MAX) * coeffs[3], yuv_shift));

            const VECTOR2 c0 = SAT_CAST2(c0i);
            const VECTOR2 c1 = SAT_CAST2(c1i);
            const VECTOR2 c2 = SAT_CAST2(c2i);
#endif

#if dcn == 4
            vstore8((VECTOR8)(c0.s0, c1.s0, c2.s0, MAX_NUM, c0.s1, c1.s1, c2.s1, MAX_NUM), 0, dst_ptr);
#else
            vstore8((VECTOR8)(c0.s0, c1.s0, c2.s0, 0, c0.s1, c1.s1, c2.s1, 0), 0, dst_ptr);
#endif
        }
#elif (4 == pixels_per_work_item)
        {
#ifndef DEPTH_5
            const VECTOR16 r0 = vload16(0, src_ptr);

            const int4 Y = convert_int4(r0.s048c);
            const int4 U = convert_int4(r0.s159d);
            const int4 V = convert_int4(r0.s26ae);

            const int4 c0i = (bidx == 0) ? (Y + CV_DESCALE((V - HALF_MAX) * coeffs[3], yuv_shift)) : (Y + CV_DESCALE((U - HALF_MAX) * coeffs[0], yuv_shift));
            const int4 c1i = Y + CV_DESCALE((V - HALF_MAX) * coeffs[2] + (U - HALF_MAX) * coeffs[1], yuv_shift);
            const int4 c2i = (bidx == 0) ? (Y + CV_DESCALE((U - HALF_MAX) * coeffs[0], yuv_shift)) : (Y + CV_DESCALE((V - HALF_MAX) * coeffs[3], yuv_shift));

            const VECTOR4 c0 = SAT_CAST4(c0i);
            const VECTOR4 c1 = SAT_CAST4(c1i);
            const VECTOR4 c2 = SAT_CAST4(c2i);

#if dcn == 4
            vstore16((VECTOR16)(c0.s0, c1.s0, c2.s0, MAX_NUM, c0.s1, c1.s1, c2.s1, MAX_NUM, c0.s2, c1.s2, c2.s2, MAX_NUM, c0.s3, c1.s3, c2.s3, MAX_NUM), 0, dst_ptr);
#else
            vstore16((VECTOR16)(c0.s0, c1.s0, c2.s0, 0, c0.s1, c1.s1, c2.s1, 0, c0.s2, c1.s2, c2.s2, 0, c0.s3, c1.s3, c2.s3, 0), 0, dst_ptr);
#endif
#endif
        }
#endif  //pixels_per_work_item
    }
}

__constant int ITUR_BT_601_CY = 1220542;
__constant int ITUR_BT_601_CUB = 2116026;
__constant int ITUR_BT_601_CUG = 409993;
__constant int ITUR_BT_601_CVG = 852492;
__constant int ITUR_BT_601_CVR = 1673527;
__constant int ITUR_BT_601_SHIFT = 20;

__kernel void YUV2RGBA_NV12(int cols, int rows, int src_step, int dst_step,
                            __global const uchar* src, __global uchar* dst,
                            int src_offset, int dst_offset)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (y < rows / 2 && x < cols / 2 )
    {
        __global const uchar* ysrc = src + mad24(y << 1, src_step, (x << 1) + src_offset);
        __global const uchar* usrc = src + mad24(rows + y, src_step, (x << 1) + src_offset);
        __global uchar*       dst1 = dst + mad24(y << 1, dst_step, (x << 3) + dst_offset);
        __global uchar*       dst2 = dst + mad24((y << 1) + 1, dst_step, (x << 3) + dst_offset);

        int Y1 = ysrc[0];
        int Y2 = ysrc[1];
        int Y3 = ysrc[src_step];
        int Y4 = ysrc[src_step + 1];

        int U  = usrc[0] - 128;
        int V  = usrc[1] - 128;

        int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * V;
        int guv = (1 << (ITUR_BT_601_SHIFT - 1)) - ITUR_BT_601_CVG * V - ITUR_BT_601_CUG * U;
        int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * U;

        Y1 = max(0, Y1 - 16) * ITUR_BT_601_CY;
        dst1[2 - bidx]     = convert_uchar_sat((Y1 + ruv) >> ITUR_BT_601_SHIFT);
        dst1[1]        = convert_uchar_sat((Y1 + guv) >> ITUR_BT_601_SHIFT);
        dst1[bidx] = convert_uchar_sat((Y1 + buv) >> ITUR_BT_601_SHIFT);
        dst1[3]        = 255;

        Y2 = max(0, Y2 - 16) * ITUR_BT_601_CY;
        dst1[6 - bidx] = convert_uchar_sat((Y2 + ruv) >> ITUR_BT_601_SHIFT);
        dst1[5]        = convert_uchar_sat((Y2 + guv) >> ITUR_BT_601_SHIFT);
        dst1[4 + bidx] = convert_uchar_sat((Y2 + buv) >> ITUR_BT_601_SHIFT);
        dst1[7]        = 255;

        Y3 = max(0, Y3 - 16) * ITUR_BT_601_CY;
        dst2[2 - bidx]     = convert_uchar_sat((Y3 + ruv) >> ITUR_BT_601_SHIFT);
        dst2[1]        = convert_uchar_sat((Y3 + guv) >> ITUR_BT_601_SHIFT);
        dst2[bidx] = convert_uchar_sat((Y3 + buv) >> ITUR_BT_601_SHIFT);
        dst2[3]        = 255;

        Y4 = max(0, Y4 - 16) * ITUR_BT_601_CY;
        dst2[6 - bidx] = convert_uchar_sat((Y4 + ruv) >> ITUR_BT_601_SHIFT);
        dst2[5]        = convert_uchar_sat((Y4 + guv) >> ITUR_BT_601_SHIFT);
        dst2[4 + bidx] = convert_uchar_sat((Y4 + buv) >> ITUR_BT_601_SHIFT);
        dst2[7]        = 255;
    }
}

///////////////////////////////////// RGB <-> YCrCb //////////////////////////////////////

__constant float c_RGB2YCrCbCoeffs_f[5] = {0.299f, 0.587f, 0.114f, 0.713f, 0.564f};
__constant int   c_RGB2YCrCbCoeffs_i[5] = {R2Y, G2Y, B2Y, 11682, 9241};

__kernel void RGB2YCrCb(int cols, int rows, int src_step, int dst_step,
                      __global const DATA_TYPE* src, __global DATA_TYPE* dst,
                      int src_offset, int dst_offset)
{
    int x = get_global_id(0) * pixels_per_work_item;
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        global DATA_TYPE *src_ptr = (global DATA_TYPE *)(src + src_idx);
        global DATA_TYPE *dst_ptr = (global DATA_TYPE *)(dst + dst_idx);

#ifdef DEPTH_5
        __constant float * coeffs = c_RGB2YCrCbCoeffs_f;
#else
        __constant int * coeffs = c_RGB2YCrCbCoeffs_i;
        const int delta = HALF_MAX * (1 << yuv_shift);
#endif

#if (1 == pixels_per_work_item)
        {
            const DATA_TYPE rgb[] = {src_ptr[0], src_ptr[1], src_ptr[2]};

#ifdef DEPTH_5
            float Y  = rgb[0] * coeffs[bidx^2] + rgb[1] * coeffs[1] + rgb[2] * coeffs[bidx];
            float Cr = (rgb[bidx^2] - Y) * coeffs[3] + HALF_MAX;
            float Cb = (rgb[bidx] - Y) * coeffs[4] + HALF_MAX;
#else
            int Y =  CV_DESCALE(rgb[0] * coeffs[bidx^2] + rgb[1] * coeffs[1] + rgb[2] * coeffs[bidx], yuv_shift);
            int Cr = CV_DESCALE((rgb[bidx^2] - Y) * coeffs[3] + delta, yuv_shift);
            int Cb = CV_DESCALE((rgb[bidx] - Y) * coeffs[4] + delta, yuv_shift);
#endif

            dst_ptr[0] = SAT_CAST( Y );
            dst_ptr[1] = SAT_CAST( Cr );
            dst_ptr[2] = SAT_CAST( Cb );
        }
#elif (2 == pixels_per_work_item)
        {
            const VECTOR8 r0 = vload8(0, src_ptr);

#ifdef DEPTH_5
            const float2 c0 = r0.s04;
            const float2 c1 = r0.s15;
            const float2 c2 = r0.s26;

            const float2 Y  = (bidx == 0) ? (c0 * coeffs[2] + c1 * coeffs[1] + c2 * coeffs[0]) : (c0 * coeffs[0] + c1 * coeffs[1] + c2 * coeffs[2]);
            const float2 Cr = (bidx == 0) ? ((c2 - Y) * coeffs[3] + HALF_MAX) : ((c0 - Y) * coeffs[3] + HALF_MAX);
            const float2 Cb = (bidx == 0) ? ((c0 - Y) * coeffs[4] + HALF_MAX) : ((c2 - Y) * coeffs[4] + HALF_MAX);
#else
            const int2 c0 = convert_int2(r0.s04);
            const int2 c1 = convert_int2(r0.s15);
            const int2 c2 = convert_int2(r0.s26);

            const int2 yi = (bidx == 0) ? CV_DESCALE(c0 * coeffs[2] + c1 * coeffs[1] + c2 * coeffs[0], yuv_shift) : CV_DESCALE(c0 * coeffs[0] + c1 * coeffs[1] + c2 * coeffs[2], yuv_shift);
            const int2 ui = (bidx == 0) ? CV_DESCALE((c2 - yi) * coeffs[3] + delta, yuv_shift) : CV_DESCALE((c0 - yi) * coeffs[3] + delta, yuv_shift);
            const int2 vi = (bidx == 0) ? CV_DESCALE((c0 - yi) * coeffs[4] + delta, yuv_shift) : CV_DESCALE((c2 - yi) * coeffs[4] + delta, yuv_shift);

            const VECTOR2 Y  = SAT_CAST2(yi);
            const VECTOR2 Cr = SAT_CAST2(ui);
            const VECTOR2 Cb = SAT_CAST2(vi);
#endif

            vstore8((VECTOR8)(Y.s0, Cr.s0, Cb.s0, 0, Y.s1, Cr.s1, Cb.s1, 0), 0, dst_ptr);
        }
#elif (4 == pixels_per_work_item)
        {
#ifndef DEPTH_5
            const VECTOR16 r0 = vload16(0, src_ptr);
            const int4 c0 = convert_int4(r0.s048c);
            const int4 c1 = convert_int4(r0.s159d);
            const int4 c2 = convert_int4(r0.s26ae);

            const int4 yi = (bidx == 0) ? CV_DESCALE(c0 * coeffs[2] + c1 * coeffs[1] + c2 * coeffs[0], yuv_shift) : CV_DESCALE(c0 * coeffs[0] + c1 * coeffs[1] + c2 * coeffs[2], yuv_shift);
            const int4 ui = (bidx == 0) ? CV_DESCALE((c2 - yi) * coeffs[3] + delta, yuv_shift) : CV_DESCALE((c0 - yi) * coeffs[3] + delta, yuv_shift);
            const int4 vi = (bidx == 0) ? CV_DESCALE((c0 - yi) * coeffs[4] + delta, yuv_shift) : CV_DESCALE((c2 - yi) * coeffs[4] + delta, yuv_shift);

            const VECTOR4 Y  = SAT_CAST4(yi);
            const VECTOR4 Cr = SAT_CAST4(ui);
            const VECTOR4 Cb = SAT_CAST4(vi);

            vstore16((VECTOR16)(Y.s0, Cr.s0, Cb.s0, 0, Y.s1, Cr.s1, Cb.s1, 0, Y.s2, Cr.s2, Cb.s2, 0, Y.s3, Cr.s3, Cb.s3, 0), 0, dst_ptr);
#endif
        }
#endif //pixels_per_work_item
    }
}

__constant float c_YCrCb2RGBCoeffs_f[4] = { 1.403f, -0.714f, -0.344f, 1.773f };
__constant int   c_YCrCb2RGBCoeffs_i[4] = { 22987, -11698, -5636, 29049 };

__kernel void YCrCb2RGB(int cols, int rows, int src_step, int dst_step,
                      __global const DATA_TYPE* src, __global DATA_TYPE* dst,
                      int src_offset, int dst_offset)
{
    int x = get_global_id(0) * pixels_per_work_item;
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        global DATA_TYPE *src_ptr = (global DATA_TYPE *)(src + src_idx);
        global DATA_TYPE *dst_ptr = (global DATA_TYPE *)(dst + dst_idx);

#ifdef DEPTH_5
        __constant float * coeffs = c_YCrCb2RGBCoeffs_f;
#else
        __constant int * coeffs = c_YCrCb2RGBCoeffs_i;
#endif

#if (1 == pixels_per_work_item)
        {
            const DATA_TYPE ycrcb[] = {src_ptr[0], src_ptr[1], src_ptr[2]};

#ifdef DEPTH_5
            float B = ycrcb[0] + (ycrcb[2] - HALF_MAX) * coeffs[3];
            float G = ycrcb[0] + (ycrcb[2] - HALF_MAX) * coeffs[2] + (ycrcb[1] - HALF_MAX) * coeffs[1];
            float R = ycrcb[0] + (ycrcb[1] - HALF_MAX) * coeffs[0];
#else
            int B = ycrcb[0] + CV_DESCALE((ycrcb[2] - HALF_MAX) * coeffs[3], yuv_shift);
            int G = ycrcb[0] + CV_DESCALE((ycrcb[2] - HALF_MAX) * coeffs[2] + (ycrcb[1] - HALF_MAX) * coeffs[1], yuv_shift);
            int R = ycrcb[0] + CV_DESCALE((ycrcb[1] - HALF_MAX) * coeffs[0], yuv_shift);
#endif

            dst_ptr[bidx]     = SAT_CAST( B );
            dst_ptr[1]        = SAT_CAST( G );
            dst_ptr[(bidx^2)] = SAT_CAST( R );
#if dcn == 4
            dst_ptr[3]         = MAX_NUM;
#endif
        }
#elif (2 == pixels_per_work_item)
        {
            const VECTOR8 r0 = vload8(0, src_ptr);

#ifdef DEPTH_5
            const float2 Y  = r0.s04;
            const float2 Cr = r0.s15;
            const float2 Cb = r0.s26;

            const float2 c0 = (bidx == 0) ? (Y + (Cb - HALF_MAX) * coeffs[3]) : (Y + (Cr - HALF_MAX) * coeffs[0]);
            const float2 c1 = Y + (Cb - HALF_MAX) * coeffs[2] + (Cr - HALF_MAX) * coeffs[1];
            const float2 c2 = (bidx == 0) ? (Y + (Cr - HALF_MAX) * coeffs[0]) : (Y + (Cb - HALF_MAX) * coeffs[3]);
#else
            const int2 Y  = convert_int2(r0.s04);
            const int2 Cr = convert_int2(r0.s15);
            const int2 Cb = convert_int2(r0.s26);

            const int2 c0i = (bidx == 0) ? (Y + CV_DESCALE((Cb - HALF_MAX) * coeffs[3], yuv_shift)) : (Y + CV_DESCALE((Cr - HALF_MAX) * coeffs[0], yuv_shift));
            const int2 c1i = Y + CV_DESCALE((Cb - HALF_MAX) * coeffs[2] + (Cr - HALF_MAX) * coeffs[1], yuv_shift);
            const int2 c2i = (bidx == 0) ? (Y + CV_DESCALE((Cr - HALF_MAX) * coeffs[0], yuv_shift)) : (Y + CV_DESCALE((Cb - HALF_MAX) * coeffs[3], yuv_shift));

            const VECTOR2 c0 = SAT_CAST2(c0i);
            const VECTOR2 c1 = SAT_CAST2(c1i);
            const VECTOR2 c2 = SAT_CAST2(c2i);
#endif

#if dcn == 4
            vstore8((VECTOR8)(c0.s0, c1.s0, c2.s0, MAX_NUM, c0.s1, c1.s1, c2.s1, MAX_NUM), 0, dst_ptr);
#else
            vstore8((VECTOR8)(c0.s0, c1.s0, c2.s0, 0, c0.s1, c1.s1, c2.s1, 0), 0, dst_ptr);
#endif
        }
#elif (4 == pixels_per_work_item)
        {
#ifndef DEPTH_5
            const VECTOR16 r0 = vload16(0, src_ptr);

            const int4 Y  = convert_int4(r0.s048c);
            const int4 Cr = convert_int4(r0.s159d);
            const int4 Cb = convert_int4(r0.s26ae);

            const int4 c0i = (bidx == 0) ? (Y + CV_DESCALE((Cb - HALF_MAX) * coeffs[3], yuv_shift)) : (Y + CV_DESCALE((Cr - HALF_MAX) * coeffs[0], yuv_shift));
            const int4 c1i = Y + CV_DESCALE((Cb - HALF_MAX) * coeffs[2] + (Cr - HALF_MAX) * coeffs[1], yuv_shift);
            const int4 c2i = (bidx == 0) ? (Y + CV_DESCALE((Cr - HALF_MAX) * coeffs[0], yuv_shift)) : (Y + CV_DESCALE((Cb - HALF_MAX) * coeffs[3], yuv_shift));

            const VECTOR4 c0 = SAT_CAST4(c0i);
            const VECTOR4 c1 = SAT_CAST4(c1i);
            const VECTOR4 c2 = SAT_CAST4(c2i);

#if dcn == 4
            vstore16((VECTOR16)(c0.s0, c1.s0, c2.s0, MAX_NUM, c0.s1, c1.s1, c2.s1, MAX_NUM, c0.s2, c1.s2, c2.s2, MAX_NUM, c0.s3, c1.s3, c2.s3, MAX_NUM), 0, dst_ptr);
#else
            vstore16((VECTOR16)(c0.s0, c1.s0, c2.s0, 0, c0.s1, c1.s1, c2.s1, 0, c0.s2, c1.s2, c2.s2, 0, c0.s3, c1.s3, c2.s3, 0), 0, dst_ptr);
#endif
#endif
        }
#endif //pixels_per_work_item
    }
}

///////////////////////////////////// RGB <-> XYZ //////////////////////////////////////

__kernel void RGB2XYZ(int cols, int rows, int src_step, int dst_step,
                      __global const DATA_TYPE* src, __global DATA_TYPE* dst,
                      int src_offset, int dst_offset, __constant COEFF_TYPE * coeffs)
{
    int dx = get_global_id(0) * pixels_per_work_item;
    int dy = get_global_id(1);

    if (dy < rows && dx < cols)
    {
        dx <<= 2;
        int src_idx = mad24(dy, src_step, src_offset + dx);
        int dst_idx = mad24(dy, dst_step, dst_offset + dx);

        global DATA_TYPE *src_ptr = (global DATA_TYPE *)(src + src_idx);
        global DATA_TYPE *dst_ptr = (global DATA_TYPE *)(dst + dst_idx);

#if (1 == pixels_per_work_item)
        {
            DATA_TYPE R = src_ptr[0], G = src_ptr[1], B = src_ptr[2];

#ifdef DEPTH_5
            float X = R * coeffs[0] + G * coeffs[1] + B * coeffs[2];
            float Y = R * coeffs[3] + G * coeffs[4] + B * coeffs[5];
            float Z = R * coeffs[6] + G * coeffs[7] + B * coeffs[8];
#else
            int X = CV_DESCALE(R * coeffs[0] + G * coeffs[1] + B * coeffs[2], xyz_shift);
            int Y = CV_DESCALE(R * coeffs[3] + G * coeffs[4] + B * coeffs[5], xyz_shift);
            int Z = CV_DESCALE(R * coeffs[6] + G * coeffs[7] + B * coeffs[8], xyz_shift);
#endif

            dst_ptr[0] = SAT_CAST( X );
            dst_ptr[1] = SAT_CAST( Y );
            dst_ptr[2] = SAT_CAST( Z );
        }
#elif (2 == pixels_per_work_item)
        {
            const VECTOR8 r0 = vload8(0, src_ptr);

#ifdef DEPTH_5
            const float2 R = r0.s04;
            const float2 G = r0.s15;
            const float2 B = r0.s26;

            const float2 X = R * coeffs[0] + G * coeffs[1] + B * coeffs[2];
            const float2 Y = R * coeffs[3] + G * coeffs[4] + B * coeffs[5];
            const float2 Z = R * coeffs[6] + G * coeffs[7] + B * coeffs[8];
#else
            const int2 R = convert_int2(r0.s04);
            const int2 G = convert_int2(r0.s15);
            const int2 B = convert_int2(r0.s26);

            const int2 xi = CV_DESCALE(R * coeffs[0] + G * coeffs[1] + B * coeffs[2], xyz_shift);
            const int2 yi = CV_DESCALE(R * coeffs[3] + G * coeffs[4] + B * coeffs[5], xyz_shift);
            const int2 zi = CV_DESCALE(R * coeffs[6] + G * coeffs[7] + B * coeffs[8], xyz_shift);

            const VECTOR2 X = SAT_CAST2(xi);
            const VECTOR2 Y = SAT_CAST2(yi);
            const VECTOR2 Z = SAT_CAST2(zi);
#endif

            vstore8((VECTOR8)(X.s0, Y.s0, Z.s0, 0, X.s1, Y.s1, Z.s1, 0), 0, dst_ptr);
        }
#elif (4 == pixels_per_work_item)
        {
#ifndef DEPTH_5
            const VECTOR16 r0 = vload16(0, src_ptr);

            const int4 R = convert_int4(r0.s048c);
            const int4 G = convert_int4(r0.s159d);
            const int4 B = convert_int4(r0.s26ae);

            const int4 xi = CV_DESCALE(R * coeffs[0] + G * coeffs[1] + B * coeffs[2], xyz_shift);
            const int4 yi = CV_DESCALE(R * coeffs[3] + G * coeffs[4] + B * coeffs[5], xyz_shift);
            const int4 zi = CV_DESCALE(R * coeffs[6] + G * coeffs[7] + B * coeffs[8], xyz_shift);

            const VECTOR4 X = SAT_CAST4(xi);
            const VECTOR4 Y = SAT_CAST4(yi);
            const VECTOR4 Z = SAT_CAST4(zi);

            vstore16((VECTOR16)(X.s0, Y.s0, Z.s0, 0, X.s1, Y.s1, Z.s1, 0, X.s2, Y.s2, Z.s2, 0, X.s3, Y.s3, Z.s3, 0), 0, dst_ptr);
#endif
        }
#endif //pixels_per_work_item
    }
}

__kernel void XYZ2RGB(int cols, int rows, int src_step, int dst_step,
                      __global const DATA_TYPE* src, __global DATA_TYPE* dst,
                      int src_offset, int dst_offset, __constant COEFF_TYPE * coeffs)
{
    int dx = get_global_id(0) * pixels_per_work_item;
    int dy = get_global_id(1);

    if (dy < rows && dx < cols)
    {
        dx <<= 2;
        int src_idx = mad24(dy, src_step, src_offset + dx);
        int dst_idx = mad24(dy, dst_step, dst_offset + dx);

        global DATA_TYPE *src_ptr = (global DATA_TYPE *)(src + src_idx);
        global DATA_TYPE *dst_ptr = (global DATA_TYPE *)(dst + dst_idx);

#if (1 == pixels_per_work_item)
        {
            const DATA_TYPE X = src_ptr[0], Y = src_ptr[1], Z = src_ptr[2];

#ifdef DEPTH_5
            float B = X * coeffs[0] + Y * coeffs[1] + Z * coeffs[2];
            float G = X * coeffs[3] + Y * coeffs[4] + Z * coeffs[5];
            float R = X * coeffs[6] + Y * coeffs[7] + Z * coeffs[8];
#else
            int B = CV_DESCALE(X * coeffs[0] + Y * coeffs[1] + Z * coeffs[2], xyz_shift);
            int G = CV_DESCALE(X * coeffs[3] + Y * coeffs[4] + Z * coeffs[5], xyz_shift);
            int R = CV_DESCALE(X * coeffs[6] + Y * coeffs[7] + Z * coeffs[8], xyz_shift);
#endif

            dst_ptr[0] = SAT_CAST( B );
            dst_ptr[1] = SAT_CAST( G );
            dst_ptr[2] = SAT_CAST( R );
#if dcn == 4
            dst_ptr[3] = MAX_NUM;
#endif
        }
#elif (2 == pixels_per_work_item)
        {
            const VECTOR8 r0 = vload8(0, src_ptr);

#ifdef DEPTH_5
            const float2 X = r0.s04;
            const float2 Y = r0.s15;
            const float2 Z = r0.s26;

            float2 B = X * coeffs[0] + Y * coeffs[1] + Z * coeffs[2];
            float2 G = X * coeffs[3] + Y * coeffs[4] + Z * coeffs[5];
            float2 R = X * coeffs[6] + Y * coeffs[7] + Z * coeffs[8];
#else
            const int2 xi = convert_int2(r0.s04);
            const int2 yi = convert_int2(r0.s15);
            const int2 zi = convert_int2(r0.s26);

            const int2 bi = CV_DESCALE(xi * coeffs[0] + yi * coeffs[1] + zi * coeffs[2], xyz_shift);
            const int2 gi = CV_DESCALE(xi * coeffs[3] + yi * coeffs[4] + zi * coeffs[5], xyz_shift);
            const int2 ri = CV_DESCALE(xi * coeffs[6] + yi * coeffs[7] + zi * coeffs[8], xyz_shift);

            const VECTOR2 R = SAT_CAST2(ri);
            const VECTOR2 G = SAT_CAST2(gi);
            const VECTOR2 B = SAT_CAST2(bi);
#endif

#if dcn == 4
            vstore8((VECTOR8)(B.s0, G.s0, R.s0, MAX_NUM, B.s1, G.s1, R.s1, MAX_NUM), 0, dst_ptr);
#else
            vstore8((VECTOR8)(B.s0, G.s0, R.s0, 0, B.s1, G.s1, R.s1, 0), 0, dst_ptr);
#endif
        }
#elif (4 == pixels_per_work_item)
        {
#ifndef DEPTH_5
            const VECTOR16 r0 = vload16(0, src_ptr);

            const int4 xi = convert_int4(r0.s048c);
            const int4 yi = convert_int4(r0.s159d);
            const int4 zi = convert_int4(r0.s26ae);

            const int4 bi = CV_DESCALE(xi * coeffs[0] + yi * coeffs[1] + zi * coeffs[2], xyz_shift);
            const int4 gi = CV_DESCALE(xi * coeffs[3] + yi * coeffs[4] + zi * coeffs[5], xyz_shift);
            const int4 ri = CV_DESCALE(xi * coeffs[6] + yi * coeffs[7] + zi * coeffs[8], xyz_shift);

            const VECTOR4 R = SAT_CAST4(ri);
            const VECTOR4 G = SAT_CAST4(gi);
            const VECTOR4 B = SAT_CAST4(bi);

#if dcn == 4
            vstore16((VECTOR16)(B.s0, G.s0, R.s0, MAX_NUM, B.s1, G.s1, R.s1, MAX_NUM, B.s2, G.s2, R.s2, MAX_NUM, B.s3, G.s3, R.s3, MAX_NUM), 0, dst_ptr);
#else
            vstore16((VECTOR16)(B.s0, G.s0, R.s0, 0, B.s1, G.s1, R.s1, 0, B.s2, G.s2, R.s2, 0, B.s3, G.s3, R.s3, 0), 0, dst_ptr);
#endif
#endif
        }
#endif // pixels_per_work_item
    }
}

///////////////////////////////////// RGB[A] <-> BGR[A] //////////////////////////////////////

__kernel void RGB(int cols, int rows, int src_step, int dst_step,
                  __global const DATA_TYPE * src, __global DATA_TYPE * dst,
                  int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

#ifndef INTEL_DEVICE
#ifdef REVERSE
        dst[dst_idx] = src[src_idx + 2];
        dst[dst_idx + 1] = src[src_idx + 1];
        dst[dst_idx + 2] = src[src_idx];
#elif defined ORDER
        dst[dst_idx] = src[src_idx];
        dst[dst_idx + 1] = src[src_idx + 1];
        dst[dst_idx + 2] = src[src_idx + 2];
#endif

#if dcn == 4
#if scn == 3
        dst[dst_idx + 3] = MAX_NUM;
#else
        dst[dst_idx + 3] = src[src_idx + 3];
#endif
#endif
#else //INTEL_DEVICE
        global DATA_TYPE *src_ptr = (global DATA_TYPE *)(src + src_idx);
        global DATA_TYPE *dst_ptr = (global DATA_TYPE *)(dst + dst_idx);

        const VECTOR4 r0 = vload4(0, src_ptr);
#ifdef REVERSE
        if (3 == dcn)
        {
            vstore4((VECTOR4)(r0.s210, 0), 0, dst_ptr);
        }
        else if (3 == scn)
        {
            vstore4((VECTOR4)(r0.s210, MAX_NUM), 0, dst_ptr);
        }
        else {
            vstore4((VECTOR4)(r0.s2103), 0, dst_ptr);
        }
#elif defined ORDER
        if (3 == dcn)
        {
            vstore4((VECTOR4)(r0.s012, 0), 0, dst_ptr);
        }
        else if (3 == scn)
        {
            vstore4((VECTOR4)(r0.s012, MAX_NUM), 0, dst_ptr);
        }
        else {
            vstore4(r0, 0, dst_ptr);
        }
#endif
#endif //INTEL_DEVICE
    }
}

///////////////////////////////////// RGB5x5 <-> RGB //////////////////////////////////////

__kernel void RGB5x52RGB(int cols, int rows, int src_step, int dst_step,
                         __global const ushort * src, __global uchar * dst,
                         int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + (x << 2));
        ushort t = src[src_idx];

#if greenbits == 6
        dst[dst_idx + bidx] = (uchar)(t << 3);
        dst[dst_idx + 1] = (uchar)((t >> 3) & ~3);
        dst[dst_idx + (bidx^2)] = (uchar)((t >> 8) & ~7);
#else
        dst[dst_idx + bidx] = (uchar)(t << 3);
        dst[dst_idx + 1] = (uchar)((t >> 2) & ~7);
        dst[dst_idx + (bidx^2)] = (uchar)((t >> 7) & ~7);
#endif

#if dcn == 4
#if greenbits == 6
        dst[dst_idx + 3] = 255;
#else
        dst[dst_idx + 3] = t & 0x8000 ? 255 : 0;
#endif
#endif
    }
}

__kernel void RGB2RGB5x5(int cols, int rows, int src_step, int dst_step,
                         __global const uchar * src, __global ushort * dst,
                         int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int src_idx = mad24(y, src_step, src_offset + (x << 2));
        int dst_idx = mad24(y, dst_step, dst_offset + x);

#if greenbits == 6
            dst[dst_idx] = (ushort)((src[src_idx + bidx] >> 3)|((src[src_idx + 1]&~3) << 3)|((src[src_idx + (bidx^2)]&~7) << 8));
#elif scn == 3
            dst[dst_idx] = (ushort)((src[src_idx + bidx] >> 3)|((src[src_idx + 1]&~7) << 2)|((src[src_idx + (bidx^2)]&~7) << 7));
#else
            dst[dst_idx] = (ushort)((src[src_idx + bidx] >> 3)|((src[src_idx + 1]&~7) << 2)|
                ((src[src_idx + (bidx^2)]&~7) << 7)|(src[src_idx + 3] ? 0x8000 : 0));
#endif
    }
}

///////////////////////////////////// RGB5x5 <-> RGB //////////////////////////////////////

__kernel void BGR5x52Gray(int cols, int rows, int src_step, int dst_step,
                          __global const ushort * src, __global uchar * dst,
                          int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);
        int t = src[src_idx];

#if greenbits == 6
        dst[dst_idx] = (uchar)CV_DESCALE(((t << 3) & 0xf8)*B2Y +
                                         ((t >> 3) & 0xfc)*G2Y +
                                         ((t >> 8) & 0xf8)*R2Y, yuv_shift);
#else
        dst[dst_idx] = (uchar)CV_DESCALE(((t << 3) & 0xf8)*B2Y +
                                         ((t >> 2) & 0xf8)*G2Y +
                                         ((t >> 7) & 0xf8)*R2Y, yuv_shift);
#endif
    }
}

__kernel void Gray2BGR5x5(int cols, int rows, int src_step, int dst_step,
                          __global const uchar * src, __global ushort * dst,
                          int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);
        int t = src[src_idx];

#if greenbits == 6
        dst[dst_idx] = (ushort)((t >> 3) | ((t & ~3) << 3) | ((t & ~7) << 8));
#else
        t >>= 3;
        dst[dst_idx] = (ushort)(t|(t << 5)|(t << 10));
#endif
    }
}

///////////////////////////////////// RGB <-> HSV //////////////////////////////////////

__constant int sector_data[][3] = { {1, 3, 0}, { 1, 0, 2 }, { 3, 0, 1 }, { 0, 2, 1 }, { 0, 1, 3 }, { 2, 1, 0 } };

#ifdef DEPTH_0

__kernel void RGB2HSV(int cols, int rows, int src_step, int dst_step,
                      __global const uchar * src, __global uchar * dst,
                      int src_offset, int dst_offset,
                      __constant int * sdiv_table, __constant int * hdiv_table)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        int b = src[src_idx + bidx], g = src[src_idx + 1], r = src[src_idx + (bidx^2)];
        int h, s, v = b;
        int vmin = b, diff;
        int vr, vg;

        v = max( v, g );
        v = max( v, r );
        vmin = min( vmin, g );
        vmin = min( vmin, r );

        diff = v - vmin;
        vr = v == r ? -1 : 0;
        vg = v == g ? -1 : 0;

        s = (diff * sdiv_table[v] + (1 << (hsv_shift-1))) >> hsv_shift;
        h = (vr & (g - b)) +
            (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
        h = (h * hdiv_table[diff] + (1 << (hsv_shift-1))) >> hsv_shift;
        h += h < 0 ? hrange : 0;

        dst[dst_idx] = convert_uchar_sat_rte(h);
        dst[dst_idx + 1] = (uchar)s;
        dst[dst_idx + 2] = (uchar)v;
    }
}

__kernel void HSV2RGB(int cols, int rows, int src_step, int dst_step,
                      __global const uchar * src, __global uchar * dst,
                      int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        float h = src[src_idx], s = src[src_idx + 1]*(1/255.f), v = src[src_idx + 2]*(1/255.f);
        float b, g, r;

        if (s != 0)
        {
            float tab[4];
            int sector;
            h *= hscale;
            if( h < 0 )
                do h += 6; while( h < 0 );
            else if( h >= 6 )
                do h -= 6; while( h >= 6 );
            sector = convert_int_sat_rtn(h);
            h -= sector;
            if( (unsigned)sector >= 6u )
            {
                sector = 0;
                h = 0.f;
            }

            tab[0] = v;
            tab[1] = v*(1.f - s);
            tab[2] = v*(1.f - s*h);
            tab[3] = v*(1.f - s*(1.f - h));

            b = tab[sector_data[sector][0]];
            g = tab[sector_data[sector][1]];
            r = tab[sector_data[sector][2]];
        }
        else
            b = g = r = v;

        dst[dst_idx + bidx] = convert_uchar_sat_rte(b*255.f);
        dst[dst_idx + 1] = convert_uchar_sat_rte(g*255.f);
        dst[dst_idx + (bidx^2)] = convert_uchar_sat_rte(r*255.f);
#if dcn == 4
        dst[dst_idx + 3] = MAX_NUM;
#endif
    }
}

#elif defined DEPTH_5

__kernel void RGB2HSV(int cols, int rows, int src_step, int dst_step,
                      __global const float * src, __global float * dst,
                      int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        float b = src[src_idx + bidx], g = src[src_idx + 1], r = src[src_idx + (bidx^2)];
        float h, s, v;

        float vmin, diff;

        v = vmin = r;
        if( v < g ) v = g;
        if( v < b ) v = b;
        if( vmin > g ) vmin = g;
        if( vmin > b ) vmin = b;

        diff = v - vmin;
        s = diff/(float)(fabs(v) + FLT_EPSILON);
        diff = (float)(60./(diff + FLT_EPSILON));
        if( v == r )
            h = (g - b)*diff;
        else if( v == g )
            h = (b - r)*diff + 120.f;
        else
            h = (r - g)*diff + 240.f;

        if( h < 0 ) h += 360.f;

        dst[dst_idx] = h*hscale;
        dst[dst_idx + 1] = s;
        dst[dst_idx + 2] = v;
    }
}

__kernel void HSV2RGB(int cols, int rows, int src_step, int dst_step,
                      __global const float * src, __global float * dst,
                      int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        float h = src[src_idx], s = src[src_idx + 1], v = src[src_idx + 2];
        float b, g, r;

        if (s != 0)
        {
            float tab[4];
            int sector;
            h *= hscale;
            if(h < 0)
                do h += 6; while (h < 0);
            else if (h >= 6)
                do h -= 6; while (h >= 6);
            sector = convert_int_sat_rtn(h);
            h -= sector;
            if ((unsigned)sector >= 6u)
            {
                sector = 0;
                h = 0.f;
            }

            tab[0] = v;
            tab[1] = v*(1.f - s);
            tab[2] = v*(1.f - s*h);
            tab[3] = v*(1.f - s*(1.f - h));

            b = tab[sector_data[sector][0]];
            g = tab[sector_data[sector][1]];
            r = tab[sector_data[sector][2]];
        }
        else
            b = g = r = v;

        dst[dst_idx + bidx] = b;
        dst[dst_idx + 1] = g;
        dst[dst_idx + (bidx^2)] = r;
#if dcn == 4
        dst[dst_idx + 3] = MAX_NUM;
#endif
    }
}

#endif

///////////////////////////////////// RGB <-> HLS //////////////////////////////////////

#ifdef DEPTH_0

__kernel void RGB2HLS(int cols, int rows, int src_step, int dst_step,
                      __global const uchar * src, __global uchar * dst,
                      int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        float b = src[src_idx + bidx]*(1/255.f), g = src[src_idx + 1]*(1/255.f), r = src[src_idx + (bidx^2)]*(1/255.f);
        float h = 0.f, s = 0.f, l;
        float vmin, vmax, diff;

        vmax = vmin = r;
        if (vmax < g) vmax = g;
        if (vmax < b) vmax = b;
        if (vmin > g) vmin = g;
        if (vmin > b) vmin = b;

        diff = vmax - vmin;
        l = (vmax + vmin)*0.5f;

        if (diff > FLT_EPSILON)
        {
            s = l < 0.5f ? diff/(vmax + vmin) : diff/(2 - vmax - vmin);
            diff = 60.f/diff;

            if( vmax == r )
                h = (g - b)*diff;
            else if( vmax == g )
                h = (b - r)*diff + 120.f;
            else
                h = (r - g)*diff + 240.f;

            if( h < 0.f ) h += 360.f;
        }

        dst[dst_idx] = convert_uchar_sat_rte(h*hscale);
        dst[dst_idx + 1] = convert_uchar_sat_rte(l*255.f);
        dst[dst_idx + 2] = convert_uchar_sat_rte(s*255.f);
    }
}

__kernel void HLS2RGB(int cols, int rows, int src_step, int dst_step,
                      __global const uchar * src, __global uchar * dst,
                      int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        float h = src[src_idx], l = src[src_idx + 1]*(1.f/255.f), s = src[src_idx + 2]*(1.f/255.f);
        float b, g, r;

        if (s != 0)
        {
            float tab[4];

            float p2 = l <= 0.5f ? l*(1 + s) : l + s - l*s;
            float p1 = 2*l - p2;

            h *= hscale;
            if( h < 0 )
                do h += 6; while( h < 0 );
            else if( h >= 6 )
                do h -= 6; while( h >= 6 );

            int sector = convert_int_sat_rtn(h);
            h -= sector;

            tab[0] = p2;
            tab[1] = p1;
            tab[2] = p1 + (p2 - p1)*(1-h);
            tab[3] = p1 + (p2 - p1)*h;

            b = tab[sector_data[sector][0]];
            g = tab[sector_data[sector][1]];
            r = tab[sector_data[sector][2]];
        }
        else
            b = g = r = l;

        dst[dst_idx + bidx] = convert_uchar_sat_rte(b*255.f);
        dst[dst_idx + 1] = convert_uchar_sat_rte(g*255.f);
        dst[dst_idx + (bidx^2)] = convert_uchar_sat_rte(r*255.f);
#if dcn == 4
        dst[dst_idx + 3] = MAX_NUM;
#endif
    }
}

#elif defined DEPTH_5

__kernel void RGB2HLS(int cols, int rows, int src_step, int dst_step,
                      __global const float * src, __global float * dst,
                      int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        float b = src[src_idx + bidx], g = src[src_idx + 1], r = src[src_idx + (bidx^2)];
        float h = 0.f, s = 0.f, l;
        float vmin, vmax, diff;

        vmax = vmin = r;
        if (vmax < g) vmax = g;
        if (vmax < b) vmax = b;
        if (vmin > g) vmin = g;
        if (vmin > b) vmin = b;

        diff = vmax - vmin;
        l = (vmax + vmin)*0.5f;

        if (diff > FLT_EPSILON)
        {
            s = l < 0.5f ? diff/(vmax + vmin) : diff/(2 - vmax - vmin);
            diff = 60.f/diff;

            if( vmax == r )
                h = (g - b)*diff;
            else if( vmax == g )
                h = (b - r)*diff + 120.f;
            else
                h = (r - g)*diff + 240.f;

            if( h < 0.f ) h += 360.f;
        }

        dst[dst_idx] = h*hscale;
        dst[dst_idx + 1] = l;
        dst[dst_idx + 2] = s;
    }
}

__kernel void HLS2RGB(int cols, int rows, int src_step, int dst_step,
                      __global const float * src, __global float * dst,
                      int src_offset, int dst_offset)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        float h = src[src_idx], l = src[src_idx + 1], s = src[src_idx + 2];
        float b, g, r;

        if (s != 0)
        {
            float tab[4];
            int sector;

            float p2 = l <= 0.5f ? l*(1 + s) : l + s - l*s;
            float p1 = 2*l - p2;

            h *= hscale;
            if( h < 0 )
                do h += 6; while( h < 0 );
            else if( h >= 6 )
                do h -= 6; while( h >= 6 );

            sector = convert_int_sat_rtn(h);
            h -= sector;

            tab[0] = p2;
            tab[1] = p1;
            tab[2] = p1 + (p2 - p1)*(1-h);
            tab[3] = p1 + (p2 - p1)*h;

            b = tab[sector_data[sector][0]];
            g = tab[sector_data[sector][1]];
            r = tab[sector_data[sector][2]];
        }
        else
            b = g = r = l;

        dst[dst_idx + bidx] = b;
        dst[dst_idx + 1] = g;
        dst[dst_idx + (bidx^2)] = r;
#if dcn == 4
        dst[dst_idx + 3] = MAX_NUM;
#endif
    }
}

#endif

/////////////////////////// RGBA <-> mRGBA (alpha premultiplied) //////////////

#ifdef DEPTH_0

__kernel void RGBA2mRGBA(int cols, int rows, int src_step, int dst_step,
                        __global const uchar * src, __global uchar * dst,
                        int src_offset, int dst_offset)
{
    int x = get_global_id(0) * pixels_per_work_item;
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        global DATA_TYPE *src_ptr = (global DATA_TYPE *)(src + src_idx);
        global DATA_TYPE *dst_ptr = (global DATA_TYPE *)(dst + dst_idx);

#if (1 == pixels_per_work_item)
        {
            const uchar4 r0 = vload4(0, src_ptr);

            dst_ptr[0] = (r0.s0 * r0.s3 + HALF_MAX) / MAX_NUM;
            dst_ptr[1] = (r0.s1 * r0.s3 + HALF_MAX) / MAX_NUM;
            dst_ptr[2] = (r0.s2 * r0.s3 + HALF_MAX) / MAX_NUM;
            dst_ptr[3] = r0.s3;
        }
#elif (2 == pixels_per_work_item)
        {
            const uchar8 r0 = vload8(0, src_ptr);

            const int2 v0 = convert_int2(r0.s04);
            const int2 v1 = convert_int2(r0.s15);
            const int2 v2 = convert_int2(r0.s26);
            const int2 v3 = convert_int2(r0.s37);

            const int2 ri = (v0 * v3 + HALF_MAX) / MAX_NUM;
            const int2 gi = (v1 * v3 + HALF_MAX) / MAX_NUM;
            const int2 bi = (v2 * v3 + HALF_MAX) / MAX_NUM;

            const uchar2 r = convert_uchar2(ri);
            const uchar2 g = convert_uchar2(gi);
            const uchar2 b = convert_uchar2(bi);

            vstore8((uchar8)(r.s0, g.s0, b.s0, v3.s0, r.s1, g.s1, b.s1, v3.s1), 0, dst_ptr);
        }
#elif (4 == pixels_per_work_item)
        {
            const uchar16 r0 = vload16(0, src_ptr);

            const int4 v0 = convert_int4(r0.s048c);
            const int4 v1 = convert_int4(r0.s159d);
            const int4 v2 = convert_int4(r0.s26ae);
            const int4 v3 = convert_int4(r0.s37bf);

            const int4 ri = (v0 * v3 + HALF_MAX) / MAX_NUM;
            const int4 gi = (v1 * v3 + HALF_MAX) / MAX_NUM;
            const int4 bi = (v2 * v3 + HALF_MAX) / MAX_NUM;

            const uchar4 r = convert_uchar4(ri);
            const uchar4 g = convert_uchar4(gi);
            const uchar4 b = convert_uchar4(bi);

            vstore16((uchar16)(r.s0, g.s0, b.s0, v3.s0, r.s1, g.s1, b.s1, v3.s1, r.s2, g.s2, b.s2, v3.s2, r.s3, g.s3, b.s3, v3.s3), 0, dst_ptr);
        }
#endif // pixels_per_work_item
    }
}

__kernel void mRGBA2RGBA(int cols, int rows, int src_step, int dst_step,
                        __global const uchar * src, __global uchar * dst,
                        int src_offset, int dst_offset)
{
    int x = get_global_id(0) * pixels_per_work_item;
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        x <<= 2;
        int src_idx = mad24(y, src_step, src_offset + x);
        int dst_idx = mad24(y, dst_step, dst_offset + x);

        global DATA_TYPE *src_ptr = (global DATA_TYPE *)(src + src_idx);
        global DATA_TYPE *dst_ptr = (global DATA_TYPE *)(dst + dst_idx);

#if (1 == pixels_per_work_item)
        {
            const uchar4 r0 = vload4(0, src_ptr);
            const uchar v3_half = r0.s3 / 2;

            const uchar r = (r0.s3 == 0) ? 0 : (r0.s0 * MAX_NUM + v3_half) / r0.s3;
            const uchar g = (r0.s3 == 0) ? 0 : (r0.s1 * MAX_NUM + v3_half) / r0.s3;
            const uchar b = (r0.s3 == 0) ? 0 : (r0.s2 * MAX_NUM + v3_half) / r0.s3;

            vstore4((uchar4)(r, g, b, r0.s3), 0, dst_ptr);
        }
#elif (2 == pixels_per_work_item)
        {
            const uchar8 r0 = vload8(0, src_ptr);

            const int2 v0 = convert_int2(r0.s04);
            const int2 v1 = convert_int2(r0.s15);
            const int2 v2 = convert_int2(r0.s26);
            const int2 v3 = convert_int2(r0.s37);
            const int2 v3_half = v3 / 2;

            const int2 ri = (v3 == 0) ? 0 : (v0 * MAX_NUM + v3_half) / v3;
            const int2 gi = (v3 == 0) ? 0 : (v1 * MAX_NUM + v3_half) / v3;
            const int2 bi = (v3 == 0) ? 0 : (v2 * MAX_NUM + v3_half) / v3;

            const uchar2 r = convert_uchar2(ri);
            const uchar2 g = convert_uchar2(gi);
            const uchar2 b = convert_uchar2(bi);

            vstore8((uchar8)(r.s0, g.s0, b.s0, v3.s0, r.s1, g.s1, b.s1, v3.s1), 0, dst_ptr);
        }
#elif (4 == pixels_per_work_item)
        {
            const uchar16 r0 = vload16(0, src_ptr);

            const int4 v0 = convert_int4(r0.s048c);
            const int4 v1 = convert_int4(r0.s159d);
            const int4 v2 = convert_int4(r0.s26ae);
            const int4 v3 = convert_int4(r0.s37bf);
            const int4 v3_half = v3 / 2;


            const int4 ri = (v3 == 0) ? 0 : (v0 * MAX_NUM + v3_half) / v3;
            const int4 gi = (v3 == 0) ? 0 : (v1 * MAX_NUM + v3_half) / v3;
            const int4 bi = (v3 == 0) ? 0 : (v2 * MAX_NUM + v3_half) / v3;

            const uchar4 r = convert_uchar4(ri);
            const uchar4 g = convert_uchar4(gi);
            const uchar4 b = convert_uchar4(bi);

            vstore16((uchar16)(r.s0, g.s0, b.s0, v3.s0, r.s1, g.s1, b.s1, v3.s1, r.s2, g.s2, b.s2, v3.s2, r.s3, g.s3, b.s3, v3.s3), 0, dst_ptr);
        }
#endif // pixels_per_work_item
    }
}

#endif
