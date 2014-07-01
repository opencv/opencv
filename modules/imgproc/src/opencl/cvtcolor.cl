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

#if depth == 0
    #define DATA_TYPE uchar
    #define MAX_NUM  255
    #define HALF_MAX 128
    #define COEFF_TYPE int
    #define SAT_CAST(num) convert_uchar_sat(num)
    #define DEPTH_0
#elif depth == 2
    #define DATA_TYPE ushort
    #define MAX_NUM  65535
    #define HALF_MAX 32768
    #define COEFF_TYPE int
    #define SAT_CAST(num) convert_ushort_sat(num)
    #define DEPTH_2
#elif depth == 5
    #define DATA_TYPE float
    #define MAX_NUM  1.0f
    #define HALF_MAX 0.5f
    #define COEFF_TYPE float
    #define SAT_CAST(num) (num)
    #define DEPTH_5
#else
    #error "invalid depth: should be 0 (CV_8U), 2 (CV_16U) or 5 (CV_32F)"
#endif

#ifndef STRIPE_SIZE
#define STRIPE_SIZE 1
#endif

#define CV_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))

enum
{
    yuv_shift  = 14,
    xyz_shift  = 12,
    hsv_shift = 12,
    R2Y        = 4899,
    G2Y        = 9617,
    B2Y        = 1868,
    BLOCK_SIZE = 256
};

#define scnbytes ((int)sizeof(DATA_TYPE)*scn)
#define dcnbytes ((int)sizeof(DATA_TYPE)*dcn)

#ifndef hscale
#define hscale 0
#endif

#ifndef hrange
#define hrange 0
#endif

#if bidx == 0
#define R_COMP z
#define G_COMP y
#define B_COMP x
#elif bidx == 2
#define R_COMP x
#define G_COMP y
#define B_COMP z
#elif bidx == 3
// The only kernel that uses bidx == 3 doesn't use these macros.
// But we still need to make the compiler happy.
#define R_COMP w
#define G_COMP w
#define B_COMP w
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define DATA_TYPE_4 CAT(DATA_TYPE, 4)

///////////////////////////////////// RGB <-> GRAY //////////////////////////////////////

__kernel void RGB2Gray(__global const uchar* srcptr, int srcstep, int srcoffset,
                       __global uchar* dstptr, int dststep, int dstoffset,
                       int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + mad24(y, srcstep, srcoffset + x * scnbytes));
                __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + mad24(y, dststep, dstoffset + x * dcnbytes));
                DATA_TYPE_4 src_pix = vload4(0, src);
#ifdef DEPTH_5
                dst[0] = src_pix.B_COMP * 0.114f + src_pix.G_COMP * 0.587f + src_pix.R_COMP * 0.299f;
#else
                dst[0] = (DATA_TYPE)CV_DESCALE((src_pix.B_COMP * B2Y + src_pix.G_COMP * G2Y + src_pix.R_COMP * R2Y), yuv_shift);
#endif
            }
            ++y;
        }
    }
}

__kernel void Gray2RGB(__global const uchar* srcptr, int srcstep, int srcoffset,
                       __global uchar* dstptr, int dststep, int dstoffset,
                       int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + mad24(y, srcstep, srcoffset + x * scnbytes));
                __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + mad24(y, dststep, dstoffset + x * dcnbytes));
                DATA_TYPE val = src[0];
                dst[0] = dst[1] = dst[2] = val;
#if dcn == 4
                dst[3] = MAX_NUM;
#endif
            }
            ++y;
        }
    }
}

///////////////////////////////////// RGB <-> YUV //////////////////////////////////////

__constant float c_RGB2YUVCoeffs_f[5]  = { 0.114f, 0.587f, 0.299f, 0.492f, 0.877f };
__constant int   c_RGB2YUVCoeffs_i[5]  = { B2Y, G2Y, R2Y, 8061, 14369 };

__kernel void RGB2YUV(__global const uchar* srcptr, int srcstep, int srcoffset,
                      __global uchar* dstptr, int dststep, int dstoffset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + mad24(y, srcstep, srcoffset + x * scnbytes));
                __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + mad24(y, dststep, dstoffset + x * dcnbytes));
                DATA_TYPE_4 src_pix = vload4(0, src);
                DATA_TYPE b=src_pix.B_COMP, g=src_pix.G_COMP, r=src_pix.R_COMP;

#ifdef DEPTH_5
                __constant float * coeffs = c_RGB2YUVCoeffs_f;
                const DATA_TYPE Y  = b * coeffs[0] + g * coeffs[1] + r * coeffs[2];
                const DATA_TYPE U = (b - Y) * coeffs[3] + HALF_MAX;
                const DATA_TYPE V = (r - Y) * coeffs[4] + HALF_MAX;
#else
                __constant int * coeffs = c_RGB2YUVCoeffs_i;
                const int delta = HALF_MAX * (1 << yuv_shift);
                const int Y = CV_DESCALE(b * coeffs[0] + g * coeffs[1] + r * coeffs[2], yuv_shift);
                const int U = CV_DESCALE((b - Y) * coeffs[3] + delta, yuv_shift);
                const int V = CV_DESCALE((r - Y) * coeffs[4] + delta, yuv_shift);
#endif

                dst[0] = SAT_CAST( Y );
                dst[1] = SAT_CAST( U );
                dst[2] = SAT_CAST( V );
            }
            ++y;
        }
    }
}

__constant float c_YUV2RGBCoeffs_f[5] = { 2.032f, -0.395f, -0.581f, 1.140f };
__constant int   c_YUV2RGBCoeffs_i[5] = { 33292, -6472, -9519, 18678 };

__kernel void YUV2RGB(__global const uchar* srcptr, int srcstep, int srcoffset,
                      __global uchar* dstptr, int dststep, int dstoffset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + mad24(y, srcstep, srcoffset + x * scnbytes));
                __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + mad24(y, dststep, dstoffset + x * dcnbytes));
                DATA_TYPE_4 src_pix = vload4(0, src);
                DATA_TYPE Y = src_pix.x, U = src_pix.y, V = src_pix.z;

#ifdef DEPTH_5
                __constant float * coeffs = c_YUV2RGBCoeffs_f;
                const float r = Y + (V - HALF_MAX) * coeffs[3];
                const float g = Y + (V - HALF_MAX) * coeffs[2] + (U - HALF_MAX) * coeffs[1];
                const float b = Y + (U - HALF_MAX) * coeffs[0];
#else
                __constant int * coeffs = c_YUV2RGBCoeffs_i;
                const int r = Y + CV_DESCALE((V - HALF_MAX) * coeffs[3], yuv_shift);
                const int g = Y + CV_DESCALE((V - HALF_MAX) * coeffs[2] + (U - HALF_MAX) * coeffs[1], yuv_shift);
                const int b = Y + CV_DESCALE((U - HALF_MAX) * coeffs[0], yuv_shift);
#endif

                dst[bidx] = SAT_CAST( b );
                dst[1] = SAT_CAST( g );
                dst[bidx^2] = SAT_CAST( r );
#if dcn == 4
                dst[3] = MAX_NUM;
#endif
            }
            ++y;
        }
    }
}

__constant int ITUR_BT_601_CY = 1220542;
__constant int ITUR_BT_601_CUB = 2116026;
__constant int ITUR_BT_601_CUG = 409993;
__constant int ITUR_BT_601_CVG = 852492;
__constant int ITUR_BT_601_CVR = 1673527;
__constant int ITUR_BT_601_SHIFT = 20;

__kernel void YUV2RGB_NV12(__global const uchar* srcptr, int srcstep, int srcoffset,
                            __global uchar* dstptr, int dststep, int dstoffset,
                            int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols / 2)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows / 2 )
            {
                __global const uchar* ysrc = srcptr + mad24(y << 1, srcstep, (x << 1) + srcoffset);
                __global const uchar* usrc = srcptr + mad24(rows + y, srcstep, (x << 1) + srcoffset);
                __global uchar*       dst1 = dstptr + mad24(y << 1, dststep, x * (dcn<<1) + dstoffset);
                __global uchar*       dst2 = dstptr + mad24((y << 1) + 1, dststep, x * (dcn<<1) + dstoffset);

                int Y1 = ysrc[0];
                int Y2 = ysrc[1];
                int Y3 = ysrc[srcstep];
                int Y4 = ysrc[srcstep + 1];

                int U  = usrc[0] - 128;
                int V  = usrc[1] - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * V;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) - ITUR_BT_601_CVG * V - ITUR_BT_601_CUG * U;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * U;

                Y1 = max(0, Y1 - 16) * ITUR_BT_601_CY;
                dst1[2 - bidx]     = convert_uchar_sat((Y1 + ruv) >> ITUR_BT_601_SHIFT);
                dst1[1]        = convert_uchar_sat((Y1 + guv) >> ITUR_BT_601_SHIFT);
                dst1[bidx] = convert_uchar_sat((Y1 + buv) >> ITUR_BT_601_SHIFT);
#if dcn == 4
                dst1[3]        = 255;
#endif

                Y2 = max(0, Y2 - 16) * ITUR_BT_601_CY;
                dst1[dcn + 2 - bidx] = convert_uchar_sat((Y2 + ruv) >> ITUR_BT_601_SHIFT);
                dst1[dcn + 1]        = convert_uchar_sat((Y2 + guv) >> ITUR_BT_601_SHIFT);
                dst1[dcn + bidx] = convert_uchar_sat((Y2 + buv) >> ITUR_BT_601_SHIFT);
#if dcn == 4
                dst1[7]        = 255;
#endif

                Y3 = max(0, Y3 - 16) * ITUR_BT_601_CY;
                dst2[2 - bidx]     = convert_uchar_sat((Y3 + ruv) >> ITUR_BT_601_SHIFT);
                dst2[1]        = convert_uchar_sat((Y3 + guv) >> ITUR_BT_601_SHIFT);
                dst2[bidx] = convert_uchar_sat((Y3 + buv) >> ITUR_BT_601_SHIFT);
#if dcn == 4
                dst2[3]        = 255;
#endif

                Y4 = max(0, Y4 - 16) * ITUR_BT_601_CY;
                dst2[dcn + 2 - bidx] = convert_uchar_sat((Y4 + ruv) >> ITUR_BT_601_SHIFT);
                dst2[dcn + 1]        = convert_uchar_sat((Y4 + guv) >> ITUR_BT_601_SHIFT);
                dst2[dcn + bidx] = convert_uchar_sat((Y4 + buv) >> ITUR_BT_601_SHIFT);
#if dcn == 4
                dst2[7]        = 255;
#endif
            }
            ++y;
        }
    }
}

///////////////////////////////////// RGB <-> YCrCb //////////////////////////////////////

__constant float c_RGB2YCrCbCoeffs_f[5] = {0.299f, 0.587f, 0.114f, 0.713f, 0.564f};
__constant int   c_RGB2YCrCbCoeffs_i[5] = {R2Y, G2Y, B2Y, 11682, 9241};

__kernel void RGB2YCrCb(__global const uchar* srcptr, int srcstep, int srcoffset,
                        __global uchar* dstptr, int dststep, int dstoffset,
                        int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + mad24(y, srcstep, srcoffset + x * scnbytes));
                __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + mad24(y, dststep, dstoffset + x * dcnbytes));
                DATA_TYPE_4 src_pix = vload4(0, src);
                DATA_TYPE b=src_pix.B_COMP, g=src_pix.G_COMP, r=src_pix.R_COMP;

#ifdef DEPTH_5
                __constant float * coeffs = c_RGB2YCrCbCoeffs_f;
                DATA_TYPE Y = b * coeffs[2] + g * coeffs[1] + r * coeffs[0];
                DATA_TYPE Cr = (r - Y) * coeffs[3] + HALF_MAX;
                DATA_TYPE Cb = (b - Y) * coeffs[4] + HALF_MAX;
#else
                __constant int * coeffs = c_RGB2YCrCbCoeffs_i;
                int delta = HALF_MAX * (1 << yuv_shift);
                int Y =  CV_DESCALE(b * coeffs[2] + g * coeffs[1] + r * coeffs[0], yuv_shift);
                int Cr = CV_DESCALE((r - Y) * coeffs[3] + delta, yuv_shift);
                int Cb = CV_DESCALE((b - Y) * coeffs[4] + delta, yuv_shift);
#endif

                dst[0] = SAT_CAST( Y );
                dst[1] = SAT_CAST( Cr );
                dst[2] = SAT_CAST( Cb );
            }
            ++y;
        }
    }
}

__constant float c_YCrCb2RGBCoeffs_f[4] = { 1.403f, -0.714f, -0.344f, 1.773f };
__constant int   c_YCrCb2RGBCoeffs_i[4] = { 22987, -11698, -5636, 29049 };

__kernel void YCrCb2RGB(__global const uchar* src, int src_step, int src_offset,
                        __global uchar* dst, int dst_step, int dst_offset,
                        int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);
                __global const DATA_TYPE * srcptr = (__global const DATA_TYPE*)(src + src_idx);
                __global DATA_TYPE * dstptr = (__global DATA_TYPE*)(dst + dst_idx);

                DATA_TYPE_4 src_pix = vload4(0, srcptr);
                DATA_TYPE y = src_pix.x, cr = src_pix.y, cb = src_pix.z;

#ifdef DEPTH_5
                __constant float * coeff = c_YCrCb2RGBCoeffs_f;
                float r = y + coeff[0] * (cr - HALF_MAX);
                float g = y + coeff[1] * (cr - HALF_MAX) + coeff[2] * (cb - HALF_MAX);
                float b = y + coeff[3] * (cb - HALF_MAX);
#else
                __constant int * coeff = c_YCrCb2RGBCoeffs_i;
                int r = y + CV_DESCALE(coeff[0] * (cr - HALF_MAX), yuv_shift);
                int g = y + CV_DESCALE(coeff[1] * (cr - HALF_MAX) + coeff[2] * (cb - HALF_MAX), yuv_shift);
                int b = y + CV_DESCALE(coeff[3] * (cb - HALF_MAX), yuv_shift);
#endif

                dstptr[(bidx^2)] = SAT_CAST(r);
                dstptr[1] = SAT_CAST(g);
                dstptr[bidx] = SAT_CAST(b);
#if dcn == 4
                dstptr[3] = MAX_NUM;
#endif
            }
            ++y;
        }
    }
}

///////////////////////////////////// RGB <-> XYZ //////////////////////////////////////

__kernel void RGB2XYZ(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset,
                      int rows, int cols, __constant COEFF_TYPE * coeffs)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1) * PIX_PER_WI_Y;

    if (dx < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (dy < rows)
            {
                int src_idx = mad24(dy, src_step, src_offset + dx * scnbytes);
                int dst_idx = mad24(dy, dst_step, dst_offset + dx * dcnbytes);

                __global const DATA_TYPE * src = (__global const DATA_TYPE *)(srcptr + src_idx);
                __global DATA_TYPE * dst = (__global DATA_TYPE *)(dstptr + dst_idx);

                DATA_TYPE_4 src_pix = vload4(0, src);
                DATA_TYPE r = src_pix.x, g = src_pix.y, b = src_pix.z;

#ifdef DEPTH_5
                float x = r * coeffs[0] + g * coeffs[1] + b * coeffs[2];
                float y = r * coeffs[3] + g * coeffs[4] + b * coeffs[5];
                float z = r * coeffs[6] + g * coeffs[7] + b * coeffs[8];
#else
                int x = CV_DESCALE(r * coeffs[0] + g * coeffs[1] + b * coeffs[2], xyz_shift);
                int y = CV_DESCALE(r * coeffs[3] + g * coeffs[4] + b * coeffs[5], xyz_shift);
                int z = CV_DESCALE(r * coeffs[6] + g * coeffs[7] + b * coeffs[8], xyz_shift);
#endif
                dst[0] = SAT_CAST(x);
                dst[1] = SAT_CAST(y);
                dst[2] = SAT_CAST(z);
            }
            ++dy;
        }
    }
}

__kernel void XYZ2RGB(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset,
                      int rows, int cols, __constant COEFF_TYPE * coeffs)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1) * PIX_PER_WI_Y;

    if (dx < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (dy < rows)
            {
                int src_idx = mad24(dy, src_step, src_offset + dx * scnbytes);
                int dst_idx = mad24(dy, dst_step, dst_offset + dx * dcnbytes);

                __global const DATA_TYPE * src = (__global const DATA_TYPE *)(srcptr + src_idx);
                __global DATA_TYPE * dst = (__global DATA_TYPE *)(dstptr + dst_idx);

                DATA_TYPE_4 src_pix = vload4(0, src);
                DATA_TYPE x = src_pix.x, y = src_pix.y, z = src_pix.z;

#ifdef DEPTH_5
                float b = x * coeffs[0] + y * coeffs[1] + z * coeffs[2];
                float g = x * coeffs[3] + y * coeffs[4] + z * coeffs[5];
                float r = x * coeffs[6] + y * coeffs[7] + z * coeffs[8];
#else
                int b = CV_DESCALE(x * coeffs[0] + y * coeffs[1] + z * coeffs[2], xyz_shift);
                int g = CV_DESCALE(x * coeffs[3] + y * coeffs[4] + z * coeffs[5], xyz_shift);
                int r = CV_DESCALE(x * coeffs[6] + y * coeffs[7] + z * coeffs[8], xyz_shift);
#endif
                dst[0] = SAT_CAST(b);
                dst[1] = SAT_CAST(g);
                dst[2] = SAT_CAST(r);
#if dcn == 4
                dst[3] = MAX_NUM;
#endif
            }
            ++dy;
        }
    }
}

///////////////////////////////////// RGB[A] <-> BGR[A] //////////////////////////////////////

__kernel void RGB(__global const uchar* srcptr, int src_step, int src_offset,
                  __global uchar* dstptr, int dst_step, int dst_offset,
                  int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);

                __global const DATA_TYPE * src = (__global const DATA_TYPE *)(srcptr + src_idx);
                __global DATA_TYPE * dst = (__global DATA_TYPE *)(dstptr + dst_idx);
                DATA_TYPE_4 src_pix = vload4(0, src);

#ifdef REVERSE
                dst[0] = src_pix.z;
                dst[1] = src_pix.y;
                dst[2] = src_pix.x;
#else
                dst[0] = src_pix.x;
                dst[1] = src_pix.y;
                dst[2] = src_pix.z;
#endif

#if dcn == 4
#if scn == 3
                dst[3] = MAX_NUM;
#else
                dst[3] = src[3];
#endif
#endif
            }
            ++y;
        }
    }
}

///////////////////////////////////// RGB5x5 <-> RGB //////////////////////////////////////

__kernel void RGB5x52RGB(__global const uchar* src, int src_step, int src_offset,
                         __global uchar* dst, int dst_step, int dst_offset,
                         int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);
                ushort t = *((__global const ushort*)(src + src_idx));

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
            ++y;
        }
    }
}

__kernel void RGB2RGB5x5(__global const uchar* src, int src_step, int src_offset,
                         __global uchar* dst, int dst_step, int dst_offset,
                         int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);
                uchar4 src_pix = vload4(0, src + src_idx);

#if greenbits == 6
                    *((__global ushort*)(dst + dst_idx)) = (ushort)((src_pix.B_COMP >> 3)|((src_pix.G_COMP&~3) << 3)|((src_pix.R_COMP&~7) << 8));
#elif scn == 3
                    *((__global ushort*)(dst + dst_idx)) = (ushort)((src_pix.B_COMP >> 3)|((src_pix.G_COMP&~7) << 2)|((src_pix.R_COMP&~7) << 7));
#else
                    *((__global ushort*)(dst + dst_idx)) = (ushort)((src_pix.B_COMP >> 3)|((src_pix.G_COMP&~7) << 2)|
                        ((src_pix.R_COMP&~7) << 7)|(src_pix.w ? 0x8000 : 0));
#endif
            }
            ++y;
        }
    }
}

///////////////////////////////////// RGB5x5 <-> Gray //////////////////////////////////////

__kernel void BGR5x52Gray(__global const uchar* src, int src_step, int src_offset,
                          __global uchar* dst, int dst_step, int dst_offset,
                          int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x);
                int t = *((__global const ushort*)(src + src_idx));

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
            ++y;
        }
    }
}

__kernel void Gray2BGR5x5(__global const uchar* src, int src_step, int src_offset,
                          __global uchar* dst, int dst_step, int dst_offset,
                          int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);
                int t = src[src_idx];

#if greenbits == 6
                *((__global ushort*)(dst + dst_idx)) = (ushort)((t >> 3) | ((t & ~3) << 3) | ((t & ~7) << 8));
#else
                t >>= 3;
                *((__global ushort*)(dst + dst_idx)) = (ushort)(t|(t << 5)|(t << 10));
#endif
            }
            ++y;
        }
    }
}

//////////////////////////////////// RGB <-> HSV //////////////////////////////////////

__constant int sector_data[][3] = { { 1, 3, 0 },
                                    { 1, 0, 2 },
                                    { 3, 0, 1 },
                                    { 0, 2, 1 },
                                    { 0, 1, 3 },
                                    { 2, 1, 0 } };

#ifdef DEPTH_0

__kernel void RGB2HSV(__global const uchar* src, int src_step, int src_offset,
                      __global uchar* dst, int dst_step, int dst_offset,
                      int rows, int cols,
                      __constant int * sdiv_table, __constant int * hdiv_table)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);
                uchar4 src_pix = vload4(0, src + src_idx);

                int b = src_pix.B_COMP, g = src_pix.G_COMP, r = src_pix.R_COMP;
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
            ++y;
        }
    }
}

__kernel void HSV2RGB(__global const uchar* src, int src_step, int src_offset,
                      __global uchar* dst, int dst_step, int dst_offset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);
                uchar4 src_pix = vload4(0, src + src_idx);

                float h = src_pix.x, s = src_pix.y*(1/255.f), v = src_pix.z*(1/255.f);
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
            ++y;
        }
    }
}

#elif defined DEPTH_5

__kernel void RGB2HSV(__global const uchar* srcptr, int src_step, int src_offset,
                      __global uchar* dstptr, int dst_step, int dst_offset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);

                __global const float * src = (__global const float *)(srcptr + src_idx);
                __global float * dst = (__global float *)(dstptr + dst_idx);
                float4 src_pix = vload4(0, src);

                float b = src_pix.B_COMP, g = src_pix.G_COMP, r = src_pix.R_COMP;
                float h, s, v;

                float vmin, diff;

                v = vmin = r;
                if( v < g ) v = g;
                if( v < b ) v = b;
                if( vmin > g ) vmin = g;
                if( vmin > b ) vmin = b;

                diff = v - vmin;
                s = diff/(float)(fabs(v) + FLT_EPSILON);
                diff = (float)(60.f/(diff + FLT_EPSILON));
                if( v == r )
                    h = (g - b)*diff;
                else if( v == g )
                    h = (b - r)*diff + 120.f;
                else
                    h = (r - g)*diff + 240.f;

                if( h < 0 ) h += 360.f;

                dst[0] = h*hscale;
                dst[1] = s;
                dst[2] = v;
            }
            ++y;
        }
    }
}

__kernel void HSV2RGB(__global const uchar* srcptr, int src_step, int src_offset,
                      __global uchar* dstptr, int dst_step, int dst_offset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);

                __global const float * src = (__global const float *)(srcptr + src_idx);
                __global float * dst = (__global float *)(dstptr + dst_idx);
                float4 src_pix = vload4(0, src);

                float h = src_pix.x, s = src_pix.y, v = src_pix.z;
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

                dst[bidx] = b;
                dst[1] = g;
                dst[bidx^2] = r;
#if dcn == 4
                dst[3] = MAX_NUM;
#endif
            }
            ++y;
        }
    }
}

#endif

///////////////////////////////////// RGB <-> HLS //////////////////////////////////////

#ifdef DEPTH_0

__kernel void RGB2HLS(__global const uchar* src, int src_step, int src_offset,
                      __global uchar* dst, int dst_step, int dst_offset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);
                uchar4 src_pix = vload4(0, src + src_idx);

                float b = src_pix.B_COMP*(1/255.f), g = src_pix.G_COMP*(1/255.f), r = src_pix.R_COMP*(1/255.f);
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
            ++y;
        }
    }
}

__kernel void HLS2RGB(__global const uchar* src, int src_step, int src_offset,
                      __global uchar* dst, int dst_step, int dst_offset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);
                uchar4 src_pix = vload4(0, src + src_idx);

                float h = src_pix.x, l = src_pix.y*(1.f/255.f), s = src_pix.z*(1.f/255.f);
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
            ++y;
        }
    }
}

#elif defined DEPTH_5

__kernel void RGB2HLS(__global const uchar* srcptr, int src_step, int src_offset,
                      __global uchar* dstptr, int dst_step, int dst_offset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);

                __global const float * src = (__global const float *)(srcptr + src_idx);
                __global float * dst = (__global float *)(dstptr + dst_idx);
                float4 src_pix = vload4(0, src);

                float b = src_pix.B_COMP, g = src_pix.G_COMP, r = src_pix.R_COMP;
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

                dst[0] = h*hscale;
                dst[1] = l;
                dst[2] = s;
            }
            ++y;
        }
    }
}

__kernel void HLS2RGB(__global const uchar* srcptr, int src_step, int src_offset,
                      __global uchar* dstptr, int dst_step, int dst_offset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);

                __global const float * src = (__global const float *)(srcptr + src_idx);
                __global float * dst = (__global float *)(dstptr + dst_idx);
                float4 src_pix = vload4(0, src);

                float h = src_pix.x, l = src_pix.y, s = src_pix.z;
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

                dst[bidx] = b;
                dst[1] = g;
                dst[bidx^2] = r;
#if dcn == 4
                dst[3] = MAX_NUM;
#endif
            }
            ++y;
        }
    }
}

#endif

/////////////////////////// RGBA <-> mRGBA (alpha premultiplied) //////////////

#ifdef DEPTH_0

__kernel void RGBA2mRGBA(__global const uchar* src, int src_step, int src_offset,
                         __global uchar* dst, int dst_step, int dst_offset,
                         int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + (x << 2));
                int dst_idx = mad24(y, dst_step, dst_offset + (x << 2));
                uchar4 src_pix = vload4(0, src + src_idx);

                uchar v0 = src_pix.x, v1 = src_pix.y;
                uchar v2 = src_pix.z, v3 = src_pix.w;

                dst[dst_idx] = (v0 * v3 + HALF_MAX) / MAX_NUM;
                dst[dst_idx + 1] = (v1 * v3 + HALF_MAX) / MAX_NUM;
                dst[dst_idx + 2] = (v2 * v3 + HALF_MAX) / MAX_NUM;
                dst[dst_idx + 3] = v3;
            }
            ++y;
        }
    }
}

__kernel void mRGBA2RGBA(__global const uchar* src, int src_step, int src_offset,
                         __global uchar* dst, int dst_step, int dst_offset,
                         int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + (x << 2));
                int dst_idx = mad24(y, dst_step, dst_offset + (x << 2));
                uchar4 src_pix = vload4(0, src + src_idx);

                uchar v0 = src_pix.x, v1 = src_pix.y;
                uchar v2 = src_pix.z, v3 = src_pix.w;
                uchar v3_half = v3 / 2;

                dst[dst_idx] = v3 == 0 ? 0 : (v0 * MAX_NUM + v3_half) / v3;
                dst[dst_idx + 1] = v3 == 0 ? 0 : (v1 * MAX_NUM + v3_half) / v3;
                dst[dst_idx + 2] = v3 == 0 ? 0 : (v2 * MAX_NUM + v3_half) / v3;
                dst[dst_idx + 3] = v3;
            }
            ++y;
        }
    }
}

#endif

/////////////////////////////////// [l|s]RGB <-> Lab ///////////////////////////

#define lab_shift xyz_shift
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define GAMMA_TAB_SIZE 1024
#define GammaTabScale (float)GAMMA_TAB_SIZE

inline float splineInterpolate(float x, __global const float * tab, int n)
{
    int ix = clamp(convert_int_sat_rtn(x), 0, n-1);
    x -= ix;
    tab += ix*4;
    return ((tab[3]*x + tab[2])*x + tab[1])*x + tab[0];
}

#ifdef DEPTH_0

__kernel void BGR2Lab(__global const uchar * src, int src_step, int src_offset,
                      __global uchar * dst, int dst_step, int dst_offset, int rows, int cols,
                      __global const ushort * gammaTab, __global ushort * LabCbrtTab_b,
                      __constant int * coeffs, int Lscale, int Lshift)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);

                __global const uchar* src_ptr = src + src_idx;
                __global uchar* dst_ptr = dst + dst_idx;
                uchar4 src_pix = vload4(0, src_ptr);

                int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
                    C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
                    C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

                int R = gammaTab[src_pix.x], G = gammaTab[src_pix.y], B = gammaTab[src_pix.z];
                int fX = LabCbrtTab_b[CV_DESCALE(R*C0 + G*C1 + B*C2, lab_shift)];
                int fY = LabCbrtTab_b[CV_DESCALE(R*C3 + G*C4 + B*C5, lab_shift)];
                int fZ = LabCbrtTab_b[CV_DESCALE(R*C6 + G*C7 + B*C8, lab_shift)];

                int L = CV_DESCALE( Lscale*fY + Lshift, lab_shift2 );
                int a = CV_DESCALE( 500*(fX - fY) + 128*(1 << lab_shift2), lab_shift2 );
                int b = CV_DESCALE( 200*(fY - fZ) + 128*(1 << lab_shift2), lab_shift2 );

                dst_ptr[0] = SAT_CAST(L);
                dst_ptr[1] = SAT_CAST(a);
                dst_ptr[2] = SAT_CAST(b);
            }
            ++y;
        }
    }
}

#elif defined DEPTH_5

__kernel void BGR2Lab(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float _1_3, float _a)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);

                __global const float * src = (__global const float *)(srcptr + src_idx);
                __global float * dst = (__global float *)(dstptr + dst_idx);
                float4 src_pix = vload4(0, src);

                float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
                      C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
                      C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

                float R = clamp(src_pix.x, 0.0f, 1.0f);
                float G = clamp(src_pix.y, 0.0f, 1.0f);
                float B = clamp(src_pix.z, 0.0f, 1.0f);

#ifdef SRGB
                R = splineInterpolate(R * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif

                float X = R*C0 + G*C1 + B*C2;
                float Y = R*C3 + G*C4 + B*C5;
                float Z = R*C6 + G*C7 + B*C8;

                float FX = X > 0.008856f ? pow(X, _1_3) : (7.787f * X + _a);
                float FY = Y > 0.008856f ? pow(Y, _1_3) : (7.787f * Y + _a);
                float FZ = Z > 0.008856f ? pow(Z, _1_3) : (7.787f * Z + _a);

                float L = Y > 0.008856f ? (116.f * FY - 16.f) : (903.3f * Y);
                float a = 500.f * (FX - FY);
                float b = 200.f * (FY - FZ);

                dst[0] = L;
                dst[1] = a;
                dst[2] = b;
            }
            ++y;
        }
    }
}

#endif

inline void Lab2BGR_f(const float * srcbuf, float * dstbuf,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float lThresh, float fThresh)
{
    float li = srcbuf[0], ai = srcbuf[1], bi = srcbuf[2];

    float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
          C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
          C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

    float y, fy;
    if (li <= lThresh)
    {
        y = li / 903.3f;
        fy = 7.787f * y + 16.0f / 116.0f;
    }
    else
    {
        fy = (li + 16.0f) / 116.0f;
        y = fy * fy * fy;
    }

    float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };

    for (int j = 0; j < 2; j++)
        if (fxz[j] <= fThresh)
            fxz[j] = (fxz[j] - 16.0f / 116.0f) / 7.787f;
        else
            fxz[j] = fxz[j] * fxz[j] * fxz[j];

    float x = fxz[0], z = fxz[1];
    float ro = clamp(C0 * x + C1 * y + C2 * z, 0.0f, 1.0f);
    float go = clamp(C3 * x + C4 * y + C5 * z, 0.0f, 1.0f);
    float bo = clamp(C6 * x + C7 * y + C8 * z, 0.0f, 1.0f);

#ifdef SRGB
    ro = splineInterpolate(ro * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
    go = splineInterpolate(go * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
    bo = splineInterpolate(bo * GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif

    dstbuf[0] = ro, dstbuf[1] = go, dstbuf[2] = bo;
}

#ifdef DEPTH_0

__kernel void Lab2BGR(__global const uchar * src, int src_step, int src_offset,
                      __global uchar * dst, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float lThresh, float fThresh)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);

                __global const uchar* src_ptr = src + src_idx;
                __global uchar* dst_ptr = dst + dst_idx;
                uchar4 src_pix = vload4(0, src_ptr);

                float srcbuf[3], dstbuf[3];
                srcbuf[0] = src_pix.x*(100.f/255.f);
                srcbuf[1] = convert_float(src_pix.y - 128);
                srcbuf[2] = convert_float(src_pix.z - 128);

                Lab2BGR_f(&srcbuf[0], &dstbuf[0],
#ifdef SRGB
                    gammaTab,
#endif
                    coeffs, lThresh, fThresh);

                dst_ptr[0] = SAT_CAST(dstbuf[0] * 255.0f);
                dst_ptr[1] = SAT_CAST(dstbuf[1] * 255.0f);
                dst_ptr[2] = SAT_CAST(dstbuf[2] * 255.0f);
#if dcn == 4
                dst_ptr[3] = MAX_NUM;
#endif
            }
            ++y;
        }
    }
}

#elif defined DEPTH_5

__kernel void Lab2BGR(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float lThresh, float fThresh)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
                int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);

                __global const float * src = (__global const float *)(srcptr + src_idx);
                __global float * dst = (__global float *)(dstptr + dst_idx);
                float4 src_pix = vload4(0, src);

                float srcbuf[3], dstbuf[3];
                srcbuf[0] = src_pix.x, srcbuf[1] = src_pix.y, srcbuf[2] = src_pix.z;

                Lab2BGR_f(&srcbuf[0], &dstbuf[0],
#ifdef SRGB
                    gammaTab,
#endif
                    coeffs, lThresh, fThresh);

                dst[0] = dstbuf[0], dst[1] = dstbuf[1], dst[2] = dstbuf[2];
#if dcn == 4
                dst[3] = MAX_NUM;
#endif
            }
            ++y;
        }
    }
}

#endif

/////////////////////////////////// [l|s]RGB <-> Luv ///////////////////////////

#define LAB_CBRT_TAB_SIZE 1024
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))

__constant float LabCbrtTabScale = LAB_CBRT_TAB_SIZE/1.5f;

#ifdef DEPTH_5

__kernel void BGR2Luv(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __global const float * LabCbrtTab, __constant float * coeffs, float _un, float _vn)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src_idx = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_idx = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        __global const float * src = (__global const float *)(srcptr + src_idx);
        __global float * dst = (__global float *)(dstptr + dst_idx);

        float R = src[0], G = src[1], B = src[2];

#ifdef SRGB
        R = splineInterpolate(R*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
        G = splineInterpolate(G*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
        B = splineInterpolate(B*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif
        float X = R*coeffs[0] + G*coeffs[1] + B*coeffs[2];
        float Y = R*coeffs[3] + G*coeffs[4] + B*coeffs[5];
        float Z = R*coeffs[6] + G*coeffs[7] + B*coeffs[8];

        float L = splineInterpolate(Y*LabCbrtTabScale, LabCbrtTab, LAB_CBRT_TAB_SIZE);
        L = 116.f*L - 16.f;

        float d = (4*13) / max(X + 15 * Y + 3 * Z, FLT_EPSILON);
        float u = L*(X*d - _un);
        float v = L*((9*0.25f)*Y*d - _vn);

        dst[0] = L;
        dst[1] = u;
        dst[2] = v;
    }
}

#elif defined DEPTH_0

__kernel void BGR2Luv(__global const uchar * src, int src_step, int src_offset,
                      __global uchar * dst, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __global const float * LabCbrtTab, __constant float * coeffs, float _un, float _vn)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src_idx = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_idx = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        src += src_idx;
        dst += dst_idx;

        float scale = 1.0f / 255.0f;
        float R = src[0]*scale, G = src[1]*scale, B = src[2]*scale;

#ifdef SRGB
        R = splineInterpolate(R*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
        G = splineInterpolate(G*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
        B = splineInterpolate(B*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif
        float X = R*coeffs[0] + G*coeffs[1] + B*coeffs[2];
        float Y = R*coeffs[3] + G*coeffs[4] + B*coeffs[5];
        float Z = R*coeffs[6] + G*coeffs[7] + B*coeffs[8];

        float L = splineInterpolate(Y*LabCbrtTabScale, LabCbrtTab, LAB_CBRT_TAB_SIZE);
        L = 116.f*L - 16.f;

        float d = (4*13) / max(X + 15 * Y + 3 * Z, FLT_EPSILON);
        float u = L*(X*d - _un);
        float v = L*((9*0.25f)*Y*d - _vn);

        dst[0] = SAT_CAST(L * 2.55f);
        dst[1] = SAT_CAST(mad(u, 0.72033898305084743f, 96.525423728813564f));
        dst[2] = SAT_CAST(mad(v, 0.99609375f, 139.453125f));
    }
}

#endif

#ifdef DEPTH_5

__kernel void Luv2BGR(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float _un, float _vn)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src_idx = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_idx = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        __global const float * src = (__global const float *)(srcptr + src_idx);
        __global float * dst = (__global float *)(dstptr + dst_idx);

        float L = src[0], u = src[1], v = src[2], d, X, Y, Z;
        Y = (L + 16.f) * (1.f/116.f);
        Y = Y*Y*Y;
        d = (1.f/13.f)/L;
        u = u*d + _un;
        v = v*d + _vn;
        float iv = 1.f/v;
        X = 2.25f * u * Y * iv ;
        Z = (12 - 3 * u - 20 * v) * Y * 0.25f * iv;

        float R = X*coeffs[0] + Y*coeffs[1] + Z*coeffs[2];
        float G = X*coeffs[3] + Y*coeffs[4] + Z*coeffs[5];
        float B = X*coeffs[6] + Y*coeffs[7] + Z*coeffs[8];

#ifdef SRGB
        R = splineInterpolate(R*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
        G = splineInterpolate(G*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
        B = splineInterpolate(B*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif

        dst[0] = R;
        dst[1] = G;
        dst[2] = B;
#if dcn == 4
        dst[3] = MAX_NUM;
#endif
    }
}

#elif defined DEPTH_0

__kernel void Luv2BGR(__global const uchar * src, int src_step, int src_offset,
                      __global uchar * dst, int dst_step, int dst_offset, int rows, int cols,
#ifdef SRGB
                      __global const float * gammaTab,
#endif
                      __constant float * coeffs, float _un, float _vn)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src_idx = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_idx = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        src += src_idx;
        dst += dst_idx;

        float d, X, Y, Z;
        float L = src[0]*(100.f/255.f);
        float u = (float)(src[1]*1.388235294117647f - 134.f);
        float v = (float)(src[2]*1.003921568627451f - 140.f);
        Y = (L + 16.f) * (1.f/116.f);
        Y = Y*Y*Y;
        d = (1.f/13.f)/L;
        u = u*d + _un;
        v = v*d + _vn;
        float iv = 1.f/v;
        X = 2.25f * u * Y * iv ;
        Z = (12 - 3 * u - 20 * v) * Y * 0.25f * iv;

        float R = X*coeffs[0] + Y*coeffs[1] + Z*coeffs[2];
        float G = X*coeffs[3] + Y*coeffs[4] + Z*coeffs[5];
        float B = X*coeffs[6] + Y*coeffs[7] + Z*coeffs[8];

#ifdef SRGB
        R = splineInterpolate(R*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
        G = splineInterpolate(G*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
        B = splineInterpolate(B*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif

        dst[0] = SAT_CAST(R * 255.0f);
        dst[1] = SAT_CAST(G * 255.0f);
        dst[2] = SAT_CAST(B * 255.0f);

#if dcn == 4
        dst[3] = MAX_NUM;
#endif
    }
}

#endif
