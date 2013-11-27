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

#if defined (DOUBLE_SUPPORT)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif

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

#define CV_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))

enum
{
    yuv_shift  = 14,
    xyz_shift  = 12,
    R2Y        = 4899,
    G2Y        = 9617,
    B2Y        = 1868,
    BLOCK_SIZE = 256
};

#define scnbytes ((int)sizeof(DATA_TYPE)*scn)
#define dcnbytes ((int)sizeof(DATA_TYPE)*dcn)

///////////////////////////////////// RGB <-> GRAY //////////////////////////////////////

__kernel void RGB2Gray(__global const uchar* srcptr, int srcstep, int srcoffset,
                       __global uchar* dstptr, int dststep, int dstoffset,
                       int rows, int cols)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + mad24(y, srcstep, srcoffset + x * scnbytes));
        __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + mad24(y, dststep, dstoffset + x * dcnbytes));
#ifdef DEPTH_5
        dst[0] = src[bidx] * 0.114f + src[1] * 0.587f + src[(bidx^2)] * 0.299f;
#else
        dst[0] = (DATA_TYPE)CV_DESCALE((src[bidx] * B2Y + src[1] * G2Y + src[(bidx^2)] * R2Y), yuv_shift);
#endif
    }
}

__kernel void Gray2RGB(__global const uchar* srcptr, int srcstep, int srcoffset,
                       __global uchar* dstptr, int dststep, int dstoffset,
                       int rows, int cols)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + mad24(y, srcstep, srcoffset + x * scnbytes));
        __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + mad24(y, dststep, dstoffset + x * dcnbytes));
        DATA_TYPE val = src[0];
        dst[0] = dst[1] = dst[2] = val;
#if dcn == 4
        dst[3] = MAX_NUM;
#endif
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
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + mad24(y, srcstep, srcoffset + x * scnbytes));
        __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + mad24(y, dststep, dstoffset + x * dcnbytes));
        DATA_TYPE b=src[bidx], g=src[1], r=src[bidx^2];

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
}

__constant float c_YUV2RGBCoeffs_f[5] = { 2.032f, -0.395f, -0.581f, 1.140f };
__constant int   c_YUV2RGBCoeffs_i[5] = { 33292, -6472, -9519, 18678 };

__kernel void YUV2RGB(__global const uchar* srcptr, int srcstep, int srcoffset,
                      __global uchar* dstptr, int dststep, int dstoffset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + mad24(y, srcstep, srcoffset + x * scnbytes));
        __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + mad24(y, dststep, dstoffset + x * dcnbytes));
        DATA_TYPE Y = src[0], U = src[1], V = src[2];

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
    int y = get_global_id(1);

    if (y < rows / 2 && x < cols / 2 )
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
}

///////////////////////////////////// RGB <-> YCrCb //////////////////////////////////////

__constant float c_RGB2YCrCbCoeffs_f[5] = {0.299f, 0.587f, 0.114f, 0.713f, 0.564f};
__constant int   c_RGB2YCrCbCoeffs_i[5] = {R2Y, G2Y, B2Y, 11682, 9241};

__kernel void RGB2YCrCb(__global const uchar* srcptr, int srcstep, int srcoffset,
                        __global uchar* dstptr, int dststep, int dstoffset,
                        int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + mad24(y, srcstep, srcoffset + x * scnbytes));
        __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + mad24(y, dststep, dstoffset + x * dcnbytes));
        DATA_TYPE b=src[bidx], g=src[1], r=src[bidx^2];

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
}

__constant float c_YCrCb2RGBCoeffs_f[4] = { 1.403f, -0.714f, -0.344f, 1.773f };
__constant int   c_YCrCb2RGBCoeffs_i[4] = { 22987, -11698, -5636, 29049 };

__kernel void YCrCb2RGB(__global const uchar* src, int src_step, int src_offset,
                        __global uchar* dst, int dst_step, int dst_offset,
                        int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
        int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);
        __global const DATA_TYPE * srcptr = (__global const DATA_TYPE*)(src + src_idx);
        __global DATA_TYPE * dstptr = (__global DATA_TYPE*)(dst + dst_idx);

        DATA_TYPE y = srcptr[0], cr = srcptr[1], cb = srcptr[2];

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
}

///////////////////////////////////// RGB <-> XYZ //////////////////////////////////////

__kernel void RGB2XYZ(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset,
                      int rows, int cols, __constant COEFF_TYPE * coeffs, int a1, int a2)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dy < rows && dx < cols)
    {
        int src_idx = mad24(dy, src_step, src_offset + dx * scnbytes);
        int dst_idx = mad24(dy, dst_step, dst_offset + dx * dcnbytes);

        __global const DATA_TYPE * src = (__global const DATA_TYPE *)(srcptr + src_idx);
        __global DATA_TYPE * dst = (__global DATA_TYPE *)(dstptr + dst_idx);

        DATA_TYPE r = src[0], g = src[1], b = src[2];

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
}

__kernel void XYZ2RGB(__global const uchar * srcptr, int src_step, int src_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset,
                      int rows, int cols, __constant COEFF_TYPE * coeffs, int a1, int a2)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dy < rows && dx < cols)
    {
        int src_idx = mad24(dy, src_step, src_offset + dx * scnbytes);
        int dst_idx = mad24(dy, dst_step, dst_offset + dx * dcnbytes);

        __global const DATA_TYPE * src = (__global const DATA_TYPE *)(srcptr + src_idx);
        __global DATA_TYPE * dst = (__global DATA_TYPE *)(dstptr + dst_idx);

        DATA_TYPE x = src[0], y = src[1], z = src[2];

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
}

///////////////////////////////////// RGB[A] <-> BGR[A] //////////////////////////////////////

__kernel void RGB(__global const uchar* srcptr, int src_step, int src_offset,
                  __global uchar* dstptr, int dst_step, int dst_offset,
                  int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int src_idx = mad24(y, src_step, src_offset + x * scnbytes);
        int dst_idx = mad24(y, dst_step, dst_offset + x * dcnbytes);

        __global const DATA_TYPE * src = (__global const DATA_TYPE *)(srcptr + src_idx);
        __global DATA_TYPE * dst = (__global DATA_TYPE *)(dstptr + dst_idx);

#ifdef REVERSE
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
#elif defined ORDER
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
#endif

#if dcn == 4
#if scn == 3
        dst[3] = MAX_NUM;
#else
        dst[3] = src[3];
#endif
#endif
    }
}



















