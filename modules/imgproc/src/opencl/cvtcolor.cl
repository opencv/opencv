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
    #define HALF_MAX_NUM 128
    #define COEFF_TYPE int
    #define SAT_CAST(num) convert_uchar_sat(num)
    #define DEPTH_0
#elif depth == 2
    #define DATA_TYPE ushort
    #define MAX_NUM  65535
    #define HALF_MAX_NUM 32768
    #define COEFF_TYPE int
    #define SAT_CAST(num) convert_ushort_sat(num)
    #define DEPTH_2
#elif depth == 5
    #define DATA_TYPE float
    #define MAX_NUM  1.0f
    #define HALF_MAX_NUM 0.5f
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
    hsv_shift  = 12,
    R2Y        = 4899,
    G2Y        = 9617,
    B2Y        = 1868,
    BLOCK_SIZE = 256
};

//constants for conversion from/to RGB and Gray, YUV, YCrCb according to BT.601
#define B2YF 0.114f
#define G2YF 0.587f
#define R2YF 0.299f
//to YCbCr
#define YCBF 0.564f
#define YCRF 0.713f
#define YCBI 9241
#define YCRI 11682
//to YUV
#define B2UF 0.492f
#define R2VF 0.877f
#define B2UI 8061
#define R2VI 14369
//from YUV
#define U2BF 2.032f
#define U2GF -0.395f
#define V2GF -0.581f
#define V2RF 1.140f
#define U2BI 33292
#define U2GI -6472
#define V2GI -9519
#define V2RI 18678
//from YCrCb
#define CR2RF 1.403f
#define CB2GF -0.344f
#define CR2GF -0.714f
#define CB2BF 1.773f
#define CR2RI 22987
#define CB2GI -5636
#define CR2GI -11698
#define CB2BI 29049

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
#else
#define R_COMP x
#define G_COMP y
#define B_COMP z
#endif

#ifndef uidx
#define uidx 0
#endif

#ifndef yidx
#define yidx 0
#endif

#ifndef PIX_PER_WI_X
#define PIX_PER_WI_X 1
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define DATA_TYPE_4 CAT(DATA_TYPE, 4)
#define DATA_TYPE_3 CAT(DATA_TYPE, 3)

///////////////////////////////////// RGB <-> GRAY //////////////////////////////////////

__kernel void RGB2Gray(__global const uchar * srcptr, int src_step, int src_offset,
                       __global uchar * dstptr, int dst_step, int dst_offset,
                       int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + src_index);
                __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + dst_index);
                DATA_TYPE_3 src_pix = vload3(0, src);
#ifdef DEPTH_5
                dst[0] = fma(src_pix.B_COMP, B2YF, fma(src_pix.G_COMP, G2YF, src_pix.R_COMP * R2YF));
#else
                dst[0] = (DATA_TYPE)CV_DESCALE(mad24(src_pix.B_COMP, B2Y, mad24(src_pix.G_COMP, G2Y, mul24(src_pix.R_COMP, R2Y))), yuv_shift);
#endif
                ++y;
                src_index += src_step;
                dst_index += dst_step;
            }
        }
    }
}

__kernel void Gray2RGB(__global const uchar * srcptr, int src_step, int src_offset,
                       __global uchar * dstptr, int dst_step, int dst_offset,
                       int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + src_index);
                __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + dst_index);
                DATA_TYPE val = src[0];
#if dcn == 3 || defined DEPTH_5
                dst[0] = dst[1] = dst[2] = val;
#if dcn == 4
                dst[3] = MAX_NUM;
#endif
#else
                *(__global DATA_TYPE_4 *)dst = (DATA_TYPE_4)(val, val, val, MAX_NUM);
#endif
                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}

///////////////////////////////////// RGB <-> YUV //////////////////////////////////////

__constant float c_RGB2YUVCoeffs_f[5]  = { B2YF, G2YF, R2YF, B2UF, R2VF };
__constant int   c_RGB2YUVCoeffs_i[5]  = { B2Y, G2Y, R2Y, B2UI, R2VI };

__kernel void RGB2YUV(__global const uchar* srcptr, int src_step, int src_offset,
                      __global uchar* dstptr, int dst_step, int dt_offset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dt_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + src_index);
                __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + dst_index);
                DATA_TYPE_3 src_pix = vload3(0, src);
                DATA_TYPE b = src_pix.B_COMP, g = src_pix.G_COMP, r = src_pix.R_COMP;

#ifdef DEPTH_5
                __constant float * coeffs = c_RGB2YUVCoeffs_f;
                const DATA_TYPE Y = fma(b, coeffs[0], fma(g, coeffs[1], r * coeffs[2]));
                const DATA_TYPE U = fma(b - Y, coeffs[3], HALF_MAX_NUM);
                const DATA_TYPE V = fma(r - Y, coeffs[4], HALF_MAX_NUM);
#else
                __constant int * coeffs = c_RGB2YUVCoeffs_i;
                const int delta = HALF_MAX_NUM * (1 << yuv_shift);
                const int Y = CV_DESCALE(mad24(b, coeffs[0], mad24(g, coeffs[1], mul24(r, coeffs[2]))), yuv_shift);
                const int U = CV_DESCALE(mad24(b - Y, coeffs[3], delta), yuv_shift);
                const int V = CV_DESCALE(mad24(r - Y, coeffs[4], delta), yuv_shift);
#endif

                dst[0] = SAT_CAST( Y );
                dst[1] = SAT_CAST( U );
                dst[2] = SAT_CAST( V );

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}

__constant float c_YUV2RGBCoeffs_f[4] = { U2BF, U2GF, V2GF, V2RF };
__constant int   c_YUV2RGBCoeffs_i[4] = { U2BI, U2GI, V2GI, V2RI };

__kernel void YUV2RGB(__global const uchar* srcptr, int src_step, int src_offset,
                      __global uchar* dstptr, int dst_step, int dt_offset,
                      int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dt_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + src_index);
                __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + dst_index);
                DATA_TYPE_4 src_pix = vload4(0, src);
                DATA_TYPE Y = src_pix.x, U = src_pix.y, V = src_pix.z;

#ifdef DEPTH_5
                __constant float * coeffs = c_YUV2RGBCoeffs_f;
                float r = fma(V - HALF_MAX_NUM, coeffs[3], Y);
                float g = fma(V - HALF_MAX_NUM, coeffs[2], fma(U - HALF_MAX_NUM, coeffs[1], Y));
                float b = fma(U - HALF_MAX_NUM, coeffs[0], Y);
#else
                __constant int * coeffs = c_YUV2RGBCoeffs_i;
                const int r = Y + CV_DESCALE(mul24(V - HALF_MAX_NUM, coeffs[3]), yuv_shift);
                const int g = Y + CV_DESCALE(mad24(V - HALF_MAX_NUM, coeffs[2], mul24(U - HALF_MAX_NUM, coeffs[1])), yuv_shift);
                const int b = Y + CV_DESCALE(mul24(U - HALF_MAX_NUM, coeffs[0]), yuv_shift);
#endif

                dst[bidx] = SAT_CAST( b );
                dst[1] = SAT_CAST( g );
                dst[bidx^2] = SAT_CAST( r );
#if dcn == 4
                dst[3] = MAX_NUM;
#endif
                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}
__constant float c_YUV2RGBCoeffs_420[5] = { 1.163999557f, 2.017999649f, -0.390999794f,
                                            -0.812999725f, 1.5959997177f };

__kernel void YUV2RGB_NVx(__global const uchar* srcptr, int src_step, int src_offset,
                            __global uchar* dstptr, int dst_step, int dt_offset,
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
                __global const uchar* ysrc = srcptr + mad24(y << 1, src_step, (x << 1) + src_offset);
                __global const uchar* usrc = srcptr + mad24(rows + y, src_step, (x << 1) + src_offset);
                __global uchar*       dst1 = dstptr + mad24(y << 1, dst_step, mad24(x, dcn<<1, dt_offset));
                __global uchar*       dst2 = dst1 + dst_step;

                float Y1 = ysrc[0];
                float Y2 = ysrc[1];
                float Y3 = ysrc[src_step];
                float Y4 = ysrc[src_step + 1];

                float U  = ((float)usrc[uidx]) - HALF_MAX_NUM;
                float V  = ((float)usrc[1-uidx]) - HALF_MAX_NUM;

                __constant float* coeffs = c_YUV2RGBCoeffs_420;
                float ruv = fma(coeffs[4], V, 0.5f);
                float guv = fma(coeffs[3], V, fma(coeffs[2], U, 0.5f));
                float buv = fma(coeffs[1], U, 0.5f);

                Y1 = max(0.f, Y1 - 16.f) * coeffs[0];
                dst1[2 - bidx] = convert_uchar_sat(Y1 + ruv);
                dst1[1]        = convert_uchar_sat(Y1 + guv);
                dst1[bidx]     = convert_uchar_sat(Y1 + buv);
#if dcn == 4
                dst1[3]        = 255;
#endif

                Y2 = max(0.f, Y2 - 16.f) * coeffs[0];
                dst1[dcn + 2 - bidx] = convert_uchar_sat(Y2 + ruv);
                dst1[dcn + 1]        = convert_uchar_sat(Y2 + guv);
                dst1[dcn + bidx]     = convert_uchar_sat(Y2 + buv);
#if dcn == 4
                dst1[7]        = 255;
#endif

                Y3 = max(0.f, Y3 - 16.f) * coeffs[0];
                dst2[2 - bidx] = convert_uchar_sat(Y3 + ruv);
                dst2[1]        = convert_uchar_sat(Y3 + guv);
                dst2[bidx]     = convert_uchar_sat(Y3 + buv);
#if dcn == 4
                dst2[3]        = 255;
#endif

                Y4 = max(0.f, Y4 - 16.f) * coeffs[0];
                dst2[dcn + 2 - bidx] = convert_uchar_sat(Y4 + ruv);
                dst2[dcn + 1]        = convert_uchar_sat(Y4 + guv);
                dst2[dcn + bidx]     = convert_uchar_sat(Y4 + buv);
#if dcn == 4
                dst2[7]        = 255;
#endif
            }
            ++y;
        }
    }
}

#if uidx < 2

__kernel void YUV2RGB_YV12_IYUV(__global const uchar* srcptr, int src_step, int src_offset,
                                __global uchar* dstptr, int dst_step, int dt_offset,
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
                __global const uchar* ysrc = srcptr + mad24(y << 1, src_step, (x << 1) + src_offset);
                __global uchar*       dst1 = dstptr + mad24(y << 1, dst_step, x * (dcn<<1) + dt_offset);
                __global uchar*       dst2 = dst1 + dst_step;

                float Y1 = ysrc[0];
                float Y2 = ysrc[1];
                float Y3 = ysrc[src_step];
                float Y4 = ysrc[src_step + 1];

#ifdef SRC_CONT
                __global const uchar* uvsrc = srcptr + mad24(rows, src_step, src_offset);
                int u_ind = mad24(y, cols >> 1, x);
                float uv[2] = { ((float)uvsrc[u_ind]) - HALF_MAX_NUM, ((float)uvsrc[u_ind + ((rows * cols) >> 2)]) - HALF_MAX_NUM };
#else
                int vsteps[2] = { cols >> 1, src_step - (cols >> 1)};
                __global const uchar* usrc = srcptr + mad24(rows + (y>>1), src_step, src_offset + (y%2)*(cols >> 1) + x);
                __global const uchar* vsrc = usrc + mad24(rows >> 2, src_step, rows % 4 ? vsteps[y%2] : 0);
                float uv[2] = { ((float)usrc[0]) - HALF_MAX_NUM, ((float)vsrc[0]) - HALF_MAX_NUM };
#endif
                float U = uv[uidx];
                float V = uv[1-uidx];

                __constant float* coeffs = c_YUV2RGBCoeffs_420;
                float ruv = fma(coeffs[4], V, 0.5f);
                float guv = fma(coeffs[3], V, fma(coeffs[2], U, 0.5f));
                float buv = fma(coeffs[1], U, 0.5f);

                Y1 = max(0.f, Y1 - 16.f) * coeffs[0];
                dst1[2 - bidx] = convert_uchar_sat(Y1 + ruv);
                dst1[1]        = convert_uchar_sat(Y1 + guv);
                dst1[bidx]     = convert_uchar_sat(Y1 + buv);
#if dcn == 4
                dst1[3]        = 255;
#endif

                Y2 = max(0.f, Y2 - 16.f) * coeffs[0];
                dst1[dcn + 2 - bidx] = convert_uchar_sat(Y2 + ruv);
                dst1[dcn + 1]        = convert_uchar_sat(Y2 + guv);
                dst1[dcn + bidx]     = convert_uchar_sat(Y2 + buv);
#if dcn == 4
                dst1[7]        = 255;
#endif

                Y3 = max(0.f, Y3 - 16.f) * coeffs[0];
                dst2[2 - bidx] = convert_uchar_sat(Y3 + ruv);
                dst2[1]        = convert_uchar_sat(Y3 + guv);
                dst2[bidx]     = convert_uchar_sat(Y3 + buv);
#if dcn == 4
                dst2[3]        = 255;
#endif

                Y4 = max(0.f, Y4 - 16.f) * coeffs[0];
                dst2[dcn + 2 - bidx] = convert_uchar_sat(Y4 + ruv);
                dst2[dcn + 1]        = convert_uchar_sat(Y4 + guv);
                dst2[dcn + bidx]     = convert_uchar_sat(Y4 + buv);
#if dcn == 4
                dst2[7]        = 255;
#endif
            }
            ++y;
        }
    }
}

#endif

#if uidx < 2

__constant float c_RGB2YUVCoeffs_420[8] = { 0.256999969f, 0.50399971f, 0.09799957f, -0.1479988098f, -0.2909994125f,
                                            0.438999176f, -0.3679990768f, -0.0709991455f };

__kernel void RGB2YUV_YV12_IYUV(__global const uchar* srcptr, int src_step, int src_offset,
                                __global uchar* dstptr, int dst_step, int dst_offset,
                                int rows, int cols)
{
    int x = get_global_id(0) * PIX_PER_WI_X;
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols/2)
    {
        int src_index  = mad24(y << 1, src_step, mad24(x << 1, scn, src_offset));
        int ydst_index = mad24(y << 1, dst_step, (x << 1) + dst_offset);
        int y_rows = rows / 3 * 2;
        int vsteps[2] = { cols >> 1, dst_step - (cols >> 1)};
        __constant float* coeffs = c_RGB2YUVCoeffs_420;

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows / 3)
            {
                __global const uchar* src1 = srcptr + src_index;
                __global const uchar* src2 = src1 + src_step;
                __global uchar* ydst1 = dstptr + ydst_index;
                __global uchar* ydst2 = ydst1 + dst_step;

                __global uchar* udst = dstptr + mad24(y_rows + (y>>1), dst_step, dst_offset + (y%2)*(cols >> 1) + x);
                __global uchar* vdst = udst + mad24(y_rows >> 2, dst_step, y_rows % 4 ? vsteps[y%2] : 0);

#if PIX_PER_WI_X == 2
                int s11 = *((__global const int*) src1);
                int s12 = *((__global const int*) src1 + 1);
                int s13 = *((__global const int*) src1 + 2);
#if scn == 4
                int s14 = *((__global const int*) src1 + 3);
#endif
                int s21 = *((__global const int*) src2);
                int s22 = *((__global const int*) src2 + 1);
                int s23 = *((__global const int*) src2 + 2);
#if scn == 4
                int s24 = *((__global const int*) src2 + 3);
#endif
                float src_pix1[scn * 4], src_pix2[scn * 4];

                *((float4*) src_pix1)     = convert_float4(as_uchar4(s11));
                *((float4*) src_pix1 + 1) = convert_float4(as_uchar4(s12));
                *((float4*) src_pix1 + 2) = convert_float4(as_uchar4(s13));
#if scn == 4
                *((float4*) src_pix1 + 3) = convert_float4(as_uchar4(s14));
#endif
                *((float4*) src_pix2)     = convert_float4(as_uchar4(s21));
                *((float4*) src_pix2 + 1) = convert_float4(as_uchar4(s22));
                *((float4*) src_pix2 + 2) = convert_float4(as_uchar4(s23));
#if scn == 4
                *((float4*) src_pix2 + 3) = convert_float4(as_uchar4(s24));
#endif
                uchar4 y1, y2;
                y1.x = convert_uchar_sat(fma(coeffs[0], src_pix1[      2-bidx], fma(coeffs[1], src_pix1[      1], fma(coeffs[2], src_pix1[      bidx], 16.5f))));
                y1.y = convert_uchar_sat(fma(coeffs[0], src_pix1[  scn+2-bidx], fma(coeffs[1], src_pix1[  scn+1], fma(coeffs[2], src_pix1[  scn+bidx], 16.5f))));
                y1.z = convert_uchar_sat(fma(coeffs[0], src_pix1[2*scn+2-bidx], fma(coeffs[1], src_pix1[2*scn+1], fma(coeffs[2], src_pix1[2*scn+bidx], 16.5f))));
                y1.w = convert_uchar_sat(fma(coeffs[0], src_pix1[3*scn+2-bidx], fma(coeffs[1], src_pix1[3*scn+1], fma(coeffs[2], src_pix1[3*scn+bidx], 16.5f))));
                y2.x = convert_uchar_sat(fma(coeffs[0], src_pix2[      2-bidx], fma(coeffs[1], src_pix2[      1], fma(coeffs[2], src_pix2[      bidx], 16.5f))));
                y2.y = convert_uchar_sat(fma(coeffs[0], src_pix2[  scn+2-bidx], fma(coeffs[1], src_pix2[  scn+1], fma(coeffs[2], src_pix2[  scn+bidx], 16.5f))));
                y2.z = convert_uchar_sat(fma(coeffs[0], src_pix2[2*scn+2-bidx], fma(coeffs[1], src_pix2[2*scn+1], fma(coeffs[2], src_pix2[2*scn+bidx], 16.5f))));
                y2.w = convert_uchar_sat(fma(coeffs[0], src_pix2[3*scn+2-bidx], fma(coeffs[1], src_pix2[3*scn+1], fma(coeffs[2], src_pix2[3*scn+bidx], 16.5f))));

                *((__global int*) ydst1) = as_int(y1);
                *((__global int*) ydst2) = as_int(y2);

                float uv[4] = { fma(coeffs[3], src_pix1[      2-bidx], fma(coeffs[4], src_pix1[      1], fma(coeffs[5], src_pix1[      bidx], 128.5f))),
                                fma(coeffs[5], src_pix1[      2-bidx], fma(coeffs[6], src_pix1[      1], fma(coeffs[7], src_pix1[      bidx], 128.5f))),
                                fma(coeffs[3], src_pix1[2*scn+2-bidx], fma(coeffs[4], src_pix1[2*scn+1], fma(coeffs[5], src_pix1[2*scn+bidx], 128.5f))),
                                fma(coeffs[5], src_pix1[2*scn+2-bidx], fma(coeffs[6], src_pix1[2*scn+1], fma(coeffs[7], src_pix1[2*scn+bidx], 128.5f))) };

                udst[0] = convert_uchar_sat(uv[uidx]    );
                vdst[0] = convert_uchar_sat(uv[1 - uidx]);
                udst[1] = convert_uchar_sat(uv[2 + uidx]);
                vdst[1] = convert_uchar_sat(uv[3 - uidx]);
#else
                float4 src_pix1 = convert_float4(vload4(0, src1));
                float4 src_pix2 = convert_float4(vload4(0, src1+scn));
                float4 src_pix3 = convert_float4(vload4(0, src2));
                float4 src_pix4 = convert_float4(vload4(0, src2+scn));

                ydst1[0] = convert_uchar_sat(fma(coeffs[0], src_pix1.R_COMP, fma(coeffs[1], src_pix1.G_COMP, fma(coeffs[2], src_pix1.B_COMP, 16.5f))));
                ydst1[1] = convert_uchar_sat(fma(coeffs[0], src_pix2.R_COMP, fma(coeffs[1], src_pix2.G_COMP, fma(coeffs[2], src_pix2.B_COMP, 16.5f))));
                ydst2[0] = convert_uchar_sat(fma(coeffs[0], src_pix3.R_COMP, fma(coeffs[1], src_pix3.G_COMP, fma(coeffs[2], src_pix3.B_COMP, 16.5f))));
                ydst2[1] = convert_uchar_sat(fma(coeffs[0], src_pix4.R_COMP, fma(coeffs[1], src_pix4.G_COMP, fma(coeffs[2], src_pix4.B_COMP, 16.5f))));

                float uv[2] = { fma(coeffs[3], src_pix1.R_COMP, fma(coeffs[4], src_pix1.G_COMP, fma(coeffs[5], src_pix1.B_COMP, 128.5f))),
                                fma(coeffs[5], src_pix1.R_COMP, fma(coeffs[6], src_pix1.G_COMP, fma(coeffs[7], src_pix1.B_COMP, 128.5f))) };

                udst[0] = convert_uchar_sat(uv[uidx]  );
                vdst[0] = convert_uchar_sat(uv[1-uidx]);
#endif
                ++y;
                src_index += 2*src_step;
                ydst_index += 2*dst_step;
            }
        }
    }
}

#endif

__kernel void YUV2RGB_422(__global const uchar* srcptr, int src_step, int src_offset,
                          __global uchar* dstptr, int dst_step, int dst_offset,
                          int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols / 2)
    {
        __global const uchar* src = srcptr + mad24(y, src_step, (x << 2) + src_offset);
        __global uchar*       dst = dstptr + mad24(y, dst_step, mad24(x << 1, dcn, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows )
            {
                __constant float* coeffs = c_YUV2RGBCoeffs_420;

#ifndef USE_OPTIMIZED_LOAD
                float U = ((float) src[uidx]) - HALF_MAX_NUM;
                float V = ((float) src[(2 + uidx) % 4]) - HALF_MAX_NUM;
                float y00 = max(0.f, ((float) src[yidx]) - 16.f) * coeffs[0];
                float y01 = max(0.f, ((float) src[yidx + 2]) - 16.f) * coeffs[0];
#else
                int load_src = *((__global int*) src);
                float vec_src[4] = { load_src & 0xff, (load_src >> 8) & 0xff, (load_src >> 16) & 0xff, (load_src >> 24) & 0xff};
                float U = vec_src[uidx] - HALF_MAX_NUM;
                float V = vec_src[(2 + uidx) % 4] - HALF_MAX_NUM;
                float y00 = max(0.f, vec_src[yidx] - 16.f) * coeffs[0];
                float y01 = max(0.f, vec_src[yidx + 2] - 16.f) * coeffs[0];
#endif

                float ruv = fma(coeffs[4], V, 0.5f);
                float guv = fma(coeffs[3], V, fma(coeffs[2], U, 0.5f));
                float buv = fma(coeffs[1], U, 0.5f);

                dst[2 - bidx] = convert_uchar_sat(y00 + ruv);
                dst[1]        = convert_uchar_sat(y00 + guv);
                dst[bidx]     = convert_uchar_sat(y00 + buv);
#if dcn == 4
                dst[3]        = 255;
#endif

                dst[dcn + 2 - bidx] = convert_uchar_sat(y01 + ruv);
                dst[dcn + 1]        = convert_uchar_sat(y01 + guv);
                dst[dcn + bidx]     = convert_uchar_sat(y01 + buv);
#if dcn == 4
                dst[7]        = 255;
#endif
            }
            ++y;
            src += src_step;
            dst += dst_step;
        }
    }
}

///////////////////////////////////// RGB <-> YCrCb //////////////////////////////////////

__constant float c_RGB2YCrCbCoeffs_f[5] = {R2YF, G2YF, B2YF, YCRF, YCBF};
__constant int   c_RGB2YCrCbCoeffs_i[5] = {R2Y, G2Y, B2Y, YCRI, YCBI};

__kernel void RGB2YCrCb(__global const uchar* srcptr, int src_step, int src_offset,
                        __global uchar* dstptr, int dst_step, int dt_offset,
                        int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dt_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + src_index);
                __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + dst_index);
                DATA_TYPE_4 src_pix = vload4(0, src);
                DATA_TYPE b = src_pix.B_COMP, g = src_pix.G_COMP, r = src_pix.R_COMP;

#ifdef DEPTH_5
                __constant float * coeffs = c_RGB2YCrCbCoeffs_f;
                DATA_TYPE Y = fma(b, coeffs[2], fma(g, coeffs[1], r * coeffs[0]));
                DATA_TYPE Cr = fma(r - Y, coeffs[3], HALF_MAX_NUM);
                DATA_TYPE Cb = fma(b - Y, coeffs[4], HALF_MAX_NUM);
#else
                __constant int * coeffs = c_RGB2YCrCbCoeffs_i;
                int delta = HALF_MAX_NUM * (1 << yuv_shift);
                int Y =  CV_DESCALE(mad24(b, coeffs[2], mad24(g, coeffs[1], mul24(r, coeffs[0]))), yuv_shift);
                int Cr = CV_DESCALE(mad24(r - Y, coeffs[3], delta), yuv_shift);
                int Cb = CV_DESCALE(mad24(b - Y, coeffs[4], delta), yuv_shift);
#endif

                dst[0] = SAT_CAST( Y );
                dst[1] = SAT_CAST( Cr );
                dst[2] = SAT_CAST( Cb );

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}

__constant float c_YCrCb2RGBCoeffs_f[4] = { CR2RF, CR2GF, CB2GF, CB2BF };
__constant int   c_YCrCb2RGBCoeffs_i[4] = { CR2RI, CR2GI, CB2GI, CB2BI };

__kernel void YCrCb2RGB(__global const uchar* src, int src_step, int src_offset,
                        __global uchar* dst, int dst_step, int dst_offset,
                        int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE * srcptr = (__global const DATA_TYPE*)(src + src_index);
                __global DATA_TYPE * dstptr = (__global DATA_TYPE*)(dst + dst_index);

                DATA_TYPE_4 src_pix = vload4(0, srcptr);
                DATA_TYPE yp = src_pix.x, cr = src_pix.y, cb = src_pix.z;

#ifdef DEPTH_5
                __constant float * coeff = c_YCrCb2RGBCoeffs_f;
                float r = fma(coeff[0], cr - HALF_MAX_NUM, yp);
                float g = fma(coeff[1], cr - HALF_MAX_NUM, fma(coeff[2], cb - HALF_MAX_NUM, yp));
                float b = fma(coeff[3], cb - HALF_MAX_NUM, yp);
#else
                __constant int * coeff = c_YCrCb2RGBCoeffs_i;
                int r = yp + CV_DESCALE(coeff[0] * (cr - HALF_MAX_NUM), yuv_shift);
                int g = yp + CV_DESCALE(mad24(coeff[1], cr - HALF_MAX_NUM, coeff[2] * (cb - HALF_MAX_NUM)), yuv_shift);
                int b = yp + CV_DESCALE(coeff[3] * (cb - HALF_MAX_NUM), yuv_shift);
#endif

                dstptr[(bidx^2)] = SAT_CAST(r);
                dstptr[1] = SAT_CAST(g);
                dstptr[bidx] = SAT_CAST(b);
#if dcn == 4
                dstptr[3] = MAX_NUM;
#endif

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(dy, src_step, mad24(dx, scnbytes, src_offset));
        int dst_index = mad24(dy, dst_step, mad24(dx, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (dy < rows)
            {
                __global const DATA_TYPE * src = (__global const DATA_TYPE *)(srcptr + src_index);
                __global DATA_TYPE * dst = (__global DATA_TYPE *)(dstptr + dst_index);

                DATA_TYPE_4 src_pix = vload4(0, src);
                DATA_TYPE r = src_pix.x, g = src_pix.y, b = src_pix.z;

#ifdef DEPTH_5
                float x = fma(r, coeffs[0], fma(g, coeffs[1], b * coeffs[2]));
                float y = fma(r, coeffs[3], fma(g, coeffs[4], b * coeffs[5]));
                float z = fma(r, coeffs[6], fma(g, coeffs[7], b * coeffs[8]));
#else
                int x = CV_DESCALE(mad24(r, coeffs[0], mad24(g, coeffs[1], b * coeffs[2])), xyz_shift);
                int y = CV_DESCALE(mad24(r, coeffs[3], mad24(g, coeffs[4], b * coeffs[5])), xyz_shift);
                int z = CV_DESCALE(mad24(r, coeffs[6], mad24(g, coeffs[7], b * coeffs[8])), xyz_shift);
#endif
                dst[0] = SAT_CAST(x);
                dst[1] = SAT_CAST(y);
                dst[2] = SAT_CAST(z);

                ++dy;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(dy, src_step, mad24(dx, scnbytes, src_offset));
        int dst_index = mad24(dy, dst_step, mad24(dx, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (dy < rows)
            {
                __global const DATA_TYPE * src = (__global const DATA_TYPE *)(srcptr + src_index);
                __global DATA_TYPE * dst = (__global DATA_TYPE *)(dstptr + dst_index);

                DATA_TYPE_4 src_pix = vload4(0, src);
                DATA_TYPE x = src_pix.x, y = src_pix.y, z = src_pix.z;

#ifdef DEPTH_5
                float b = fma(x, coeffs[0], fma(y, coeffs[1], z * coeffs[2]));
                float g = fma(x, coeffs[3], fma(y, coeffs[4], z * coeffs[5]));
                float r = fma(x, coeffs[6], fma(y, coeffs[7], z * coeffs[8]));
#else
                int b = CV_DESCALE(mad24(x, coeffs[0], mad24(y, coeffs[1], z * coeffs[2])), xyz_shift);
                int g = CV_DESCALE(mad24(x, coeffs[3], mad24(y, coeffs[4], z * coeffs[5])), xyz_shift);
                int r = CV_DESCALE(mad24(x, coeffs[6], mad24(y, coeffs[7], z * coeffs[8])), xyz_shift);
#endif

                DATA_TYPE dst0 = SAT_CAST(b);
                DATA_TYPE dst1 = SAT_CAST(g);
                DATA_TYPE dst2 = SAT_CAST(r);
#if dcn == 3 || defined DEPTH_5
                dst[0] = dst0;
                dst[1] = dst1;
                dst[2] = dst2;
#if dcn == 4
                dst[3] = MAX_NUM;
#endif
#else
                *(__global DATA_TYPE_4 *)dst = (DATA_TYPE_4)(dst0, dst1, dst2, MAX_NUM);
#endif

                ++dy;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE * src = (__global const DATA_TYPE *)(srcptr + src_index);
                __global DATA_TYPE * dst = (__global DATA_TYPE *)(dstptr + dst_index);
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

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                ushort t = *((__global const ushort*)(src + src_index));

#if greenbits == 6
                dst[dst_index + bidx] = (uchar)(t << 3);
                dst[dst_index + 1] = (uchar)((t >> 3) & ~3);
                dst[dst_index + (bidx^2)] = (uchar)((t >> 8) & ~7);
#else
                dst[dst_index + bidx] = (uchar)(t << 3);
                dst[dst_index + 1] = (uchar)((t >> 2) & ~7);
                dst[dst_index + (bidx^2)] = (uchar)((t >> 7) & ~7);
#endif

#if dcn == 4
#if greenbits == 6
                dst[dst_index + 3] = 255;
#else
                dst[dst_index + 3] = t & 0x8000 ? 255 : 0;
#endif
#endif

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                uchar4 src_pix = vload4(0, src + src_index);

#if greenbits == 6
                    *((__global ushort*)(dst + dst_index)) = (ushort)((src_pix.B_COMP >> 3)|((src_pix.G_COMP&~3) << 3)|((src_pix.R_COMP&~7) << 8));
#elif scn == 3
                    *((__global ushort*)(dst + dst_index)) = (ushort)((src_pix.B_COMP >> 3)|((src_pix.G_COMP&~7) << 2)|((src_pix.R_COMP&~7) << 7));
#else
                    *((__global ushort*)(dst + dst_index)) = (ushort)((src_pix.B_COMP >> 3)|((src_pix.G_COMP&~7) << 2)|
                        ((src_pix.R_COMP&~7) << 7)|(src_pix.w ? 0x8000 : 0));
#endif

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, dst_offset + x);

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int t = *((__global const ushort*)(src + src_index));

#if greenbits == 6
                dst[dst_index] = (uchar)CV_DESCALE(mad24((t << 3) & 0xf8, B2Y, mad24((t >> 3) & 0xfc, G2Y, ((t >> 8) & 0xf8) * R2Y)), yuv_shift);
#else
                dst[dst_index] = (uchar)CV_DESCALE(mad24((t << 3) & 0xf8, B2Y, mad24((t >> 2) & 0xf8, G2Y, ((t >> 7) & 0xf8) * R2Y)), yuv_shift);
#endif
                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, src_offset + x);
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                int t = src[src_index];

#if greenbits == 6
                *((__global ushort*)(dst + dst_index)) = (ushort)((t >> 3) | ((t & ~3) << 3) | ((t & ~7) << 8));
#else
                t >>= 3;
                *((__global ushort*)(dst + dst_index)) = (ushort)(t|(t << 5)|(t << 10));
#endif
                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                uchar4 src_pix = vload4(0, src + src_index);

                int b = src_pix.B_COMP, g = src_pix.G_COMP, r = src_pix.R_COMP;
                int h, s, v = b;
                int vmin = b, diff;
                int vr, vg;

                v = max(v, g);
                v = max(v, r);
                vmin = min(vmin, g);
                vmin = min(vmin, r);

                diff = v - vmin;
                vr = v == r ? -1 : 0;
                vg = v == g ? -1 : 0;

                s = mad24(diff, sdiv_table[v], (1 << (hsv_shift-1))) >> hsv_shift;
                h = (vr & (g - b)) +
                    (~vr & ((vg & mad24(diff, 2, b - r)) + ((~vg) & mad24(4, diff, r - g))));
                h = mad24(h, hdiv_table[diff], (1 << (hsv_shift-1))) >> hsv_shift;
                h += h < 0 ? hrange : 0;

                dst[dst_index] = convert_uchar_sat_rte(h);
                dst[dst_index + 1] = (uchar)s;
                dst[dst_index + 2] = (uchar)v;

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                uchar4 src_pix = vload4(0, src + src_index);

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

                dst[dst_index + bidx] = convert_uchar_sat_rte(b*255.f);
                dst[dst_index + 1] = convert_uchar_sat_rte(g*255.f);
                dst[dst_index + (bidx^2)] = convert_uchar_sat_rte(r*255.f);
#if dcn == 4
                dst[dst_index + 3] = MAX_NUM;
#endif

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);
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
                    h = fma(b - r, diff, 120.f);
                else
                    h = fma(r - g, diff, 240.f);

                if( h < 0 )
                    h += 360.f;

                dst[0] = h*hscale;
                dst[1] = s;
                dst[2] = v;

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {

                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);
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

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                uchar4 src_pix = vload4(0, src + src_index);

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
                        h = fma(b - r, diff, 120.f);
                    else
                        h = fma(r - g, diff, 240.f);

                    if( h < 0.f )
                        h += 360.f;
                }

                dst[dst_index] = convert_uchar_sat_rte(h*hscale);
                dst[dst_index + 1] = convert_uchar_sat_rte(l*255.f);
                dst[dst_index + 2] = convert_uchar_sat_rte(s*255.f);

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                uchar4 src_pix = vload4(0, src + src_index);

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
                    tab[2] = fma(p2 - p1, 1-h, p1);
                    tab[3] = fma(p2 - p1, h, p1);

                    b = tab[sector_data[sector][0]];
                    g = tab[sector_data[sector][1]];
                    r = tab[sector_data[sector][2]];
                }
                else
                    b = g = r = l;

                dst[dst_index + bidx] = convert_uchar_sat_rte(b*255.f);
                dst[dst_index + 1] = convert_uchar_sat_rte(g*255.f);
                dst[dst_index + (bidx^2)] = convert_uchar_sat_rte(r*255.f);
#if dcn == 4
                dst[dst_index + 3] = MAX_NUM;
#endif

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);
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
                        h = fma(b - r, diff, 120.f);
                    else
                        h = fma(r - g, diff, 240.f);

                    if( h < 0.f ) h += 360.f;
                }

                dst[0] = h*hscale;
                dst[1] = l;
                dst[2] = s;

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);
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
                    tab[2] = fma(p2 - p1, 1-h, p1);
                    tab[3] = fma(p2 - p1, h, p1);

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

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, src_offset + (x << 2));
        int dst_index = mad24(y, dst_step, dst_offset + (x << 2));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                uchar4 src_pix = *(__global const uchar4 *)(src + src_index);

                *(__global uchar4 *)(dst + dst_index) =
                    (uchar4)(mad24(src_pix.x, src_pix.w, HALF_MAX_NUM) / MAX_NUM,
                             mad24(src_pix.y, src_pix.w, HALF_MAX_NUM) / MAX_NUM,
                             mad24(src_pix.z, src_pix.w, HALF_MAX_NUM) / MAX_NUM, src_pix.w);

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, 4, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, 4, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                uchar4 src_pix = *(__global const uchar4 *)(src + src_index);
                uchar v3 = src_pix.w, v3_half = v3 / 2;

                if (v3 == 0)
                    *(__global uchar4 *)(dst + dst_index) = (uchar4)(0, 0, 0, 0);
                else
                    *(__global uchar4 *)(dst + dst_index) =
                        (uchar4)(mad24(src_pix.x, MAX_NUM, v3_half) / v3,
                                 mad24(src_pix.y, MAX_NUM, v3_half) / v3,
                                 mad24(src_pix.z, MAX_NUM, v3_half) / v3, v3);

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
    tab += ix << 2;
    return fma(fma(fma(tab[3], x, tab[2]), x, tab[1]), x, tab[0]);
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const uchar* src_ptr = src + src_index;
                __global uchar* dst_ptr = dst + dst_index;
                uchar4 src_pix = vload4(0, src_ptr);

                int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
                    C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
                    C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

                int R = gammaTab[src_pix.x], G = gammaTab[src_pix.y], B = gammaTab[src_pix.z];
                int fX = LabCbrtTab_b[CV_DESCALE(mad24(R, C0, mad24(G, C1, B*C2)), lab_shift)];
                int fY = LabCbrtTab_b[CV_DESCALE(mad24(R, C3, mad24(G, C4, B*C5)), lab_shift)];
                int fZ = LabCbrtTab_b[CV_DESCALE(mad24(R, C6, mad24(G, C7, B*C8)), lab_shift)];

                int L = CV_DESCALE( Lscale*fY + Lshift, lab_shift2 );
                int a = CV_DESCALE( mad24(500, fX - fY, 128*(1 << lab_shift2)), lab_shift2 );
                int b = CV_DESCALE( mad24(200, fY - fZ, 128*(1 << lab_shift2)), lab_shift2 );

                dst_ptr[0] = SAT_CAST(L);
                dst_ptr[1] = SAT_CAST(a);
                dst_ptr[2] = SAT_CAST(b);

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);
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

                // 7.787f = (29/3)^3/(29*4), 0.008856f = (6/29)^3, 903.3 = (29/3)^3
                float X = fma(R, C0, fma(G, C1, B*C2));
                float Y = fma(R, C3, fma(G, C4, B*C5));
                float Z = fma(R, C6, fma(G, C7, B*C8));

                float FX = X > 0.008856f ? rootn(X, 3) : fma(7.787f, X, _a);
                float FY = Y > 0.008856f ? rootn(Y, 3) : fma(7.787f, Y, _a);
                float FZ = Z > 0.008856f ? rootn(Z, 3) : fma(7.787f, Z, _a);

                float L = Y > 0.008856f ? fma(116.f, FY, -16.f) : (903.3f * Y);
                float a = 500.f * (FX - FY);
                float b = 200.f * (FY - FZ);

                dst[0] = L;
                dst[1] = a;
                dst[2] = b;

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
    // 903.3 = (29/3)^3, 7.787 = (29/3)^3/(29*4)
    if (li <= lThresh)
    {
        y = li / 903.3f;
        fy = fma(7.787f, y, 16.0f / 116.0f);
    }
    else
    {
        fy = (li + 16.0f) / 116.0f;
        y = fy * fy * fy;
    }

    float fxz[] = { ai / 500.0f + fy, fy - bi / 200.0f };

    #pragma unroll
    for (int j = 0; j < 2; j++)
        if (fxz[j] <= fThresh)
            fxz[j] = (fxz[j] - 16.0f / 116.0f) / 7.787f;
        else
            fxz[j] = fxz[j] * fxz[j] * fxz[j];

    float x = fxz[0], z = fxz[1];
    float ro = clamp(fma(C0, x, fma(C1, y, C2 * z)), 0.0f, 1.0f);
    float go = clamp(fma(C3, x, fma(C4, y, C5 * z)), 0.0f, 1.0f);
    float bo = clamp(fma(C6, x, fma(C7, y, C8 * z)), 0.0f, 1.0f);

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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const uchar* src_ptr = src + src_index;
                __global uchar * dst_ptr = dst + dst_index;
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

#if dcn == 3
                dst_ptr[0] = SAT_CAST(dstbuf[0] * 255.0f);
                dst_ptr[1] = SAT_CAST(dstbuf[1] * 255.0f);
                dst_ptr[2] = SAT_CAST(dstbuf[2] * 255.0f);
#else
                *(__global uchar4 *)dst_ptr = (uchar4)(SAT_CAST(dstbuf[0] * 255.0f),
                    SAT_CAST(dstbuf[1] * 255.0f), SAT_CAST(dstbuf[2] * 255.0f), MAX_NUM);
#endif
                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);
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
                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
            if (y < rows)
            {
                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);

                float R = src[0], G = src[1], B = src[2];

                R = clamp(R, 0.f, 1.f);
                G = clamp(G, 0.f, 1.f);
                B = clamp(B, 0.f, 1.f);

#ifdef SRGB
                R = splineInterpolate(R*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif
                float X = fma(R, coeffs[0], fma(G, coeffs[1], B*coeffs[2]));
                float Y = fma(R, coeffs[3], fma(G, coeffs[4], B*coeffs[5]));
                float Z = fma(R, coeffs[6], fma(G, coeffs[7], B*coeffs[8]));

                float L = splineInterpolate(Y*LabCbrtTabScale, LabCbrtTab, LAB_CBRT_TAB_SIZE);
                L = fma(116.f, L, -16.f);

                float d = 52.0f / fmax(fma(15.0f, Y, fma(3.0f, Z, X)), FLT_EPSILON);
                float u = L*fma(X, d, -_un);
                float v = L*fma(2.25f, Y*d, -_vn);

                dst[0] = L;
                dst[1] = u;
                dst[2] = v;

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        src += mad24(y, src_step, mad24(x, scnbytes, src_offset));
        dst += mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
            if (y < rows)
            {
                float scale = 1.0f / 255.0f;
                float R = src[0]*scale, G = src[1]*scale, B = src[2]*scale;

#ifdef SRGB
                R = splineInterpolate(R*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif
                float X = fma(R, coeffs[0], fma(G, coeffs[1], B*coeffs[2]));
                float Y = fma(R, coeffs[3], fma(G, coeffs[4], B*coeffs[5]));
                float Z = fma(R, coeffs[6], fma(G, coeffs[7], B*coeffs[8]));

                float L = splineInterpolate(Y*LabCbrtTabScale, LabCbrtTab, LAB_CBRT_TAB_SIZE);
                L = 116.f*L - 16.f;

                float d = (4*13) / fmax(fma(15.0f, Y, fma(3.0f, Z, X)), FLT_EPSILON);
                float u = L*(X*d - _un);
                float v = L*fma(2.25f, Y*d, -_vn);

                dst[0] = SAT_CAST(L * 2.55f);
                //0.72033 = 255/(220+134), 96.525 = 134*255/(220+134)
                dst[1] = SAT_CAST(fma(u, 0.72033898305084743f, 96.525423728813564f));
                //0.9732 = 255/(140+122), 136.259 = 140*255/(140+122)
                dst[2] = SAT_CAST(fma(v, 0.9732824427480916f, 136.259541984732824f));

                ++y;
                dst += dst_step;
                src += src_step;
            }
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
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        int src_index = mad24(y, src_step, mad24(x, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
            if (y < rows)
            {
                __global const float * src = (__global const float *)(srcptr + src_index);
                __global float * dst = (__global float *)(dstptr + dst_index);

                float L = src[0], u = src[1], v = src[2], X, Y, Z;
                if(L >= 8)
                {
                    Y = fma(L, 1.f/116.f, 16.f/116.f);
                    Y = Y*Y*Y;
                }
                else
                {
                    Y = L * (1.0f/903.3f); // L*(3./29.)^3
                }
                float up = 3.f*fma(L, _un, u);
                float vp = 0.25f/fma(L, _vn, v);
                vp = clamp(vp, -0.25f, 0.25f);
                X = 3.f*Y*up*vp;
                Z = Y*fma(fma(12.f*13.f, L, -up), vp, -5.f);

                float R = fma(X, coeffs[0], fma(Y, coeffs[1], Z * coeffs[2]));
                float G = fma(X, coeffs[3], fma(Y, coeffs[4], Z * coeffs[5]));
                float B = fma(X, coeffs[6], fma(Y, coeffs[7], Z * coeffs[8]));

                R = clamp(R, 0.f, 1.f);
                G = clamp(G, 0.f, 1.f);
                B = clamp(B, 0.f, 1.f);

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
                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols)
    {
        src += mad24(y, src_step, mad24(x, scnbytes, src_offset));
        dst += mad24(y, dst_step, mad24(x, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
            if (y < rows)
            {
                float d, X, Y, Z;
                float L = src[0]*(100.f/255.f);
                // 1.388235294117647 = (220+134)/255
                float u = fma(convert_float(src[1]), 1.388235294117647f, -134.f);
                // 1.027450980392157 = (140+122)/255
                float v = fma(convert_float(src[2]), 1.027450980392157f, - 140.f);
                if(L >= 8)
                {
                    Y = fma(L, 1.f/116.f, 16.f/116.f);
                    Y = Y*Y*Y;
                }
                else
                {
                    Y = L * (1.0f/903.3f); // L*(3./29.)^3
                }
                float up = 3.f*fma(L, _un, u);
                float vp = 0.25f/fma(L, _vn, v);
                vp = clamp(vp, -0.25f, 0.25f);
                X = 3.f*Y*up*vp;
                Z = Y*fma(fma(12.f*13.f, L, -up), vp, -5.f);

                float R = fma(X, coeffs[0], fma(Y, coeffs[1], Z * coeffs[2]));
                float G = fma(X, coeffs[3], fma(Y, coeffs[4], Z * coeffs[5]));
                float B = fma(X, coeffs[6], fma(Y, coeffs[7], Z * coeffs[8]));

                R = clamp(R, 0.f, 1.f);
                G = clamp(G, 0.f, 1.f);
                B = clamp(B, 0.f, 1.f);

#ifdef SRGB
                R = splineInterpolate(R*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                G = splineInterpolate(G*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
                B = splineInterpolate(B*GammaTabScale, gammaTab, GAMMA_TAB_SIZE);
#endif

                uchar dst0 = SAT_CAST(R * 255.0f);
                uchar dst1 = SAT_CAST(G * 255.0f);
                uchar dst2 = SAT_CAST(B * 255.0f);

#if dcn == 4
                *(__global uchar4 *)dst = (uchar4)(dst0, dst1, dst2, MAX_NUM);
#else
                dst[0] = dst0;
                dst[1] = dst1;
                dst[2] = dst2;
#endif

                ++y;
                dst += dst_step;
                src += src_step;
            }
    }
}

#endif
