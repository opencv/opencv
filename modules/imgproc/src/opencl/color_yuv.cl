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

#if SRC_DEPTH == 0
    #define DATA_TYPE uchar
    #define MAX_NUM  255
    #define HALF_MAX_NUM 128
    #define COEFF_TYPE int
    #define SAT_CAST(num) convert_uchar_sat(num)
    #define DEPTH_0
#elif SRC_DEPTH == 2
    #define DATA_TYPE ushort
    #define MAX_NUM  65535
    #define HALF_MAX_NUM 32768
    #define COEFF_TYPE int
    #define SAT_CAST(num) convert_ushort_sat(num)
    #define DEPTH_2
#elif SRC_DEPTH == 5
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
    R2Y        = 4899,
    G2Y        = 9617,
    B2Y        = 1868,
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

#define scnbytes ((int)sizeof(DATA_TYPE)*SCN)
#define dcnbytes ((int)sizeof(DATA_TYPE)*DCN)

#if BIDX == 0
#define R_COMP z
#define G_COMP y
#define B_COMP x
#else
#define R_COMP x
#define G_COMP y
#define B_COMP z
#endif

#ifndef UIDX
#define UIDX 0
#endif

#ifndef YIDX
#define YIDX 0
#endif

#ifndef PIX_PER_WI_X
#define PIX_PER_WI_X 1
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define DATA_TYPE_4 CAT(DATA_TYPE, 4)
#define DATA_TYPE_3 CAT(DATA_TYPE, 3)

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

                dst[BIDX] = SAT_CAST( b );
                dst[1] = SAT_CAST( g );
                dst[BIDX^2] = SAT_CAST( r );
#if DCN == 4
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
                __global uchar*       dst1 = dstptr + mad24(y << 1, dst_step, mad24(x, DCN<<1, dt_offset));
                __global uchar*       dst2 = dst1 + dst_step;

                float Y1 = ysrc[0];
                float Y2 = ysrc[1];
                float Y3 = ysrc[src_step];
                float Y4 = ysrc[src_step + 1];

                float U  = ((float)usrc[UIDX]) - HALF_MAX_NUM;
                float V  = ((float)usrc[1-UIDX]) - HALF_MAX_NUM;

                __constant float* coeffs = c_YUV2RGBCoeffs_420;
                float ruv = fma(coeffs[4], V, 0.5f);
                float guv = fma(coeffs[3], V, fma(coeffs[2], U, 0.5f));
                float buv = fma(coeffs[1], U, 0.5f);

                Y1 = max(0.f, Y1 - 16.f) * coeffs[0];
                dst1[2 - BIDX] = convert_uchar_sat(Y1 + ruv);
                dst1[1]        = convert_uchar_sat(Y1 + guv);
                dst1[BIDX]     = convert_uchar_sat(Y1 + buv);
#if DCN == 4
                dst1[3]        = 255;
#endif

                Y2 = max(0.f, Y2 - 16.f) * coeffs[0];
                dst1[DCN + 2 - BIDX] = convert_uchar_sat(Y2 + ruv);
                dst1[DCN + 1]        = convert_uchar_sat(Y2 + guv);
                dst1[DCN + BIDX]     = convert_uchar_sat(Y2 + buv);
#if DCN == 4
                dst1[7]        = 255;
#endif

                Y3 = max(0.f, Y3 - 16.f) * coeffs[0];
                dst2[2 - BIDX] = convert_uchar_sat(Y3 + ruv);
                dst2[1]        = convert_uchar_sat(Y3 + guv);
                dst2[BIDX]     = convert_uchar_sat(Y3 + buv);
#if DCN == 4
                dst2[3]        = 255;
#endif

                Y4 = max(0.f, Y4 - 16.f) * coeffs[0];
                dst2[DCN + 2 - BIDX] = convert_uchar_sat(Y4 + ruv);
                dst2[DCN + 1]        = convert_uchar_sat(Y4 + guv);
                dst2[DCN + BIDX]     = convert_uchar_sat(Y4 + buv);
#if DCN == 4
                dst2[7]        = 255;
#endif
            }
            ++y;
        }
    }
}

#if UIDX < 2

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
                __global uchar*       dst1 = dstptr + mad24(y << 1, dst_step, x * (DCN<<1) + dt_offset);
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
                float U = uv[UIDX];
                float V = uv[1-UIDX];

                __constant float* coeffs = c_YUV2RGBCoeffs_420;
                float ruv = fma(coeffs[4], V, 0.5f);
                float guv = fma(coeffs[3], V, fma(coeffs[2], U, 0.5f));
                float buv = fma(coeffs[1], U, 0.5f);

                Y1 = max(0.f, Y1 - 16.f) * coeffs[0];
                dst1[2 - BIDX] = convert_uchar_sat(Y1 + ruv);
                dst1[1]        = convert_uchar_sat(Y1 + guv);
                dst1[BIDX]     = convert_uchar_sat(Y1 + buv);
#if DCN == 4
                dst1[3]        = 255;
#endif

                Y2 = max(0.f, Y2 - 16.f) * coeffs[0];
                dst1[DCN + 2 - BIDX] = convert_uchar_sat(Y2 + ruv);
                dst1[DCN + 1]        = convert_uchar_sat(Y2 + guv);
                dst1[DCN + BIDX]     = convert_uchar_sat(Y2 + buv);
#if DCN == 4
                dst1[7]        = 255;
#endif

                Y3 = max(0.f, Y3 - 16.f) * coeffs[0];
                dst2[2 - BIDX] = convert_uchar_sat(Y3 + ruv);
                dst2[1]        = convert_uchar_sat(Y3 + guv);
                dst2[BIDX]     = convert_uchar_sat(Y3 + buv);
#if DCN == 4
                dst2[3]        = 255;
#endif

                Y4 = max(0.f, Y4 - 16.f) * coeffs[0];
                dst2[DCN + 2 - BIDX] = convert_uchar_sat(Y4 + ruv);
                dst2[DCN + 1]        = convert_uchar_sat(Y4 + guv);
                dst2[DCN + BIDX]     = convert_uchar_sat(Y4 + buv);
#if DCN == 4
                dst2[7]        = 255;
#endif
            }
            ++y;
        }
    }
}

#endif

#if UIDX < 2

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
        int src_index  = mad24(y << 1, src_step, mad24(x << 1, SCN, src_offset));
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
#if SCN == 4
                int s14 = *((__global const int*) src1 + 3);
#endif
                int s21 = *((__global const int*) src2);
                int s22 = *((__global const int*) src2 + 1);
                int s23 = *((__global const int*) src2 + 2);
#if SCN == 4
                int s24 = *((__global const int*) src2 + 3);
#endif
                float src_pix1[SCN * 4], src_pix2[SCN * 4];

                *((float4*) src_pix1)     = convert_float4(as_uchar4(s11));
                *((float4*) src_pix1 + 1) = convert_float4(as_uchar4(s12));
                *((float4*) src_pix1 + 2) = convert_float4(as_uchar4(s13));
#if SCN == 4
                *((float4*) src_pix1 + 3) = convert_float4(as_uchar4(s14));
#endif
                *((float4*) src_pix2)     = convert_float4(as_uchar4(s21));
                *((float4*) src_pix2 + 1) = convert_float4(as_uchar4(s22));
                *((float4*) src_pix2 + 2) = convert_float4(as_uchar4(s23));
#if SCN == 4
                *((float4*) src_pix2 + 3) = convert_float4(as_uchar4(s24));
#endif
                uchar4 y1, y2;
                y1.x = convert_uchar_sat(fma(coeffs[0], src_pix1[      2-BIDX], fma(coeffs[1], src_pix1[      1], fma(coeffs[2], src_pix1[      BIDX], 16.5f))));
                y1.y = convert_uchar_sat(fma(coeffs[0], src_pix1[  SCN+2-BIDX], fma(coeffs[1], src_pix1[  SCN+1], fma(coeffs[2], src_pix1[  SCN+BIDX], 16.5f))));
                y1.z = convert_uchar_sat(fma(coeffs[0], src_pix1[2*SCN+2-BIDX], fma(coeffs[1], src_pix1[2*SCN+1], fma(coeffs[2], src_pix1[2*SCN+BIDX], 16.5f))));
                y1.w = convert_uchar_sat(fma(coeffs[0], src_pix1[3*SCN+2-BIDX], fma(coeffs[1], src_pix1[3*SCN+1], fma(coeffs[2], src_pix1[3*SCN+BIDX], 16.5f))));
                y2.x = convert_uchar_sat(fma(coeffs[0], src_pix2[      2-BIDX], fma(coeffs[1], src_pix2[      1], fma(coeffs[2], src_pix2[      BIDX], 16.5f))));
                y2.y = convert_uchar_sat(fma(coeffs[0], src_pix2[  SCN+2-BIDX], fma(coeffs[1], src_pix2[  SCN+1], fma(coeffs[2], src_pix2[  SCN+BIDX], 16.5f))));
                y2.z = convert_uchar_sat(fma(coeffs[0], src_pix2[2*SCN+2-BIDX], fma(coeffs[1], src_pix2[2*SCN+1], fma(coeffs[2], src_pix2[2*SCN+BIDX], 16.5f))));
                y2.w = convert_uchar_sat(fma(coeffs[0], src_pix2[3*SCN+2-BIDX], fma(coeffs[1], src_pix2[3*SCN+1], fma(coeffs[2], src_pix2[3*SCN+BIDX], 16.5f))));

                *((__global int*) ydst1) = as_int(y1);
                *((__global int*) ydst2) = as_int(y2);

                float uv[4] = { fma(coeffs[3], src_pix1[      2-BIDX], fma(coeffs[4], src_pix1[      1], fma(coeffs[5], src_pix1[      BIDX], 128.5f))),
                                fma(coeffs[5], src_pix1[      2-BIDX], fma(coeffs[6], src_pix1[      1], fma(coeffs[7], src_pix1[      BIDX], 128.5f))),
                                fma(coeffs[3], src_pix1[2*SCN+2-BIDX], fma(coeffs[4], src_pix1[2*SCN+1], fma(coeffs[5], src_pix1[2*SCN+BIDX], 128.5f))),
                                fma(coeffs[5], src_pix1[2*SCN+2-BIDX], fma(coeffs[6], src_pix1[2*SCN+1], fma(coeffs[7], src_pix1[2*SCN+BIDX], 128.5f))) };

                udst[0] = convert_uchar_sat(uv[UIDX]    );
                vdst[0] = convert_uchar_sat(uv[1 - UIDX]);
                udst[1] = convert_uchar_sat(uv[2 + UIDX]);
                vdst[1] = convert_uchar_sat(uv[3 - UIDX]);
#else
                float4 src_pix1 = convert_float4(vload4(0, src1));
                float4 src_pix2 = convert_float4(vload4(0, src1+SCN));
                float4 src_pix3 = convert_float4(vload4(0, src2));
                float4 src_pix4 = convert_float4(vload4(0, src2+SCN));

                ydst1[0] = convert_uchar_sat(fma(coeffs[0], src_pix1.R_COMP, fma(coeffs[1], src_pix1.G_COMP, fma(coeffs[2], src_pix1.B_COMP, 16.5f))));
                ydst1[1] = convert_uchar_sat(fma(coeffs[0], src_pix2.R_COMP, fma(coeffs[1], src_pix2.G_COMP, fma(coeffs[2], src_pix2.B_COMP, 16.5f))));
                ydst2[0] = convert_uchar_sat(fma(coeffs[0], src_pix3.R_COMP, fma(coeffs[1], src_pix3.G_COMP, fma(coeffs[2], src_pix3.B_COMP, 16.5f))));
                ydst2[1] = convert_uchar_sat(fma(coeffs[0], src_pix4.R_COMP, fma(coeffs[1], src_pix4.G_COMP, fma(coeffs[2], src_pix4.B_COMP, 16.5f))));

                float uv[2] = { fma(coeffs[3], src_pix1.R_COMP, fma(coeffs[4], src_pix1.G_COMP, fma(coeffs[5], src_pix1.B_COMP, 128.5f))),
                                fma(coeffs[5], src_pix1.R_COMP, fma(coeffs[6], src_pix1.G_COMP, fma(coeffs[7], src_pix1.B_COMP, 128.5f))) };

                udst[0] = convert_uchar_sat(uv[UIDX]  );
                vdst[0] = convert_uchar_sat(uv[1-UIDX]);
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
        __global uchar*       dst = dstptr + mad24(y, dst_step, mad24(x << 1, DCN, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows )
            {
                __constant float* coeffs = c_YUV2RGBCoeffs_420;

#ifndef USE_OPTIMIZED_LOAD
                float U = ((float) src[UIDX]) - HALF_MAX_NUM;
                float V = ((float) src[(2 + UIDX) % 4]) - HALF_MAX_NUM;
                float y00 = max(0.f, ((float) src[YIDX]) - 16.f) * coeffs[0];
                float y01 = max(0.f, ((float) src[YIDX + 2]) - 16.f) * coeffs[0];
#else
                int load_src = *((__global int*) src);
                float vec_src[4] = { load_src & 0xff, (load_src >> 8) & 0xff, (load_src >> 16) & 0xff, (load_src >> 24) & 0xff};
                float U = vec_src[UIDX] - HALF_MAX_NUM;
                float V = vec_src[(2 + UIDX) % 4] - HALF_MAX_NUM;
                float y00 = max(0.f, vec_src[YIDX] - 16.f) * coeffs[0];
                float y01 = max(0.f, vec_src[YIDX + 2] - 16.f) * coeffs[0];
#endif

                float ruv = fma(coeffs[4], V, 0.5f);
                float guv = fma(coeffs[3], V, fma(coeffs[2], U, 0.5f));
                float buv = fma(coeffs[1], U, 0.5f);

                dst[2 - BIDX] = convert_uchar_sat(y00 + ruv);
                dst[1]        = convert_uchar_sat(y00 + guv);
                dst[BIDX]     = convert_uchar_sat(y00 + buv);
#if DCN == 4
                dst[3]        = 255;
#endif

                dst[DCN + 2 - BIDX] = convert_uchar_sat(y01 + ruv);
                dst[DCN + 1]        = convert_uchar_sat(y01 + guv);
                dst[DCN + BIDX]     = convert_uchar_sat(y01 + buv);
#if DCN == 4
                dst[7]        = 255;
#endif
            }
            ++y;
            src += src_step;
            dst += dst_step;
        }
    }
}

// Coefficients based on ITU.BT-601, ISBN 1-878707-09-4 (https://fourcc.org/fccyvrgb.php)
// The conversion coefficients for RGB to YUV422 are based on the ones for RGB to YUV.
// For both Y components, the coefficients are applied as given in the link to each input RGB pixel
// separately. For U and V, they are reduced by half to account for two RGB pixels contributing
// to the same U and V values. In other words, the U and V contributions from the two RGB pixels
// are averaged. The integer versions are obtained by multiplying the float versions by 16384
// and rounding to the nearest integer.

__constant float c_RGB2YUV422Coeffs_f[10]  = {0.0625, 0.5, 0.257, 0.504, 0.098, -0.074 , -0.1455, 0.2195, -0.184 , -0.0355};
__constant int   c_RGB2YUV422Coeffs_i[10]  = {1024 * HALF_MAX_NUM * 2, 8192 * HALF_MAX_NUM * 2, 4211,  8258,  1606, -1212, -2384,  3596, -3015,  -582};

__kernel void RGB2YUV_422(__global const uchar* srcptr, int src_step, int src_offset,
                          __global uchar* dstptr, int dst_step, int dst_offset,
                          int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * PIX_PER_WI_Y;

    if (x < cols/2)
    {
        int src_index = mad24(y, src_step, mad24(x << 1, scnbytes, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x << 1, dcnbytes, dst_offset));

        #pragma unroll
        for (int cy = 0; cy < PIX_PER_WI_Y; ++cy)
        {
            if (y < rows)
            {
                __global const DATA_TYPE* src = (__global const DATA_TYPE*)(srcptr + src_index);
                __global DATA_TYPE* dst = (__global DATA_TYPE*)(dstptr + dst_index);
                DATA_TYPE_3 src_pix1 = vload3(0, src);
                DATA_TYPE b1 = src_pix1.B_COMP, g1 = src_pix1.G_COMP, r1 = src_pix1.R_COMP;
                DATA_TYPE_3 src_pix2 = vload3(0, src+SCN);
                DATA_TYPE b2 = src_pix2.B_COMP, g2 = src_pix2.G_COMP, r2 = src_pix2.R_COMP;


#ifdef DEPTH_5
                __constant float * coeffs = c_RGB2YUV422Coeffs_f;
                #define MAC_fn fma
                #define res_dtype DATA_TYPE
                #define mul_fn(x,y) (x*y)
                #define output_scale_fn(x) x
#else
                __constant int * coeffs = c_RGB2YUV422Coeffs_i;
                #define MAC_fn mad24
                #define res_dtype int
                #define mul_fn mul24
                #define output_scale_fn(x) SAT_CAST(CV_DESCALE(x, yuv_shift))
#endif

                const res_dtype Y1 = MAC_fn(coeffs[2], r1, coeffs[0] + MAC_fn(coeffs[3], g1, mul_fn(coeffs[4], b1)));
                const res_dtype Y2 = MAC_fn(coeffs[2], r2, coeffs[0] + MAC_fn(coeffs[3], g2, mul_fn(coeffs[4], b2)));

                const res_dtype sr = r1+r2, sg = g1+g2, sb = b1+b2;
                const res_dtype U = MAC_fn(coeffs[5], sr, coeffs[1] + MAC_fn(coeffs[6], sg, mul_fn(coeffs[7], sb)));
                const res_dtype V = MAC_fn(coeffs[7], sr, coeffs[1] + MAC_fn(coeffs[8], sg, mul_fn(coeffs[9], sb)));

                dst[UIDX] = output_scale_fn(U);
                dst[(2 + UIDX) % 4] = output_scale_fn(V);
                dst[YIDX] = output_scale_fn(Y1);
                dst[YIDX+2] = output_scale_fn(Y2);

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
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

                dstptr[(BIDX^2)] = SAT_CAST(r);
                dstptr[1] = SAT_CAST(g);
                dstptr[BIDX] = SAT_CAST(b);
#if DCN == 4
                dstptr[3] = MAX_NUM;
#endif

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}
