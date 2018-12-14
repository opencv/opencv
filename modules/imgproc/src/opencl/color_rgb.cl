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
    R2Y        = 4899,
    G2Y        = 9617,
    B2Y        = 1868
};

//constants for conversion from/to RGB and Gray, YUV, YCrCb according to BT.601
#define B2YF 0.114f
#define G2YF 0.587f
#define R2YF 0.299f

#define scnbytes ((int)sizeof(DATA_TYPE)*scn)
#define dcnbytes ((int)sizeof(DATA_TYPE)*dcn)

#if bidx == 0
#define R_COMP z
#define G_COMP y
#define B_COMP x
#else
#define R_COMP x
#define G_COMP y
#define B_COMP z
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
#if scn == 3
                DATA_TYPE_3 src_pix = vload3(0, src);
#else
                DATA_TYPE_4 src_pix = vload4(0, src);
#endif

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
                        (uchar4)(SAT_CAST(mad24(src_pix.x, MAX_NUM, v3_half) / v3),
                                 SAT_CAST(mad24(src_pix.y, MAX_NUM, v3_half) / v3),
                                 SAT_CAST(mad24(src_pix.z, MAX_NUM, v3_half) / v3),
                                 SAT_CAST(v3));

                ++y;
                dst_index += dst_step;
                src_index += src_step;
            }
        }
    }
}

#endif
