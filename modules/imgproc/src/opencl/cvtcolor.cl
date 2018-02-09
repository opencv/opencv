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
    xyz_shift  = 12
};

#define scnbytes ((int)sizeof(DATA_TYPE)*scn)
#define dcnbytes ((int)sizeof(DATA_TYPE)*dcn)

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)

#define DATA_TYPE_4 CAT(DATA_TYPE, 4)

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
