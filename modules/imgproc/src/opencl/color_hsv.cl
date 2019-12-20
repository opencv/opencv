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
    hsv_shift  = 12
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
#else
#define R_COMP x
#define G_COMP y
#define B_COMP z
#endif

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
