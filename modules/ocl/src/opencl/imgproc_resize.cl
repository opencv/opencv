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
//    Zhang Ying, zhangying913@gmail.com
//	  Niko Li, newlife20080214@gmail.com
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


// resize kernel
// Currently, CV_8UC1, CV_8UC4, CV_32FC1 and CV_32FC4 are supported.
// We shall support other types later if necessary.

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)
#define CAST_SCALE (1.0f/(1<<CAST_BITS))
#define INC(x,l) ((x+1) >= (l) ? (x):((x)+1))

#ifdef LN

__kernel void resizeLN_C1_D0(__global uchar * dst, __global uchar const * restrict src,
                     int dst_offset, int src_offset,int dst_step, int src_step,
                     int src_cols, int src_rows, int dst_cols, int dst_rows, float ifx, float ify )
{
    int gx = get_global_id(0);
    int dy = get_global_id(1);

    float4  sx, u, xf;
    int4 x, DX;
    gx = (gx<<2) - (dst_offset&3);
    DX = (int4)(gx, gx+1, gx+2, gx+3);
    sx = (convert_float4(DX) + 0.5f) * ifx - 0.5f;
    xf = floor(sx);
    x = convert_int4(xf);
    u = sx - xf;
    float sy = ((dy+0.5f) * ify - 0.5f);
    int y = floor(sy);
    float v = sy - y;

    u = x < 0 ? 0 : u;
    u = (x >= src_cols) ? 0 : u;
    x = x < 0 ? 0 : x;
    x = (x >= src_cols) ? src_cols-1 : x;

    y<0 ? y=0,v=0 : y;
    y>=src_rows ? y=src_rows-1,v=0 : y;

    int4 U, U1;
    int V, V1;
    float4 utmp1, utmp2;
    float vtmp;
    float4 scale_vec = INTER_RESIZE_COEF_SCALE;
    utmp1 = u * scale_vec;
    utmp2 = scale_vec - utmp1;
    U = convert_int4(rint(utmp1));
    U1 = convert_int4(rint(utmp2));
    vtmp = v * INTER_RESIZE_COEF_SCALE;
    V = rint(vtmp);
    V1= rint(INTER_RESIZE_COEF_SCALE - vtmp);

    int y_ = INC(y,src_rows);
    int4 x_;
    x_ =  ((x+1 >= src_cols) != 0) ? x : x+1;

    int4 val1, val2, val;
    int4 sdata1, sdata2, sdata3, sdata4;

    int4 pos1 = mad24((int4)y, (int4)src_step, x+(int4)src_offset);
    int4 pos2 = mad24((int4)y, (int4)src_step, x_+(int4)src_offset);
    int4 pos3 = mad24((int4)y_, (int4)src_step, x+(int4)src_offset);
    int4 pos4 = mad24((int4)y_, (int4)src_step, x_+(int4)src_offset);

    sdata1.s0 = src[pos1.s0];
    sdata1.s1 = src[pos1.s1];
    sdata1.s2 = src[pos1.s2];
    sdata1.s3 = src[pos1.s3];

    sdata2.s0 = src[pos2.s0];
    sdata2.s1 = src[pos2.s1];
    sdata2.s2 = src[pos2.s2];
    sdata2.s3 = src[pos2.s3];

    sdata3.s0 = src[pos3.s0];
    sdata3.s1 = src[pos3.s1];
    sdata3.s2 = src[pos3.s2];
    sdata3.s3 = src[pos3.s3];

    sdata4.s0 = src[pos4.s0];
    sdata4.s1 = src[pos4.s1];
    sdata4.s2 = src[pos4.s2];
    sdata4.s3 = src[pos4.s3];

    val1 = mul24(U1 , sdata1) + mul24(U , sdata2);
    val2 = mul24(U1 , sdata3) + mul24(U , sdata4);
    val = mul24((int4)V1 , val1) + mul24((int4)V , val2);

    val = ((val + (1<<(CAST_BITS-1))) >> CAST_BITS);

    pos4 = mad24(dy, dst_step, gx+dst_offset);
    pos4.y++;
    pos4.z+=2;
    pos4.w+=3;
    uchar4 uval = convert_uchar4_sat(val);
        int con = (gx >= 0 && gx+3 < dst_cols && dy >= 0 && dy < dst_rows && (dst_offset&3)==0);
    if(con)
    {
        *(__global uchar4*)(dst + pos4.x)=uval;
    }
    else
    {
        if(gx >= 0 && gx < dst_cols && dy >= 0 && dy < dst_rows)
        {
            dst[pos4.x]=uval.x;
        }
        if(gx+1 >= 0 && gx+1 < dst_cols && dy >= 0 && dy < dst_rows)
        {
            dst[pos4.y]=uval.y;
        }
        if(gx+2 >= 0 && gx+2 < dst_cols && dy >= 0 && dy < dst_rows)
        {
            dst[pos4.z]=uval.z;
        }
        if(gx+3 >= 0 && gx+3 < dst_cols && dy >= 0 && dy < dst_rows)
        {
            dst[pos4.w]=uval.w;
        }
    }
}

__kernel void resizeLN_C4_D0(__global uchar4 * dst, __global uchar4 * src,
                     int dst_offset, int src_offset,int dst_step, int src_step,
                     int src_cols, int src_rows, int dst_cols, int dst_rows, float ifx, float ify )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    float sx = ((dx+0.5f) * ifx - 0.5f), sy = ((dy+0.5f) * ify - 0.5f);
    int x = floor(sx), y = floor(sy);
    float u = sx - x, v = sy - y;

    if ( x<0 ) x=0,u=0;
    if ( x>=src_cols ) x=src_cols-1,u=0;
    if ( y<0 ) y=0,v=0;
    if (y>=src_rows ) y=src_rows-1,v=0;

    u = u * INTER_RESIZE_COEF_SCALE;
    v = v * INTER_RESIZE_COEF_SCALE;

    int U = rint(u);
    int V = rint(v);
    int U1= rint(INTER_RESIZE_COEF_SCALE - u);
    int V1= rint(INTER_RESIZE_COEF_SCALE - v);

    int y_ = INC(y,src_rows);
    int x_ = INC(x,src_cols);
    int4 srcpos;
    srcpos.x = mad24(y, src_step, x+src_offset);
    srcpos.y = mad24(y, src_step, x_+src_offset);
    srcpos.z = mad24(y_, src_step, x+src_offset);
    srcpos.w = mad24(y_, src_step, x_+src_offset);
    int4 data0 = convert_int4(src[srcpos.x]);
    int4 data1 = convert_int4(src[srcpos.y]);
    int4 data2 = convert_int4(src[srcpos.z]);
    int4 data3 = convert_int4(src[srcpos.w]);
    int4 val = mul24((int4)mul24(U1, V1) ,  data0) + mul24((int4)mul24(U, V1) ,  data1)
               +mul24((int4)mul24(U1, V) ,  data2)+mul24((int4)mul24(U, V) ,  data3);
    int dstpos = mad24(dy, dst_step, dx+dst_offset);
    uchar4 uval =   convert_uchar4((val + (1<<(CAST_BITS-1)))>>CAST_BITS);
    if(dx>=0 && dx<dst_cols && dy>=0 && dy<dst_rows)
         dst[dstpos] = uval;
}

__kernel void resizeLN_C1_D5(__global float * dst, __global float * src,
                     int dst_offset, int src_offset,int dst_step, int src_step,
                     int src_cols, int src_rows, int dst_cols, int dst_rows, float ifx, float ify )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    float sx = ((dx+0.5f) * ifx - 0.5f), sy = ((dy+0.5f) * ify - 0.5f);
    int x = floor(sx), y = floor(sy);
    float u = sx - x, v = sy - y;

    if ( x<0 ) x=0,u=0;
    if ( x>=src_cols ) x=src_cols-1,u=0;
    if ( y<0 ) y=0,v=0;
    if (y>=src_rows ) y=src_rows-1,v=0;

    int y_ = INC(y,src_rows);
    int x_ = INC(x,src_cols);
    float u1 = 1.f-u;
    float v1 = 1.f-v;
    int4 srcpos;
    srcpos.x = mad24(y, src_step, x+src_offset);
    srcpos.y = mad24(y, src_step, x_+src_offset);
    srcpos.z = mad24(y_, src_step, x+src_offset);
    srcpos.w = mad24(y_, src_step, x_+src_offset);
    float data0 = src[srcpos.x];
    float data1 = src[srcpos.y];
    float data2 = src[srcpos.z];
    float data3 = src[srcpos.w];
    float val1 = u1 *  data0 +
                u  *  data1 ;
    float val2 = u1 *  data2 +
                u *  data3;
    float val = v1 * val1 + v * val2;
    int dstpos = mad24(dy, dst_step, dx+dst_offset);
    if(dx>=0 && dx<dst_cols && dy>=0 && dy<dst_rows)
         dst[dstpos] = val;
}

__kernel void resizeLN_C4_D5(__global float4 * dst, __global float4 * src,
                     int dst_offset, int src_offset,int dst_step, int src_step,
                     int src_cols, int src_rows, int dst_cols, int dst_rows, float ifx, float ify )
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    float sx = ((dx+0.5f) * ifx - 0.5f), sy = ((dy+0.5f) * ify - 0.5f);
    int x = floor(sx), y = floor(sy);
    float u = sx - x, v = sy - y;

    if ( x<0 ) x=0,u=0;
    if ( x>=src_cols ) x=src_cols-1,u=0;
    if ( y<0 ) y=0,v=0;
    if (y>=src_rows ) y=src_rows-1,v=0;

    int y_ = INC(y,src_rows);
    int x_ = INC(x,src_cols);
    float u1 = 1.f-u;
    float v1 = 1.f-v;
    int4 srcpos;
    srcpos.x = mad24(y, src_step, x+src_offset);
    srcpos.y = mad24(y, src_step, x_+src_offset);
    srcpos.z = mad24(y_, src_step, x+src_offset);
    srcpos.w = mad24(y_, src_step, x_+src_offset);
    float4 s_data1, s_data2, s_data3, s_data4;
    s_data1 = src[srcpos.x];
    s_data2 = src[srcpos.y];
    s_data3 = src[srcpos.z];
    s_data4 = src[srcpos.w];
    float4 val = u1 * v1 * s_data1 + u * v1 * s_data2
              +u1 * v *s_data3 + u * v *s_data4;
    int dstpos = mad24(dy, dst_step, dx+dst_offset);

    if(dx>=0 && dx<dst_cols && dy>=0 && dy<dst_rows)
         dst[dstpos] = val;
}

#elif defined NN

__kernel void resizeNN(__global T * dst, __global T * src,
                       int dst_offset, int src_offset, int dst_step, int src_step,
                       int src_cols, int src_rows, int dst_cols, int dst_rows, float ifx, float ify)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        float s1 = dx * ifx, s2 = dy * ify;
        int sx = min(convert_int_sat_rtn(s1), src_cols - 1);
        int sy = min(convert_int_sat_rtn(s2), src_rows - 1);

        int dst_index = mad24(dy, dst_step, dx + dst_offset);
        int src_index = mad24(sy, src_step, sx + src_offset);

        dst[dst_index] = src[src_index];
    }
}

#elif defined AREA

#ifdef AREA_FAST

__kernel void resizeAREA_FAST(__global T * dst, __global T * src,
                         int dst_offset, int src_offset, int dst_step, int src_step,
                         int src_cols, int src_rows, int dst_cols, int dst_rows, WT ifx, WT ify,
                         __global const int * dmap_tab, __global const int * smap_tab)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        int dst_index = mad24(dy, dst_step, dst_offset + dx);

        __global const int * xmap_tab = dmap_tab;
        __global const int * ymap_tab = dmap_tab + dst_cols;
        __global const int * sxmap_tab = smap_tab;
        __global const int * symap_tab = smap_tab + XSCALE * dst_cols;

        int sx = xmap_tab[dx], sy = ymap_tab[dy];
        WTV sum = (WTV)(0);

        #pragma unroll
        for (int y = 0; y < YSCALE; ++y)
        {
            int src_index = mad24(symap_tab[y + sy], src_step, src_offset);
            #pragma unroll
            for (int x = 0; x < XSCALE; ++x)
                sum += convertToWTV(src[src_index + sxmap_tab[sx + x]]);
        }

        dst[dst_index] = convertToT(convertToWT2V(sum) * (WT2V)(SCALE));
    }
}

#else

__kernel void resizeAREA(__global T * dst, __global T * src,
                         int dst_offset, int src_offset, int dst_step, int src_step,
                         int src_cols, int src_rows, int dst_cols, int dst_rows, WT ifx, WT ify,
                         __global const int * ofs_tab, __global const int * map_tab,
                         __global const float * alpha_tab)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        int dst_index = mad24(dy, dst_step, dst_offset + dx);

        __global const int * xmap_tab = map_tab;
        __global const int * ymap_tab = (__global const int *)(map_tab + (src_cols << 1));
        __global const float * xalpha_tab = alpha_tab;
        __global const float * yalpha_tab = (__global const float *)(alpha_tab + (src_cols << 1));
        __global const int * xofs_tab = ofs_tab;
        __global const int * yofs_tab = (__global const int *)(ofs_tab + dst_cols + 1);

        int xk0 = xofs_tab[dx], xk1 = xofs_tab[dx + 1];
        int yk0 = yofs_tab[dy], yk1 = yofs_tab[dy + 1];

        int sy0 = ymap_tab[yk0], sy1 = ymap_tab[yk1 - 1];
        int sx0 = xmap_tab[xk0], sx1 = xmap_tab[xk1 - 1];

        WTV sum = (WTV)(0), buf;
        int src_index = mad24(sy0, src_step, src_offset);

        for (int sy = sy0, yk = yk0; sy <= sy1; ++sy, src_index += src_step, ++yk)
        {
            WTV beta = (WTV)(yalpha_tab[yk]);
            buf = (WTV)(0);

            for (int sx = sx0, xk = xk0; sx <= sx1; ++sx, ++xk)
            {
                WTV alpha = (WTV)(xalpha_tab[xk]);
                buf += convertToWTV(src[src_index + sx]) * alpha;
            }
            sum += buf * beta;
        }

        dst[dst_index] = convertToT(sum);
    }
}

#endif

#endif
