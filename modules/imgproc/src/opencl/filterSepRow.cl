//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2014, Itseez, Inc, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
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
//

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define READ_TIMES_ROW ((2*(RADIUSX+LSIZE0)-1)/LSIZE0) //for c4 only
#define RADIUS 1

#ifdef BORDER_REPLICATE
// BORDER_REPLICATE: aaaaaa|abcdefgh|hhhhhhh
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (l_edge)   : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (r_edge)-1 : (addr))
#endif

#ifdef BORDER_REFLECT
// BORDER_REFLECT: fedcba|abcdefgh|hgfedcb
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)-1               : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-1+((r_edge)<<1) : (addr))
#endif

#ifdef BORDER_REFLECT_101
// BORDER_REFLECT_101: gfedcb|abcdefgh|gfedcba
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? -(i)                 : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? -(i)-2+((r_edge)<<1) : (addr))
#endif

#ifdef BORDER_WRAP
// BORDER_WRAP: cdefgh|abcdefgh|abcdefg
#define ADDR_L(i, l_edge, r_edge)  ((i) <  (l_edge) ? (i)+(r_edge) : (i))
#define ADDR_R(i, r_edge, addr)    ((i) >= (r_edge) ? (i)-(r_edge) : (addr))
#endif

#ifdef EXTRA_EXTRAPOLATION // border > src image size
    #ifdef BORDER_CONSTANT
        #define ELEM(i,l_edge,r_edge,elem1,elem2) (i)<(l_edge) | (i) >= (r_edge) ? (elem1) : (elem2)
    #elif defined BORDER_REPLICATE
        #define EXTRAPOLATE(t, minT, maxT) \
        { \
            t = max(min(t, (maxT) - 1), (minT)); \
        }
    #elif defined BORDER_WRAP
        #define EXTRAPOLATE(x, minT, maxT) \
        { \
            if (t < (minT)) \
                t -= ((t - (maxT) + 1) / (maxT)) * (maxT); \
            if (t >= (maxT)) \
                t %= (maxT); \
        }
    #elif defined(BORDER_REFLECT) || defined(BORDER_REFLECT_101)
        #define EXTRAPOLATE_(t, minT, maxT, delta) \
        { \
            if ((maxT) - (minT) == 1) \
                t = (minT); \
            else \
                do \
                { \
                    if (t < (minT)) \
                        t = (minT) - (t - (minT)) - 1 + delta; \
                    else \
                        t = (maxT) - 1 - (t - (maxT)) - delta; \
                } \
                while (t >= (maxT) || t < (minT)); \
            \
        }
        #ifdef BORDER_REFLECT
            #define EXTRAPOLATE(t, minT, maxT) EXTRAPOLATE_(t, minT, maxT, 0)
        #elif defined(BORDER_REFLECT_101)
            #define EXTRAPOLATE(t, minT, maxT) EXTRAPOLATE_(t, minT, maxT, 1)
        #endif
    #else
        #error No extrapolation method
    #endif //BORDER_....
#else //EXTRA_EXTRAPOLATION
    #ifdef BORDER_CONSTANT
        #define ELEM(i,l_edge,r_edge,elem1,elem2) (i)<(l_edge) | (i) >= (r_edge) ? (elem1) : (elem2)
    #else
        #define EXTRAPOLATE(t, minT, maxT) \
        { \
            int _delta = t - (minT); \
            _delta = ADDR_L(_delta, 0, (maxT) - (minT)); \
            _delta = ADDR_R(_delta, (maxT) - (minT), _delta); \
            t = _delta + (minT); \
        }
    #endif //BORDER_CONSTANT
#endif //EXTRA_EXTRAPOLATION

#define noconvert

#if CN != 3
#define loadpix(addr) *(__global const srcT *)(addr)
#define storepix(val, addr)  *(__global dstT *)(addr) = val
#define SRCSIZE (int)sizeof(srcT)
#define DSTSIZE (int)sizeof(dstT)
#else
#define loadpix(addr)  vload3(0, (__global const srcT1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global dstT1 *)(addr))
#define SRCSIZE (int)sizeof(srcT1)*3
#define DSTSIZE (int)sizeof(dstT1)*3
#endif

#define DIG(a) a,
#if defined(INTEGER_ARITHMETIC)
__constant int mat_kernel[] = { COEFF };
#else
__constant dstT1 mat_kernel[] = { COEFF };
#endif

#if defined(INTEGER_ARITHMETIC)
#define dstT4 int4
#define convertDstVec convert_int4
#else
#define dstT4 float4
#define convertDstVec convert_float4
#endif

__kernel void row_filter_C1_D0(__global const uchar * src, int src_step_in_pixel, int src_offset_x, int src_offset_y,
                               int src_cols, int src_rows, int src_whole_cols, int src_whole_rows,
                               __global float * dst, int dst_step_in_pixel, int dst_cols, int dst_rows,
                               int radiusy)
{
    int x = get_global_id(0)<<2;
    int y = get_global_id(1);
    int l_x = get_local_id(0);
    int l_y = get_local_id(1);

    int start_x = x + src_offset_x - RADIUSX & 0xfffffffc;
    int offset = src_offset_x - RADIUSX & 3;
    int start_y = y + src_offset_y - radiusy;
    int start_addr = mad24(start_y, src_step_in_pixel, start_x);

    dstT4 sum;
    uchar4 temp[READ_TIMES_ROW];

    __local uchar4 LDS_DAT[LSIZE1][READ_TIMES_ROW * LSIZE0 + 1];
#ifdef BORDER_CONSTANT
    int end_addr = mad24(src_whole_rows - 1, src_step_in_pixel, src_whole_cols);

    // read pixels from src
    for (int i = 0; i < READ_TIMES_ROW; ++i)
    {
        int current_addr = mad24(i, LSIZE0 << 2, start_addr);
        current_addr = current_addr < end_addr && current_addr > 0 ? current_addr : 0;
        temp[i] = *(__global const uchar4 *)&src[current_addr];
    }

    // judge if read out of boundary
#ifdef BORDER_ISOLATED
    for (int i = 0; i < READ_TIMES_ROW; ++i)
    {
        temp[i].x = ELEM(start_x+i*LSIZE0*4,   src_offset_x, src_offset_x + src_cols, 0,         temp[i].x);
        temp[i].y = ELEM(start_x+i*LSIZE0*4+1, src_offset_x, src_offset_x + src_cols, 0,         temp[i].y);
        temp[i].z = ELEM(start_x+i*LSIZE0*4+2, src_offset_x, src_offset_x + src_cols, 0,         temp[i].z);
        temp[i].w = ELEM(start_x+i*LSIZE0*4+3, src_offset_x, src_offset_x + src_cols, 0,         temp[i].w);
        temp[i]   = ELEM(start_y,              src_offset_y, src_offset_y + src_rows, (uchar4)0, temp[i]);
    }
#else
    for (int i = 0; i < READ_TIMES_ROW; ++i)
    {
        temp[i].x = ELEM(start_x+i*LSIZE0*4,   0, src_whole_cols, 0,         temp[i].x);
        temp[i].y = ELEM(start_x+i*LSIZE0*4+1, 0, src_whole_cols, 0,         temp[i].y);
        temp[i].z = ELEM(start_x+i*LSIZE0*4+2, 0, src_whole_cols, 0,         temp[i].z);
        temp[i].w = ELEM(start_x+i*LSIZE0*4+3, 0, src_whole_cols, 0,         temp[i].w);
        temp[i]   = ELEM(start_y,              0, src_whole_rows, (uchar4)0, temp[i]);
    }
#endif
#else // BORDER_CONSTANT
#ifdef BORDER_ISOLATED
    int not_all_in_range = (start_x<src_offset_x) | (start_x + READ_TIMES_ROW*LSIZE0*4+4>src_offset_x + src_cols)| (start_y<src_offset_y) | (start_y >= src_offset_y + src_rows);
#else
    int not_all_in_range = (start_x<0) | (start_x + READ_TIMES_ROW*LSIZE0*4+4>src_whole_cols)| (start_y<0) | (start_y >= src_whole_rows);
#endif
    int4 index[READ_TIMES_ROW], addr;
    int s_y;

    if (not_all_in_range)
    {
        // judge if read out of boundary
        for (int i = 0; i < READ_TIMES_ROW; ++i)
        {
            index[i] = (int4)(mad24(i, LSIZE0 << 2, start_x)) + (int4)(0, 1, 2, 3);
#ifdef BORDER_ISOLATED
            EXTRAPOLATE(index[i].x, src_offset_x, src_offset_x + src_cols);
            EXTRAPOLATE(index[i].y, src_offset_x, src_offset_x + src_cols);
            EXTRAPOLATE(index[i].z, src_offset_x, src_offset_x + src_cols);
            EXTRAPOLATE(index[i].w, src_offset_x, src_offset_x + src_cols);
#else
            EXTRAPOLATE(index[i].x, 0, src_whole_cols);
            EXTRAPOLATE(index[i].y, 0, src_whole_cols);
            EXTRAPOLATE(index[i].z, 0, src_whole_cols);
            EXTRAPOLATE(index[i].w, 0, src_whole_cols);
#endif
        }

        s_y = start_y;
#ifdef BORDER_ISOLATED
        EXTRAPOLATE(s_y, src_offset_y, src_offset_y + src_rows);
#else
        EXTRAPOLATE(s_y, 0, src_whole_rows);
#endif

        // read pixels from src
        for (int i = 0; i < READ_TIMES_ROW; ++i)
        {
            addr = mad24((int4)s_y, (int4)src_step_in_pixel, index[i]);
            temp[i].x = src[addr.x];
            temp[i].y = src[addr.y];
            temp[i].z = src[addr.z];
            temp[i].w = src[addr.w];
        }
    }
    else
    {
        // read pixels from src
        for (int i = 0; i < READ_TIMES_ROW; ++i)
            temp[i] = *(__global uchar4*)&src[mad24(i, LSIZE0 << 2, start_addr)];
    }
#endif //BORDER_CONSTANT

    // save pixels to lds
    for (int i = 0; i < READ_TIMES_ROW; ++i)
        LDS_DAT[l_y][mad24(i, LSIZE0, l_x)] = temp[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // read pixels from lds and calculate the result
    sum = convertDstVec(vload4(0,(__local uchar *)&LDS_DAT[l_y][l_x]+RADIUSX+offset)) * mat_kernel[RADIUSX];
    for (int i = 1; i <= RADIUSX; ++i)
    {
        temp[0] = vload4(0, (__local uchar*)&LDS_DAT[l_y][l_x] + RADIUSX + offset - i);
        temp[1] = vload4(0, (__local uchar*)&LDS_DAT[l_y][l_x] + RADIUSX + offset + i);
#if defined(INTEGER_ARITHMETIC)
        sum += mad24(convertDstVec(temp[0]), mat_kernel[RADIUSX-i], convertDstVec(temp[1]) * mat_kernel[RADIUSX + i]);
#else
        sum += mad(convertDstVec(temp[0]), mat_kernel[RADIUSX-i], convertDstVec(temp[1]) * mat_kernel[RADIUSX + i]);
#endif
    }

    start_addr = mad24(y, dst_step_in_pixel, x);

    // write the result to dst
    if ((x+3<dst_cols) & (y<dst_rows))
        *(__global dstT4*)&dst[start_addr] = sum;
    else if ((x+2<dst_cols) && (y<dst_rows))
    {
        dst[start_addr] = sum.x;
        dst[start_addr+1] = sum.y;
        dst[start_addr+2] = sum.z;
    }
    else if ((x+1<dst_cols) && (y<dst_rows))
    {
        dst[start_addr] = sum.x;
        dst[start_addr+1] = sum.y;
    }
    else if (x<dst_cols && y<dst_rows)
        dst[start_addr] = sum.x;
}

__kernel void row_filter(__global const uchar * src, int src_step, int src_offset_x, int src_offset_y,
                         int src_cols, int src_rows, int src_whole_cols, int src_whole_rows,
                         __global uchar * dst, int dst_step, int dst_cols, int dst_rows,
                         int radiusy)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int l_x = get_local_id(0);
    int l_y = get_local_id(1);

    int start_x = x + src_offset_x - RADIUSX;
    int start_y = y + src_offset_y - radiusy;
    int start_addr = mad24(start_y, src_step, start_x * SRCSIZE);

    dstT sum;
    srcT temp[READ_TIMES_ROW];

    __local srcT LDS_DAT[LSIZE1][READ_TIMES_ROW * LSIZE0 + 1];
#ifdef BORDER_CONSTANT
    int end_addr = mad24(src_whole_rows - 1, src_step, src_whole_cols * SRCSIZE);

    // read pixels from src
    for (int i = 0; i < READ_TIMES_ROW; i++)
    {
        int current_addr = mad24(i, LSIZE0 * SRCSIZE, start_addr);
        current_addr = current_addr < end_addr && current_addr >= 0 ? current_addr : 0;
        temp[i] = loadpix(src + current_addr);
    }

    // judge if read out of boundary
#ifdef BORDER_ISOLATED
    for (int i = 0; i < READ_TIMES_ROW; ++i)
    {
        temp[i] = ELEM(mad24(i, LSIZE0, start_x), src_offset_x, src_offset_x + src_cols, (srcT)(0), temp[i]);
        temp[i] = ELEM(start_y,                   src_offset_y, src_offset_y + src_rows, (srcT)(0), temp[i]);
    }
#else
    for (int i = 0; i < READ_TIMES_ROW; ++i)
    {
        temp[i] = ELEM(mad24(i, LSIZE0, start_x), 0, src_whole_cols, (srcT)(0), temp[i]);
        temp[i] = ELEM(start_y,                   0, src_whole_rows, (srcT)(0), temp[i]);
    }
#endif
#else
    int index[READ_TIMES_ROW], s_x, s_y;

    // judge if read out of boundary
    for (int i = 0; i < READ_TIMES_ROW; ++i)
    {
        s_x = mad24(i, LSIZE0, start_x);
        s_y = start_y;

#ifdef BORDER_ISOLATED
        EXTRAPOLATE(s_x, src_offset_x, src_offset_x + src_cols);
        EXTRAPOLATE(s_y, src_offset_y, src_offset_y + src_rows);
#else
        EXTRAPOLATE(s_x, 0, src_whole_cols);
        EXTRAPOLATE(s_y, 0, src_whole_rows);
#endif
        index[i] = mad24(s_y, src_step, s_x * SRCSIZE);
    }

    // read pixels from src
    for (int i = 0; i < READ_TIMES_ROW; ++i)
        temp[i] = loadpix(src + index[i]);
#endif // BORDER_CONSTANT

    // save pixels to lds
    for (int i = 0; i < READ_TIMES_ROW; ++i)
        LDS_DAT[l_y][mad24(i, LSIZE0, l_x)] = temp[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // read pixels from lds and calculate the result
    sum = convertToDstT(LDS_DAT[l_y][l_x + RADIUSX]) * mat_kernel[RADIUSX];
    for (int i = 1; i <= RADIUSX; ++i)
    {
        temp[0] = LDS_DAT[l_y][l_x + RADIUSX - i];
        temp[1] = LDS_DAT[l_y][l_x + RADIUSX + i];
#if defined(INTEGER_ARITHMETIC)
        sum += mad24(convertToDstT(temp[0]), mat_kernel[RADIUSX - i], convertToDstT(temp[1]) * mat_kernel[RADIUSX + i]);
#else
        sum += mad(convertToDstT(temp[0]), mat_kernel[RADIUSX - i], convertToDstT(temp[1]) * mat_kernel[RADIUSX + i]);
#endif
    }

    // write the result to dst
    if (x < dst_cols && y < dst_rows)
    {
        start_addr = mad24(y, dst_step, x * DSTSIZE);
        storepix(sum, dst + start_addr);
    }
}
