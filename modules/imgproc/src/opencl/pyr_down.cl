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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Dachuan Zhao, dachuan@multicorewareinc.com
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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#if defined BORDER_REPLICATE
// aaaaaa|abcdefgh|hhhhhhh
#define EXTRAPOLATE(x, maxV) clamp((x), 0, (maxV)-1)
#elif defined BORDER_WRAP
// cdefgh|abcdefgh|abcdefg
#define EXTRAPOLATE(x, maxV) ( (x) + (maxV) ) % (maxV)
#elif defined BORDER_REFLECT
// fedcba|abcdefgh|hgfedcb
#define EXTRAPOLATE(x, maxV) clamp(min(((maxV)-1)*2-(x)+1, max((x),-(x)-1) ), 0, (maxV)-1)
#elif defined BORDER_REFLECT_101 || defined BORDER_REFLECT101
// gfedcb|abcdefgh|gfedcba
#define EXTRAPOLATE(x, maxV) clamp(min(((maxV)-1)*2-(x), max((x),-(x)) ), 0, (maxV)-1)
#else
#error No extrapolation method
#endif

#if cn != 3
#define loadpix(addr)  *(__global const T*)(addr)
#define storepix(val, addr)  *(__global T*)(addr) = (val)
#define PIXSIZE ((int)sizeof(T))
#else
#define loadpix(addr)  vload3(0, (__global const T1*)(addr))
#define storepix(val, addr) vstore3((val), 0, (__global T1*)(addr))
#define PIXSIZE ((int)sizeof(T1)*3)
#endif

#define SRC(_x,_y) convertToFT(loadpix(srcData + mad24(_y, src_step, PIXSIZE * _x)))

#if kercn == 4
#define SRC4(_x,_y) convert_float4(vload4(0, srcData + mad24(_y, src_step, PIXSIZE * _x)))
#endif

#ifdef INTEL_DEVICE
#define MAD(x,y,z) fma((x),(y),(z))
#else
#define MAD(x,y,z) mad((x),(y),(z))
#endif

#define LOAD_LOCAL(col_gl, col_lcl) \
    sum0 =     co3* SRC(col_gl, EXTRAPOLATE_(src_y - 2, src_rows));         \
    sum0 = MAD(co2, SRC(col_gl, EXTRAPOLATE_(src_y - 1, src_rows)), sum0);  \
    temp = SRC(col_gl, EXTRAPOLATE_(src_y, src_rows));                      \
    sum0 = MAD(co1, temp, sum0);                                            \
    sum1 = co3 * temp;                                                      \
    temp = SRC(col_gl, EXTRAPOLATE_(src_y + 1, src_rows));                  \
    sum0 = MAD(co2, temp, sum0);                                            \
    sum1 = MAD(co2, temp, sum1);                                            \
    temp = SRC(col_gl, EXTRAPOLATE_(src_y + 2, src_rows));                  \
    sum0 = MAD(co3, temp, sum0);                                            \
    sum1 = MAD(co1, temp, sum1);                                            \
    smem[0][col_lcl] = sum0;                                                \
    sum1 = MAD(co2, SRC(col_gl, EXTRAPOLATE_(src_y + 3, src_rows)), sum1);  \
    sum1 = MAD(co3, SRC(col_gl, EXTRAPOLATE_(src_y + 4, src_rows)), sum1);  \
    smem[1][col_lcl] = sum1;


#if kercn == 4
#define LOAD_LOCAL4(col_gl, col_lcl) \
    sum40 =     co3* SRC4(col_gl, EXTRAPOLATE_(src_y - 2, src_rows));           \
    sum40 = MAD(co2, SRC4(col_gl, EXTRAPOLATE_(src_y - 1, src_rows)), sum40);   \
    temp4 = SRC4(col_gl,  EXTRAPOLATE_(src_y, src_rows));                       \
    sum40 = MAD(co1, temp4, sum40);                                             \
    sum41 = co3 * temp4;                                                        \
    temp4 = SRC4(col_gl,  EXTRAPOLATE_(src_y + 1, src_rows));                   \
    sum40 = MAD(co2, temp4, sum40);                                             \
    sum41 = MAD(co2, temp4, sum41);                                             \
    temp4 = SRC4(col_gl,  EXTRAPOLATE_(src_y + 2, src_rows));                   \
    sum40 = MAD(co3, temp4, sum40);                                             \
    sum41 = MAD(co1, temp4, sum41);                                             \
    vstore4(sum40, col_lcl, (__local float*) &smem[0][2]);                      \
    sum41 = MAD(co2, SRC4(col_gl,  EXTRAPOLATE_(src_y + 3, src_rows)), sum41);  \
    sum41 = MAD(co3, SRC4(col_gl,  EXTRAPOLATE_(src_y + 4, src_rows)), sum41);  \
    vstore4(sum41, col_lcl, (__local float*) &smem[1][2]);
#endif

#define noconvert

__kernel void pyrDown(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                         __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    const int x = get_global_id(0)*kercn;
    const int y = 2*get_global_id(1);

    __local FT smem[2][LOCAL_SIZE + 4];
    __global uchar * dstData = dst + dst_offset;
    __global const uchar * srcData = src + src_offset;

    FT sum0, sum1, temp;
    FT co1 = 0.375f;
    FT co2 = 0.25f;
    FT co3 = 0.0625f;

    const int src_y = 2*y;
    int col;

    if (src_y >= 2 && src_y < src_rows - 4)
    {
#define EXTRAPOLATE_(val, maxVal)   val
#if kercn == 1
        col = EXTRAPOLATE(x, src_cols);
        LOAD_LOCAL(col, 2 + get_local_id(0))
#else
        if (x < src_cols-4)
        {
            float4 sum40, sum41, temp4;
            LOAD_LOCAL4(x, get_local_id(0))
        }
        else
        {
            for (int i=0; i<4; i++)
            {
                col = EXTRAPOLATE(x+i, src_cols);
                LOAD_LOCAL(col, 2 + 4 * get_local_id(0) + i)
            }
        }
#endif
        if (get_local_id(0) < 2)
        {
            col = EXTRAPOLATE((int)(get_group_id(0)*LOCAL_SIZE + get_local_id(0) - 2), src_cols);
            LOAD_LOCAL(col, get_local_id(0))
        }
        else if (get_local_id(0) < 4)
        {
            col = EXTRAPOLATE((int)((get_group_id(0)+1)*LOCAL_SIZE + get_local_id(0) - 2), src_cols);
            LOAD_LOCAL(col, LOCAL_SIZE + get_local_id(0))
        }
    }
    else // need extrapolate y
    {
#define EXTRAPOLATE_(val, maxVal)   EXTRAPOLATE(val, maxVal)
#if kercn == 1
        col = EXTRAPOLATE(x, src_cols);
        LOAD_LOCAL(col, 2 + get_local_id(0))
#else
        if (x < src_cols-4)
        {
            float4 sum40, sum41, temp4;
            LOAD_LOCAL4(x, get_local_id(0))
        }
        else
        {
            for (int i=0; i<4; i++)
            {
                col = EXTRAPOLATE(x+i, src_cols);
                LOAD_LOCAL(col, 2 + 4*get_local_id(0) + i)
            }
        }
#endif
        if (get_local_id(0) < 2)
        {
            col = EXTRAPOLATE((int)(get_group_id(0)*LOCAL_SIZE + get_local_id(0) - 2), src_cols);
            LOAD_LOCAL(col, get_local_id(0))
        }
        else if (get_local_id(0) < 4)
        {
            col = EXTRAPOLATE((int)((get_group_id(0)+1)*LOCAL_SIZE + get_local_id(0) - 2), src_cols);
            LOAD_LOCAL(col, LOCAL_SIZE + get_local_id(0))
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

#if kercn == 1
    if (get_local_id(0) < LOCAL_SIZE / 2)
    {
        const int tid2 = get_local_id(0) * 2;

        const int dst_x = (get_group_id(0) * get_local_size(0) + tid2) / 2;

        if (dst_x < dst_cols)
        {
            for (int yin = y, y1 = min(dst_rows, y + 2); yin < y1; yin++)
            {
#if cn == 1
#if fdepth <= 5
                FT sum = dot(vload4(0, (__local float*) (&smem) + tid2 + (yin - y) * (LOCAL_SIZE + 4)), (float4)(co3, co2, co1, co2));
#else
                FT sum = dot(vload4(0, (__local double*) (&smem) + tid2 + (yin - y) * (LOCAL_SIZE + 4)), (double4)(co3, co2, co1, co2));
#endif
#else
                FT sum = co3 * smem[yin - y][2 + tid2 - 2];
                sum = MAD(co2, smem[yin - y][2 + tid2 - 1], sum);
                sum = MAD(co1, smem[yin - y][2 + tid2    ], sum);
                sum = MAD(co2, smem[yin - y][2 + tid2 + 1], sum);
#endif
                sum = MAD(co3, smem[yin - y][2 + tid2 + 2], sum);
                storepix(convertToT(sum), dstData + yin * dst_step + dst_x * PIXSIZE);
            }
        }
    }
#else
    int tid4 = get_local_id(0) * 4;
    int dst_x = (get_group_id(0) * LOCAL_SIZE + tid4) / 2;
    if (dst_x < dst_cols - 1)
    {
        for (int yin = y, y1 = min(dst_rows, y + 2); yin < y1; yin++)
        {

            FT sum =  co3* smem[yin - y][2 + tid4 + 2];
            sum = MAD(co3, smem[yin - y][2 + tid4 - 2], sum);
            sum = MAD(co2, smem[yin - y][2 + tid4 - 1], sum);
            sum = MAD(co1, smem[yin - y][2 + tid4    ], sum);
            sum = MAD(co2, smem[yin - y][2 + tid4 + 1], sum);
            storepix(convertToT(sum), dstData + mad24(yin, dst_step, dst_x * PIXSIZE));

            dst_x ++;
            sum =     co3* smem[yin - y][2 + tid4 + 4];
            sum = MAD(co3, smem[yin - y][2 + tid4    ], sum);
            sum = MAD(co2, smem[yin - y][2 + tid4 + 1], sum);
            sum = MAD(co1, smem[yin - y][2 + tid4 + 2], sum);
            sum = MAD(co2, smem[yin - y][2 + tid4 + 3], sum);
            storepix(convertToT(sum), dstData + mad24(yin, dst_step, dst_x * PIXSIZE));
            dst_x --;
        }

    }
    else if (dst_x < dst_cols)
    {
        for (int yin = y, y1 = min(dst_rows, y + 2); yin < y1; yin++)
        {
            FT sum =  co3* smem[yin - y][2 + tid4 + 2];
            sum = MAD(co3, smem[yin - y][2 + tid4 - 2], sum);
            sum = MAD(co2, smem[yin - y][2 + tid4 - 1], sum);
            sum = MAD(co1, smem[yin - y][2 + tid4    ], sum);
            sum = MAD(co2, smem[yin - y][2 + tid4 + 1], sum);

            storepix(convertToT(sum), dstData + mad24(yin, dst_step, dst_x * PIXSIZE));
        }
    }
#endif

}
