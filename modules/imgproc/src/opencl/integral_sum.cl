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
//    Shengen Yan,yanshengen@gmail.com
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

#define LSIZE 256
#define LSIZE_1 255
#define LSIZE_2 254
#define HF_LSIZE 128
#define LOG_LSIZE 8
#define LOG_NUM_BANKS 5
#define NUM_BANKS 32
#define GET_CONFLICT_OFFSET(lid) ((lid) >> LOG_NUM_BANKS)

#if sdepth == 4
#define sumT               int
#define vecSumT            int4
#define convertToSum4      convert_int4
#elif sdepth == 5
#define sumT               float
#define vecSumT            float4
#define convertToSum4      convert_float4
#endif


kernel void integral_sum_cols(__global const uchar4 *src, __global uchar *sum_ptr,
                              int src_offset, int rows, int cols, int src_step, int dst_step)
{
    __global sumT *sum = (__global sumT *)sum_ptr;
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    vecSumT src_t[2], sum_t[2];
    __local vecSumT lm_sum[2][LSIZE + LOG_LSIZE];
    __local sumT* sum_p;
    src_step = src_step >> 2;
    gid = gid << 1;
    int lid_prim = ((lid & 127) << 1) + 1;
    for (int i = 0; i < rows; i += LSIZE_1)
    {
        if (i + lid < rows)
        {
            int src_index = mad24((lid+i), src_step, gid + src_offset);
            src_t[0] = convertToSum4(src[src_index]);
            src_t[1] = convertToSum4(src[src_index + 1]);
        }
        else
        {
            src_t[0] = (vecSumT)0;
            src_t[1] = (vecSumT)0;
        }

        if (i == 0)
        {
            sum_t[0] = (vecSumT)0;
            sum_t[1] = (vecSumT)0;
        }
        else
        {
            sum_t[0] =  lm_sum[0][LSIZE_2 + LOG_LSIZE];
            sum_t[1] =  lm_sum[1][LSIZE_2 + LOG_LSIZE];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int bf_loc = lid + GET_CONFLICT_OFFSET(lid);

        lm_sum[0][bf_loc] = src_t[0];
        lm_sum[1][bf_loc] = src_t[1];

        int offset = 1;
        for (int d = LSIZE >> 1 ;  d > 0; d>>=1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            int ai = offset * lid_prim - 1,bi = ai + offset;
            ai += GET_CONFLICT_OFFSET(ai);
            bi += GET_CONFLICT_OFFSET(bi);

            if((lid & 127) < d)
            {
                lm_sum[lid >> 7][bi]  +=  lm_sum[lid >> 7][ai];
            }
            offset <<= 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < 2)
        {
            lm_sum[lid][LSIZE_2 + LOG_LSIZE] = 0;
        }
        for (int d = 1;  d < LSIZE; d <<= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset >>= 1;
            int ai = offset * lid_prim - 1,bi = ai + offset;
            ai += GET_CONFLICT_OFFSET(ai);
            bi += GET_CONFLICT_OFFSET(bi);

            if((lid & 127) < d)
            {
                lm_sum[lid >> 7][bi] += lm_sum[lid >> 7][ai];
                lm_sum[lid >> 7][ai] = lm_sum[lid >> 7][bi] - lm_sum[lid >> 7][ai];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid > 0 && (i+lid) <= rows)
        {
            int loc_s0 = mad24(gid, dst_step, i + lid - 1), loc_s1 = loc_s0 + dst_step;
            lm_sum[0][bf_loc] += sum_t[0];
            lm_sum[1][bf_loc] += sum_t[1];
            sum_p = (__local sumT*)(&(lm_sum[0][bf_loc]));
            for (int k = 0; k < 4; k++)
            {
                if (gid * 4 + k >= cols)
                    break;
                sum[loc_s0 + k * dst_step / 4] = sum_p[k];
            }
            sum_p = (__local sumT*)(&(lm_sum[1][bf_loc]));
            for (int k = 0; k < 4; k++)
            {
                if (gid * 4 + k + 4 >= cols)
                    break;
                sum[loc_s1 + k * dst_step / 4] = sum_p[k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


kernel void integral_sum_rows(__global const uchar *srcsum_ptr, __global uchar *sum_ptr,
                              int rows, int cols, int src_step, int sum_step, int sum_offset)
{
    __global const vecSumT *srcsum = (__global const vecSumT *)srcsum_ptr;
    __global sumT *sum = (__global sumT *)sum_ptr;
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    vecSumT src_t[2], sum_t[2];
    __local vecSumT lm_sum[2][LSIZE + LOG_LSIZE];
    __local sumT *sum_p;
    src_step = src_step >> 4;
    int lid_prim = ((lid & 127) << 1) + 1;
    for (int i = 0; i < rows; i += LSIZE_1)
    {
        if (i + lid < rows)
        {
            int sum_idx = mad24(lid + i, src_step, gid * 2);
            src_t[0] = srcsum[sum_idx];
            src_t[1] = srcsum[sum_idx + 1];
        }
        else
        {
            src_t[0] = 0;
            src_t[1] = 0;
        }
        if (i == 0)
        {
            sum_t[0] =  0;
            sum_t[1] =  0;
        }
        else
        {
            sum_t[0] =  lm_sum[0][LSIZE_2 + LOG_LSIZE];
            sum_t[1] =  lm_sum[1][LSIZE_2 + LOG_LSIZE];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        int bf_loc = lid + GET_CONFLICT_OFFSET(lid);

        lm_sum[0][bf_loc] = src_t[0];
        lm_sum[1][bf_loc] = src_t[1];

        int offset = 1;
        for (int d = LSIZE >> 1 ;  d > 0; d>>=1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            int ai = offset * lid_prim - 1, bi = ai + offset;
            ai += GET_CONFLICT_OFFSET(ai);
            bi += GET_CONFLICT_OFFSET(bi);

            if((lid & 127) < d)
            {
                lm_sum[lid >> 7][bi]  +=  lm_sum[lid >> 7][ai];
            }
            offset <<= 1;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < 2)
        {
            lm_sum[lid][LSIZE_2 + LOG_LSIZE] = 0;
        }
        for (int d = 1;  d < LSIZE; d <<= 1)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            offset >>= 1;
            int ai = offset * lid_prim - 1,bi = ai + offset;
            ai += GET_CONFLICT_OFFSET(ai);
            bi += GET_CONFLICT_OFFSET(bi);

            if ((lid & 127) < d)
            {
                lm_sum[lid >> 7][bi] += lm_sum[lid >> 7][ai];
                lm_sum[lid >> 7][ai] = lm_sum[lid >> 7][bi] - lm_sum[lid >> 7][ai];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (gid == 0 && (i + lid) <= rows)
        {
            sum[sum_offset + i + lid] = 0;
        }
        if (i + lid == 0)
        {
            int loc0 = gid * 2 * sum_step;
            for(int k = 1; k <= 8; k++)
            {
                if (gid * 8 + k > cols)
                    break;
                sum[sum_offset + loc0 + k * sum_step / 4] = 0;
            }
        }

        if (lid > 0 && (i+lid) <= rows)
        {
            int loc_s0 = sum_offset + gid * 2 * sum_step + sum_step / 4 + i + lid, loc_s1 = loc_s0 + sum_step ;
            lm_sum[0][bf_loc] += sum_t[0];
            lm_sum[1][bf_loc] += sum_t[1];
            sum_p = (__local sumT*)(&(lm_sum[0][bf_loc]));
            for(int k = 0; k < 4; k++)
            {
                if (gid * 8 + k >= cols)
                    break;
                sum[loc_s0 + k * sum_step / 4] = sum_p[k];
            }
            sum_p = (__local sumT*)(&(lm_sum[1][bf_loc]));
            for(int k = 0; k < 4; k++)
            {
                if (gid * 8 + 4 + k >= cols)
                    break;
                sum[loc_s1 + k * sum_step / 4] = sum_p[k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
