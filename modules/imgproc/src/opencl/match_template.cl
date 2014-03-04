//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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

#define DATA_SIZE ((int)sizeof(type))
#define ELEM_TYPE elem_type
#define ELEM_SIZE ((int)sizeof(elem_type))

#define SQSUMS_PTR(ox, oy) mad24(y + oy, src_sqsums_step, mad24(x + ox, cn, src_sqsums_offset))
#define SUMS_PTR(ox, oy) mad24(y + oy, src_sums_step, mad24(x + ox, cn, src_sums_offset))

inline float normAcc(float num, float denum)
{
    if (fabs(num) < denum)
        return num / denum;
    if (fabs(num) < denum * 1.125f)
        return num > 0 ? 1 : -1;
    return 0;
}

inline float normAcc_SQDIFF(float num, float denum)
{
    if (fabs(num) < denum)
        return num / denum;
    if (fabs(num) < denum * 1.125f)
        return num > 0 ? 1 : -1;
    return 1;
}

#define noconvert

#if cn == 1
#define convertToDT(value) (float)(value)
#elif cn == 2
#define convertToDT(value) (float)(value.x + value.y)
#elif cn == 4
#define convertToDT(value) (float)(value.x + value.y + value.z + value.w)
#else
#error "cn should be 1, 2 or 4"
#endif

#ifdef CALC_SUM

__kernel void calcSum(__global const uchar * srcptr, int src_step, int src_offset,
                      int cols, int total, __global float * dst)
{
    int lid = get_local_id(0), id = get_global_id(0);

    __local WT localmem[WGS2_ALIGNED];
    WT accumulator = (WT)(0), tmp;

    for ( ; id < total; id += WGS)
    {
        int src_index = mad24(id / cols, src_step, mad24(id % cols, (int)sizeof(T), src_offset));
        __global const T * src = (__global const T *)(srcptr + src_index);

        tmp = convertToWT(src[0]);
#if wdepth == 4
        accumulator = mad24(tmp, tmp, accumulator);
#else
        accumulator = mad(tmp, tmp, accumulator);
#endif
    }

    if (lid < WGS2_ALIGNED)
        localmem[lid] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid >= WGS2_ALIGNED && total >= WGS2_ALIGNED)
        localmem[lid - WGS2_ALIGNED] += accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = WGS2_ALIGNED >> 1; lsize > 0; lsize >>= 1)
    {
        if (lid < lsize)
        {
            int lid2 = lsize + lid;
            localmem[lid] += localmem[lid2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        dst[0] = convertToDT(localmem[0]);
}

#elif defined CCORR

__kernel void matchTemplate_Naive_CCORR(__global const uchar * srcptr, int src_step, int src_offset,
                                        __global const uchar * templateptr, int template_step, int template_offset, int template_rows, int template_cols,
                                        __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        WT sum = (WT)(0);

        __global const T * src = (__global const T *)(srcptr + mad24(y, src_step, mad24(x, (int)sizeof(T), src_offset)));
        __global const T * template = (__global const T *)(templateptr + template_offset);

        for (int i = 0; i < template_rows; ++i)
        {
            for (int j = 0; j < template_cols; ++j)
#if wdepth == 4
                sum = mad24(convertToWT(src[j]), convertToWT(template[j]), sum);
#else
                sum = mad(convertToWT(src[j]), convertToWT(template[j]), sum);
#endif

            src = (__global const T *)((__global const uchar *)src + src_step);
            template = (__global const T *)((__global const uchar *)template + template_step);
        }

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        *(__global float *)(dst + dst_idx) = convertToDT(sum);
    }
}

#elif defined CCORR_NORMED

__kernel void matchTemplate_CCORR_NORMED(__global const uchar * src_sqsums, int src_sqsums_step, int src_sqsums_offset,
                                         __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                         int template_rows, int template_cols, __global const float * template_sqsum)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        __global const float * sqsum = (__global const float *)(src_sqsums);

        src_sqsums_step /= sizeof(float);
        src_sqsums_offset /= sizeof(float);
        float image_sqsum_ = (float)(sqsum[SQSUMS_PTR(template_cols, template_rows)] - sqsum[SQSUMS_PTR(template_cols, 0)] -
                                     sqsum[SQSUMS_PTR(0, template_rows)] + sqsum[SQSUMS_PTR(0, 0)]);

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        __global float * dstult = (__global float *)(dst + dst_idx);
        *dstult = normAcc(*dstult, sqrt(image_sqsum_ * template_sqsum[0]));
    }
}

#elif defined SQDIFF

__kernel void matchTemplate_Naive_SQDIFF(__global const uchar * srcptr, int src_step, int src_offset,
                                         __global const uchar * templateptr, int template_step, int template_offset, int template_rows, int template_cols,
                                         __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        __global const T * src = (__global const T *)(srcptr + mad24(y, src_step, mad24(x, (int)sizeof(T), src_offset)));
        __global const T * template = (__global const T *)(templateptr + template_offset);

        WT sum = (WT)(0), value;

        for (int i = 0; i < template_rows; ++i)
        {
            for (int j = 0; j < template_cols; ++j)
            {
                value = convertToWT(src[j]) - convertToWT(template[j]);
#if wdepth == 4
                sum = mad24(value, value, sum);
#else
                sum = mad(value, value, sum);
#endif
            }

            src = (__global const T *)((__global const uchar *)src + src_step);
            template = (__global const T *)((__global const uchar *)template + template_step);
        }

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        *(__global float *)(dst + dst_idx) = convertToDT(sum);
    }
}

#elif defined SQDIFF_NORMED

__kernel void matchTemplate_SQDIFF_NORMED(__global const uchar * src_sqsums, int src_sqsums_step, int src_sqsums_offset,
                                          __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                          int template_rows, int template_cols, __global const float * template_sqsum)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        src_sqsums_step /= sizeof(float);
        src_sqsums_offset /= sizeof(float);

        __global const float * sqsum = (__global const float *)(src_sqsums);
        float image_sqsum_ = (float)(
                                 (sqsum[SQSUMS_PTR(template_cols, template_rows)] - sqsum[SQSUMS_PTR(template_cols, 0)]) -
                                 (sqsum[SQSUMS_PTR(0, template_rows)] - sqsum[SQSUMS_PTR(0, 0)]));
        float template_sqsum_value = template_sqsum[0];

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        __global float * dstult = (__global float *)(dst + dst_idx);
        *dstult = normAcc_SQDIFF(image_sqsum_ - 2.0f * dstult[0] + template_sqsum_value, sqrt(image_sqsum_ * template_sqsum_value));
    }
}

#elif defined CCOEFF

#if cn == 1

__kernel void matchTemplate_Prepared_CCOEFF(__global const uchar * src_sums, int src_sums_step, int src_sums_offset,
                                            __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                            int template_rows, int template_cols, float template_sum)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(src_sums);

        src_sums_step /= ELEM_SIZE;
        src_sums_offset /= ELEM_SIZE;
        float image_sum_ = (float)((sum[SUMS_PTR(template_cols, template_rows)] - sum[SUMS_PTR(template_cols, 0)])-
                              (sum[SUMS_PTR(0, template_rows)] - sum[SUMS_PTR(0, 0)])) * template_sum;

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        __global float * dstult = (__global float *)(dst + dst_idx);
        *dstult -= image_sum_;
    }
}

#elif cn == 2

__kernel void matchTemplate_Prepared_CCOEFF(__global const uchar * src_sums, int src_sums_step, int src_sums_offset,
                                            __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                            int template_rows, int template_cols, float template_sum_0, float template_sum_1)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        src_sums_step /= ELEM_SIZE;
        src_sums_offset /= ELEM_SIZE;

        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(src_sums);

        float image_sum_ = template_sum_0 * (float)((sum[SUMS_PTR(template_cols, template_rows)] - sum[SUMS_PTR(template_cols, 0)])    -(sum[SUMS_PTR(0, template_rows)] - sum[SUMS_PTR(0, 0)]));
        image_sum_ += template_sum_1 * (float)((sum[SUMS_PTR(template_cols, template_rows)+1] - sum[SUMS_PTR(template_cols, 0)+1])-(sum[SUMS_PTR(0, template_rows)+1] - sum[SUMS_PTR(0, 0)+1]));


        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        __global float * dstult = (__global float *)(dst+dst_idx);
        *dstult -= image_sum_;
    }
}

#elif cn == 4

__kernel void matchTemplate_Prepared_CCOEFF(__global const uchar * src_sums, int src_sums_step, int src_sums_offset,
                                            __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                            int template_rows, int template_cols, float template_sum_0, float template_sum_1, float template_sum_2, float template_sum_3)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        src_sums_step /= ELEM_SIZE;
        src_sums_offset /= ELEM_SIZE;

        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(src_sums);

        int c_r = SUMS_PTR(template_cols, template_rows);
        int c_o = SUMS_PTR(template_cols, 0);
        int o_r = SUMS_PTR(0,template_rows);
        int oo = SUMS_PTR(0, 0);

        float image_sum_ = template_sum_0 * (float)((sum[c_r]   - sum[c_o])  -(sum[o_r]   - sum[oo]));
        image_sum_ += template_sum_1 * (float)((sum[c_r+1] - sum[c_o+1])-(sum[o_r+1] - sum[oo+1]));
        image_sum_ += template_sum_2 * (float)((sum[c_r+2] - sum[c_o+2])-(sum[o_r+2] - sum[oo+2]));
        image_sum_ += template_sum_3 * (float)((sum[c_r+3] - sum[c_o+3])-(sum[o_r+3] - sum[oo+3]));

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        __global float * dstult = (__global float *)(dst+dst_idx);
        *dstult -= image_sum_;
    }
}

#else
#error "cn should be 1, 2 or 4"
#endif

#elif defined CCOEFF_NORMED

#if cn == 1

__kernel void matchTemplate_CCOEFF_NORMED(__global const uchar * src_sums, int src_sums_step, int src_sums_offset,
                                          __global const uchar * src_sqsums, int src_sqsums_step, int src_sqsums_offset,
                                          __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                          int t_rows, int t_cols, float weight, float template_sum, float template_sqsum)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        src_sums_offset   /= ELEM_SIZE;
        src_sums_step     /= ELEM_SIZE;
        src_sqsums_step   /= sizeof(float);
        src_sqsums_offset /= sizeof(float);

        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(src_sums);
        __global float * sqsum = (__global float*)(src_sqsums);

        float image_sum_ =  (float)((sum[SUMS_PTR(t_cols, t_rows)] - sum[SUMS_PTR(t_cols, 0)]) -
                                    (sum[SUMS_PTR(0, t_rows)] - sum[SUMS_PTR(0, 0)]));

        float image_sqsum_ = (float)((sqsum[SQSUMS_PTR(t_cols, t_rows)] - sqsum[SQSUMS_PTR(t_cols, 0)]) -
                                     (sqsum[SQSUMS_PTR(0, t_rows)] - sqsum[SQSUMS_PTR(0, 0)]));

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        __global float * dstult = (__global float *)(dst+dst_idx);
        *dstult = normAcc((*dstult) - image_sum_ * template_sum,
                          sqrt(template_sqsum * (image_sqsum_ - weight * image_sum_ * image_sum_)));
    }
}

#elif cn == 2

__kernel void matchTemplate_CCOEFF_NORMED(__global const uchar * src_sums, int src_sums_step, int src_sums_offset,
                                          __global const uchar * src_sqsums, int src_sqsums_step, int src_sqsums_offset,
                                          __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                          int t_rows, int t_cols, float weight, float template_sum_0, float template_sum_1, float template_sqsum)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float sum_[2];
    float sqsum_[2];

    if (x < dst_cols && y < dst_rows)
    {
        src_sums_offset   /= ELEM_SIZE;
        src_sums_step     /= ELEM_SIZE;
        src_sqsums_step   /= sizeof(float);
        src_sqsums_offset /= sizeof(float);

        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(src_sums);
        __global float * sqsum = (__global float*)(src_sqsums);

        sum_[0] =  (float)((sum[SUMS_PTR(t_cols, t_rows)] - sum[SUMS_PTR(t_cols, 0)])-(sum[SUMS_PTR(0, t_rows)] - sum[SUMS_PTR(0, 0)]));
        sum_[1] =  (float)((sum[SUMS_PTR(t_cols, t_rows)+1] - sum[SUMS_PTR(t_cols, 0)+1])-(sum[SUMS_PTR(0, t_rows)+1] - sum[SUMS_PTR(0, 0)+1]));

        sqsum_[0] = (float)((sqsum[SQSUMS_PTR(t_cols, t_rows)] - sqsum[SQSUMS_PTR(t_cols, 0)])-(sqsum[SQSUMS_PTR(0, t_rows)] - sqsum[SQSUMS_PTR(0, 0)]));
        sqsum_[1] = (float)((sqsum[SQSUMS_PTR(t_cols, t_rows)+1] - sqsum[SQSUMS_PTR(t_cols, 0)+1])-(sqsum[SQSUMS_PTR(0, t_rows)+1] - sqsum[SQSUMS_PTR(0, 0)+1]));

        float num = sum_[0]*template_sum_0 + sum_[1]*template_sum_1;

        float denum = sqrt( template_sqsum * (sqsum_[0] - weight * sum_[0]* sum_[0] +
                                         sqsum_[1] - weight * sum_[1]* sum_[1]));

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        __global float * dstult = (__global float *)(dst+dst_idx);
        *dstult = normAcc((*dstult) - num, denum);
    }
}

#elif cn == 4

__kernel void matchTemplate_CCOEFF_NORMED(__global const uchar * src_sums, int src_sums_step, int src_sums_offset,
                                          __global const uchar * src_sqsums, int src_sqsums_step, int src_sqsums_offset,
                                          __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                          int t_rows, int t_cols, float weight,
                                          float template_sum_0, float template_sum_1, float template_sum_2, float template_sum_3,
                                          float template_sqsum)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    float sum_[4];
    float sqsum_[4];

    if (x < dst_cols && y < dst_rows)
    {
        src_sums_offset   /= ELEM_SIZE;
        src_sums_step     /= ELEM_SIZE;
        src_sqsums_step   /= sizeof(float);
        src_sqsums_offset /= sizeof(float);

        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(src_sums);
        __global float * sqsum = (__global float*)(src_sqsums);

        int c_r = SUMS_PTR(t_cols, t_rows);
        int c_o = SUMS_PTR(t_cols, 0);
        int o_r = SUMS_PTR(0, t_rows);
        int o_o = SUMS_PTR(0, 0);

        sum_[0] =  (float)((sum[c_r]   - sum[c_o])  -(sum[o_r]   - sum[o_o ]));
        sum_[1] =  (float)((sum[c_r+1] - sum[c_o+1])-(sum[o_r+1] - sum[o_o +1]));
        sum_[2] =  (float)((sum[c_r+2] - sum[c_o+2])-(sum[o_r+2] - sum[o_o +2]));
        sum_[3] =  (float)((sum[c_r+3] - sum[c_o+3])-(sum[o_r+3] - sum[o_o +3]));

        c_r = SQSUMS_PTR(t_cols, t_rows);
        c_o = SQSUMS_PTR(t_cols, 0);
        o_r = SQSUMS_PTR(0, t_rows);
        o_o = SQSUMS_PTR(0, 0);

        sqsum_[0] = (float)((sqsum[c_r]   - sqsum[c_o])  -(sqsum[o_r]   - sqsum[o_o]));
        sqsum_[1] = (float)((sqsum[c_r+1] - sqsum[c_o+1])-(sqsum[o_r+1] - sqsum[o_o+1]));
        sqsum_[2] = (float)((sqsum[c_r+2] - sqsum[c_o+2])-(sqsum[o_r+2] - sqsum[o_o+2]));
        sqsum_[3] = (float)((sqsum[c_r+3] - sqsum[c_o+3])-(sqsum[o_r+3] - sqsum[o_o+3]));

        float num = sum_[0]*template_sum_0 + sum_[1]*template_sum_1 + sum_[2]*template_sum_2 + sum_[3]*template_sum_3;

        float denum = sqrt( template_sqsum * (
                                sqsum_[0] - weight * sum_[0]* sum_[0] +
                                sqsum_[1] - weight * sum_[1]* sum_[1] +
                                sqsum_[2] - weight * sum_[2]* sum_[2] +
                                sqsum_[3] - weight * sum_[3]* sum_[3] ));

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        __global float * dstult = (__global float *)(dst+dst_idx);
        *dstult = normAcc((*dstult) - num, denum);
    }
}

#else
#error "cn should be 1, 2 or 4"
#endif

#endif
