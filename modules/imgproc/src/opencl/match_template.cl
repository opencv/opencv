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

#if cn != 3
#define loadpix(addr) *(__global const T *)(addr)
#define TSIZE (int)sizeof(T)
#else
#define loadpix(addr) vload3(0, (__global const T1 *)(addr))
#define TSIZE ((int)sizeof(T1)*3)
#endif

#define SQSUMS_PTR(ox, oy) mad24(y + oy, src_sqsums_step, mad24(x + ox, cn, src_sqsums_offset))
#define SUMS_PTR(ox, oy) mad24(y + oy, src_sums_step, mad24(x + ox, cn, src_sums_offset))
#define SUMS(ox, oy)    mad24(y+oy, src_sums_step, mad24(x+ox, (int)sizeof(T1)*cn, src_sums_offset))
#define SQ_SUMS(ox, oy) mad24(y+oy, src_sqsums_step, mad24(x+ox, (int)sizeof(T1)*cn, src_sqsums_offset))

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
    return num / denum;
}

#define noconvert

#if cn == 1
#define convertToDT(value) (float)(value)
#elif cn == 2
#define convertToDT(value) (float)(value.x + value.y)
#elif cn == 3
#define convertToDT(value) (float)(value.x + value.y + value.z)
#elif cn == 4
#define convertToDT(value) (float)(value.x + value.y + value.z + value.w)
#else
#error "cn should be 1-4"
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
        int src_index = mad24(id / cols, src_step, mad24(id % cols, TSIZE, src_offset));
        T src = loadpix(srcptr + src_index);

        tmp = convertToWT(src);

        accumulator = mad(tmp, tmp, accumulator);
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

#elif defined FIRST_CHANNEL

__kernel void extractFirstChannel( const __global uchar* img, int img_step, int img_offset,
                                   __global uchar* res, int res_step, int res_offset, int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1)*PIX_PER_WI_Y;

    if(x < cols )
    {
        #pragma unroll
        for (int cy=0; cy < PIX_PER_WI_Y && y < rows; ++cy, ++y)
        {
            T1 image = *(__global const T1*)(img + mad24(y, img_step, mad24(x, (int)sizeof(T1)*cn, img_offset)));;
            int res_idx = mad24(y, res_step, mad24(x, (int)sizeof(float), res_offset));
            *(__global float *)(res + res_idx) = image;
        }
    }
}

#elif defined CCORR

#if cn==1 && PIX_PER_WI_X==4

__kernel void matchTemplate_Naive_CCORR(__global const uchar * srcptr, int src_step, int src_offset,
                                        __global const uchar * templateptr, int template_step, int template_offset, int template_rows, int template_cols,
                                        __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    int x0 = get_global_id(0)*PIX_PER_WI_X;
    int y = get_global_id(1);

    if (y < dst_rows)
    {
        if (x0 + PIX_PER_WI_X <= dst_cols)
        {
            WT sum = (WT)(0);

            int ind = mad24(y, src_step, mad24(x0, (int)sizeof(T1), src_offset));
            __global const T1 * template = (__global const T1*)(templateptr + template_offset);

            for (int i = 0; i < template_rows; ++i)
            {
                for (int j = 0; j < template_cols; ++j)
                {
                    T temp = (T)(template[j]);
                    T src = vload4(0, (__global const T1*)(srcptr + ind + j*(int)sizeof(T1)));

                    sum = mad(convertToWT(src), convertToWT(temp), sum);

                }
            ind += src_step;
            template = (__global const T1 *)((__global const uchar *)template + template_step);
            }

            T temp = (T)(template[0]);
            int dst_idx = mad24(y, dst_step, mad24(x0, (int)sizeof(float), dst_offset));
            *(__global float4 *)(dst + dst_idx) = convert_float4(sum);
        }
        else
        {
            WT1 sum [PIX_PER_WI_X];
            #pragma unroll
            for (int i=0; i < PIX_PER_WI_X; i++) sum[i] = 0;

            __global const T1 * src = (__global const T1 *)(srcptr + mad24(y, src_step, mad24(x0, (int)sizeof(T1), src_offset)));
            __global const T1 * template = (__global const T1 *)(templateptr + template_offset);

            for (int i = 0; i < template_rows; ++i)
            {
                for (int j = 0; j < template_cols; ++j)
                {
                    #pragma unroll
                    for (int cx=0, x = x0; cx < PIX_PER_WI_X && x < dst_cols; ++cx, ++x)
                    {
                        sum[cx] = mad(convertToWT1(src[j+cx]), convertToWT1(template[j]), sum[cx]);
                    }
                }

            src = (__global const T1 *)((__global const uchar *)src + src_step);
            template = (__global const T1 *)((__global const uchar *)template + template_step);
            }

            #pragma unroll
            for (int cx=0; cx < PIX_PER_WI_X && x0 < dst_cols; ++cx, ++x0)
            {
                int dst_idx = mad24(y, dst_step, mad24(x0, (int)sizeof(float), dst_offset));
                *(__global float *)(dst + dst_idx) = convertToDT(sum[cx]);
            }
        }
    }
}

#else

__kernel void matchTemplate_Naive_CCORR(__global const uchar * srcptr, int src_step, int src_offset,
                                        __global const uchar * templateptr, int template_step, int template_offset, int template_rows, int template_cols,
                                        __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        WT sum = (WT)(0);

        for (int i = 0; i < template_rows; ++i)
        {
            for (int j = 0; j < template_cols; ++j)
            {
                T src      = loadpix(srcptr      + mad24(y+i, src_step,    mad24(x+j, TSIZE, src_offset)));
                T template = loadpix(templateptr + mad24(i, template_step, mad24(j, TSIZE, template_offset)));

                sum = mad(convertToWT(src), convertToWT(template), sum);
            }
        }

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        *(__global float *)(dst + dst_idx) = convertToDT(sum);
    }
}
#endif

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
        WT sum = (WT)(0), value;

        for (int i = 0; i < template_rows; ++i)
        {
            for (int j = 0; j < template_cols; ++j)
            {
                T src      = loadpix(srcptr      + mad24(y+i, src_step,    mad24(x+j, TSIZE, src_offset)));
                T template = loadpix(templateptr + mad24(i, template_step, mad24(j, TSIZE, template_offset)));

                value = convertToWT(src) - convertToWT(template);

                sum = mad(value, value, sum);
            }
        }

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        *(__global float *)(dst + dst_idx) = convertToDT(sum);
    }
}

#elif defined SQDIFF_PREPARED

__kernel void matchTemplate_Prepared_SQDIFF(__global const uchar * src_sqsums, int src_sqsums_step, int src_sqsums_offset,
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
        *dstult = image_sqsum_ - 2.0f * dstult[0] + template_sqsum_value;
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
        __global const T* sum = (__global const T*)(src_sums + mad24(y, src_sums_step, mad24(x, (int)sizeof(T), src_sums_offset)));

        int step = src_sums_step/(int)sizeof(T);

        T image_sum = (T)(0), value;

        value = (T)(sum[mad24(template_rows, step, template_cols)] - sum[mad24(template_rows, step, 0)] - sum[template_cols] + sum[0]);

        image_sum = mad(value, template_sum , image_sum);

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        *(__global float *)(dst + dst_idx) -= convertToDT(image_sum);
    }
}

#elif cn==3

__kernel void matchTemplate_Prepared_CCOEFF(__global const uchar * src_sums, int src_sums_step, int src_sums_offset,
                                            __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                            int template_rows, int template_cols, float4 template_sum)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        T image_sum = (T)(0), value, temp_sum;

        temp_sum.x = template_sum.x;
        temp_sum.y = template_sum.y;
        temp_sum.z = template_sum.z;

        value  = vload3(0, (__global const T1 *)(src_sums + SUMS(template_cols, template_rows)));
        value -= vload3(0, (__global const T1 *)(src_sums + SUMS(0, template_rows)));
        value -= vload3(0, (__global const T1 *)(src_sums + SUMS(template_cols, 0)));
        value += vload3(0, (__global const T1 *)(src_sums + SUMS(0, 0)));

        image_sum = mad(value, temp_sum , 0);

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        *(__global float *)(dst + dst_idx) -= convertToDT(image_sum);
    }
}

#elif (cn==2 || cn==4)

__kernel void matchTemplate_Prepared_CCOEFF(__global const uchar * src_sums, int src_sums_step, int src_sums_offset,
                                            __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                            int template_rows, int template_cols, float4 template_sum)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        __global const T* sum = (__global const T*)(src_sums + mad24(y, src_sums_step, mad24(x, (int)sizeof(T), src_sums_offset)));

        int step = src_sums_step/(int)sizeof(T);

        T image_sum = (T)(0), value, temp_sum;

#if cn==2
        temp_sum.x = template_sum.x;
        temp_sum.y = template_sum.y;
#else
        temp_sum = template_sum;
#endif

        value = (sum[mad24(template_rows, step, template_cols)] - sum[mad24(template_rows, step, 0)] - sum[template_cols] + sum[0]);

        image_sum = mad(value, temp_sum , image_sum);

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        *(__global float *)(dst + dst_idx) -= convertToDT(image_sum);
    }
}

#else
#error "cn should be 1-4"
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

    float sum_[2];
    float sqsum_[2];


    if (x < dst_cols && y < dst_rows)
    {
        int step = src_sums_step/(int)sizeof(T);

        __global const T* sum   = (__global const T*)(src_sums + mad24(y, src_sums_step,     mad24(x, (int)sizeof(T), src_sums_offset)));
        __global const T* sqsum = (__global const T*)(src_sqsums + mad24(y, src_sqsums_step, mad24(x, (int)sizeof(T), src_sqsums_offset)));

        T value_sum   = sum[mad24(t_rows, step, t_cols)] - sum[mad24(t_rows, step, 0)] - sum[t_cols] + sum[0];
        T value_sqsum = sqsum[mad24(t_rows, step, t_cols)] - sqsum[mad24(t_rows, step, 0)] - sqsum[t_cols] + sqsum[0];

        float num = convertToDT(mad(value_sum, template_sum, (float)0));

        value_sqsum -= weight * value_sum * value_sum;
        float denum = sqrt(mad(template_sqsum, convertToDT(value_sqsum), (float)0));

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        __global float * dstult = (__global float *)(dst+dst_idx);
        *dstult = normAcc((*dstult) - num, denum);
    }
}

#elif cn==3

__kernel void matchTemplate_CCOEFF_NORMED(__global const uchar * src_sums, int src_sums_step, int src_sums_offset,
                                          __global const uchar * src_sqsums, int src_sqsums_step, int src_sqsums_offset,
                                          __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                          int t_rows, int t_cols, float weight, float4 template_sum, float template_sqsum)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int step = src_sums_step/(int)sizeof(T);

        T temp_sum, value_sum, value_sqsum;

        temp_sum.x = template_sum.x;
        temp_sum.y = template_sum.y;
        temp_sum.z = template_sum.z;

        value_sum  = vload3(0, (__global const T1 *)(src_sums + SUMS(t_cols, t_rows)));
        value_sum -= vload3(0, (__global const T1 *)(src_sums + SUMS(0, t_rows)));
        value_sum -= vload3(0, (__global const T1 *)(src_sums + SUMS(t_cols, 0)));
        value_sum += vload3(0, (__global const T1 *)(src_sums + SUMS(0, 0)));

        value_sqsum  = vload3(0, (__global const T1 *)(src_sqsums + SQ_SUMS(t_cols, t_rows)));
        value_sqsum -= vload3(0, (__global const T1 *)(src_sqsums + SQ_SUMS(0, t_rows)));
        value_sqsum -= vload3(0, (__global const T1 *)(src_sqsums + SQ_SUMS(t_cols, 0)));
        value_sqsum += vload3(0, (__global const T1 *)(src_sqsums + SQ_SUMS(0, 0)));

        float num = convertToDT(mad(value_sum, temp_sum, 0));

        value_sqsum -= weight * value_sum * value_sum;
        float denum = sqrt(mad(template_sqsum, convertToDT(value_sqsum), (float)0));

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        __global float * dstult = (__global float *)(dst+dst_idx);
        *dstult = normAcc((*dstult) - num, denum);
    }
}

#elif (cn==2 || cn==4)

__kernel void matchTemplate_CCOEFF_NORMED(__global const uchar * src_sums, int src_sums_step, int src_sums_offset,
                                          __global const uchar * src_sqsums, int src_sqsums_step, int src_sqsums_offset,
                                          __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                          int t_rows, int t_cols, float weight, float4 template_sum, float template_sqsum)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int step = src_sums_step/(int)sizeof(T);

        T temp_sum;

        __global const T* sum   = (__global const T*)(src_sums + mad24(y, src_sums_step,     mad24(x, (int)sizeof(T), src_sums_offset)));
        __global const T* sqsum = (__global const T*)(src_sqsums + mad24(y, src_sqsums_step, mad24(x, (int)sizeof(T), src_sqsums_offset)));

        T value_sum   = sum[mad24(t_rows, step, t_cols)] - sum[mad24(t_rows, step, 0)] - sum[t_cols] + sum[0];
        T value_sqsum = sqsum[mad24(t_rows, step, t_cols)] - sqsum[mad24(t_rows, step, 0)] - sqsum[t_cols] + sqsum[0];

#if cn==2
        temp_sum.x = template_sum.x;
        temp_sum.y = template_sum.y;
#else
        temp_sum = template_sum;
#endif

        float num = convertToDT(mad(value_sum, temp_sum, 0));

        value_sqsum -= weight * value_sum * value_sum;
        float denum = sqrt(mad(template_sqsum, convertToDT(value_sqsum), (float)0));

        int dst_idx = mad24(y, dst_step, mad24(x, (int)sizeof(float), dst_offset));
        __global float * dstult = (__global float *)(dst+dst_idx);
        *dstult = normAcc((*dstult) - num, denum);
    }
}

#else
#error "cn should be 1-4"
#endif

#endif
