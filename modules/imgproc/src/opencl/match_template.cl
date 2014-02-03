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
#define CN cn

#define SQSUMS_PTR(ox, oy) mad24(gidy + oy, img_sqsums_step, gidx*CN + img_sqsums_offset + ox*CN)
#define SUMS_PTR(ox, oy) mad24(gidy + oy, img_sums_step,   gidx*CN + img_sums_offset + ox*CN)

inline float normAcc(float num, float denum)
{
    if(fabs(num) < denum)
    {
        return num / denum;
    }
    if(fabs(num) < denum * 1.125f)
    {
        return num > 0 ? 1 : -1;
    }
    return 0;
}

inline float normAcc_SQDIFF(float num, float denum)
{
    if(fabs(num) < denum)
    {
        return num / denum;
    }
    if(fabs(num) < denum * 1.125f)
    {
        return num > 0 ? 1 : -1;
    }
    return 1;
}

//////////////////////////////////////////CCORR/////////////////////////////////////////////////////////////////////////

__kernel void matchTemplate_Naive_CCORR (__global const uchar * img,int img_step,int img_offset,
                                         __global const uchar * tpl,int tpl_step,int tpl_offset,int tpl_rows, int tpl_cols,
                                         __global uchar * res,int res_step,int res_offset,int res_rows,int res_cols)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int i,j;
    float sum = 0;

    int res_idx = mad24(gidy, res_step, res_offset + gidx * (int)sizeof(float));

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            __global const ELEM_TYPE * img_ptr = (__global const ELEM_TYPE *)(img + mad24(gidy + i, img_step, gidx*DATA_SIZE + img_offset));
            __global const ELEM_TYPE * tpl_ptr = (__global const ELEM_TYPE *)(tpl + mad24(i, tpl_step, tpl_offset));

            for(j = 0; j < tpl_cols; j ++)

#pragma unroll
                for (int c = 0; c < CN; c++)

                    sum += (float)(img_ptr[j*CN+c] * tpl_ptr[j*CN+c]);

        }
        __global float * result = (__global float *)(res+res_idx);
        *result = sum;
    }
}

__kernel void matchTemplate_CCORR_NORMED ( __global const uchar * img_sqsums, int img_sqsums_step, int img_sqsums_offset,
                                           __global uchar * res, int res_step, int res_offset, int res_rows, int res_cols,
                                           int tpl_rows, int tpl_cols, float tpl_sqsum)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sqsums_step /= sizeof(float);
    img_sqsums_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx * (int)sizeof(float));

    if(gidx < res_cols && gidy < res_rows)
    {
        __global float * sqsum = (__global float*)(img_sqsums);
        float image_sqsum_ = (float)(
                                 (sqsum[SQSUMS_PTR(tpl_cols, tpl_rows)] - sqsum[SQSUMS_PTR(tpl_cols, 0)]) -
                                 (sqsum[SQSUMS_PTR(0, tpl_rows)] - sqsum[SQSUMS_PTR(0, 0)]));

        __global float * result = (__global float *)(res+res_idx);
        *result = normAcc(*result, sqrt(image_sqsum_ * tpl_sqsum));
    }
}

////////////////////////////////////////////SQDIFF////////////////////////////////////////////////////////////////////////

__kernel void matchTemplate_Naive_SQDIFF(__global const uchar * img,int img_step,int img_offset,
                                         __global const uchar * tpl,int tpl_step,int tpl_offset,int tpl_rows, int tpl_cols,
                                         __global uchar * res,int res_step,int res_offset,int res_rows,int res_cols)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int i,j;
    float delta;
    float sum = 0;

    int res_idx = mad24(gidy, res_step, res_offset + gidx * (int)sizeof(float));

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            __global const ELEM_TYPE * img_ptr = (__global const ELEM_TYPE *)(img + mad24(gidy + i, img_step, gidx*DATA_SIZE + img_offset));
            __global const ELEM_TYPE * tpl_ptr = (__global const ELEM_TYPE *)(tpl + mad24(i, tpl_step, tpl_offset));

            for(j = 0; j < tpl_cols; j ++)

#pragma unroll
                for (int c = 0; c < CN; c++)
                {
                    delta = (float)(img_ptr[j*CN+c] - tpl_ptr[j*CN+c]);
                    sum += delta*delta;
                }
        }
        __global float * result = (__global float *)(res+res_idx);
        *result = sum;
    }
}

__kernel void matchTemplate_SQDIFF_NORMED ( __global const uchar * img_sqsums, int img_sqsums_step, int img_sqsums_offset,
                                            __global uchar * res, int res_step, int res_offset, int res_rows, int res_cols,
                                            int tpl_rows, int tpl_cols, float tpl_sqsum)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sqsums_step /= sizeof(float);
    img_sqsums_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx * (int)sizeof(float));

    if(gidx < res_cols && gidy < res_rows)
    {
        __global float * sqsum = (__global float*)(img_sqsums);
        float image_sqsum_ = (float)(
                                 (sqsum[SQSUMS_PTR(tpl_cols, tpl_rows)] - sqsum[SQSUMS_PTR(tpl_cols, 0)]) -
                                 (sqsum[SQSUMS_PTR(0, tpl_rows)] - sqsum[SQSUMS_PTR(0, 0)]));

        __global float * result = (__global float *)(res+res_idx);

        *result = normAcc_SQDIFF(image_sqsum_ - 2.f * result[0] + tpl_sqsum, sqrt(image_sqsum_ * tpl_sqsum));
    }
}

////////////////////////////////////////////CCOEFF/////////////////////////////////////////////////////////////////

__kernel void matchTemplate_Prepared_CCOEFF_C1 (__global const uchar * img_sums, int img_sums_step, int img_sums_offset,
                                                  __global uchar * res, int res_step, int res_offset, int res_rows, int res_cols,
                                                  int tpl_rows, int tpl_cols, float tpl_sum)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sums_step /= ELEM_SIZE;
    img_sums_offset /= ELEM_SIZE;

    int res_idx = mad24(gidy, res_step, res_offset + gidx * (int)sizeof(float));
    float image_sum_ = 0;

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);

        image_sum_ += (float)((sum[SUMS_PTR(tpl_cols, tpl_rows)] - sum[SUMS_PTR(tpl_cols, 0)])-
                              (sum[SUMS_PTR(0, tpl_rows)] - sum[SUMS_PTR(0, 0)])) * tpl_sum;

        __global float * result = (__global float *)(res+res_idx);
        *result -= image_sum_;
    }
}

__kernel void matchTemplate_Prepared_CCOEFF_C2 (__global const uchar * img_sums, int img_sums_step, int img_sums_offset,
                                                  __global uchar * res, int res_step, int res_offset, int res_rows, int res_cols,
                                                  int tpl_rows, int tpl_cols, float tpl_sum_0,float tpl_sum_1)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sums_step /= ELEM_SIZE;
    img_sums_offset /= ELEM_SIZE;

    int res_idx = mad24(gidy, res_step, res_offset + gidx * (int)sizeof(float));
    float image_sum_ = 0;

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);

        image_sum_ += tpl_sum_0 * (float)((sum[SUMS_PTR(tpl_cols, tpl_rows)] - sum[SUMS_PTR(tpl_cols, 0)])    -(sum[SUMS_PTR(0, tpl_rows)] - sum[SUMS_PTR(0, 0)]));
        image_sum_ += tpl_sum_1 * (float)((sum[SUMS_PTR(tpl_cols, tpl_rows)+1] - sum[SUMS_PTR(tpl_cols, 0)+1])-(sum[SUMS_PTR(0, tpl_rows)+1] - sum[SUMS_PTR(0, 0)+1]));

        __global float * result = (__global float *)(res+res_idx);

        *result -= image_sum_;
    }
}

__kernel void matchTemplate_Prepared_CCOEFF_C4 (__global const uchar * img_sums, int img_sums_step, int img_sums_offset,
                                                __global uchar * res, int res_step, int res_offset, int res_rows, int res_cols,
                                                int tpl_rows, int tpl_cols, float tpl_sum_0,float tpl_sum_1,float tpl_sum_2,float tpl_sum_3)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sums_step /= ELEM_SIZE;
    img_sums_offset /= ELEM_SIZE;

    int res_idx = mad24(gidy, res_step, res_offset + gidx * (int)sizeof(float));
    float image_sum_ = 0;

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);

        int c_r = SUMS_PTR(tpl_cols, tpl_rows);
        int c_o = SUMS_PTR(tpl_cols, 0);
        int o_r = SUMS_PTR(0,tpl_rows);
        int oo = SUMS_PTR(0, 0);

        image_sum_ += tpl_sum_0 * (float)((sum[c_r]   - sum[c_o])  -(sum[o_r]   - sum[oo]));
        image_sum_ += tpl_sum_1 * (float)((sum[c_r+1] - sum[c_o+1])-(sum[o_r+1] - sum[oo+1]));
        image_sum_ += tpl_sum_2 * (float)((sum[c_r+2] - sum[c_o+2])-(sum[o_r+2] - sum[oo+2]));
        image_sum_ += tpl_sum_3 * (float)((sum[c_r+3] - sum[c_o+3])-(sum[o_r+3] - sum[oo+3]));

        __global float * result = (__global float *)(res+res_idx);

        *result -= image_sum_;
    }
}

__kernel void matchTemplate_CCOEFF_NORMED_C1 (__global const uchar * img_sums, int img_sums_step, int img_sums_offset,
                                              __global const uchar * img_sqsums, int img_sqsums_step, int img_sqsums_offset,
                                              __global uchar * res, int res_step, int res_offset, int res_rows, int res_cols,
                                              int t_rows, int t_cols, float weight, float tpl_sum, float tpl_sqsum)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sums_offset   /= ELEM_SIZE;
    img_sums_step     /= ELEM_SIZE;
    img_sqsums_step   /= sizeof(float);
    img_sqsums_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx * (int)sizeof(float));

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);
        __global float * sqsum = (__global float*)(img_sqsums);

        float image_sum_ =  (float)((sum[SUMS_PTR(t_cols, t_rows)] - sum[SUMS_PTR(t_cols, 0)]) -
                                    (sum[SUMS_PTR(0, t_rows)] - sum[SUMS_PTR(0, 0)]));

        float image_sqsum_ = (float)((sqsum[SQSUMS_PTR(t_cols, t_rows)] - sqsum[SQSUMS_PTR(t_cols, 0)]) -
                                     (sqsum[SQSUMS_PTR(0, t_rows)] - sqsum[SQSUMS_PTR(0, 0)]));

        __global float * result = (__global float *)(res+res_idx);

        *result = normAcc((*result) - image_sum_ * tpl_sum,
                          sqrt(tpl_sqsum * (image_sqsum_ - weight * image_sum_ * image_sum_)));
    }
}

__kernel void matchTemplate_CCOEFF_NORMED_C2 (__global const uchar * img_sums, int img_sums_step, int img_sums_offset,
                                              __global const uchar * img_sqsums, int img_sqsums_step, int img_sqsums_offset,
                                              __global uchar * res, int res_step, int res_offset, int res_rows, int res_cols,
                                              int t_rows, int t_cols, float weight, float tpl_sum_0, float tpl_sum_1, float tpl_sqsum)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sums_offset   /= ELEM_SIZE;
    img_sums_step     /= ELEM_SIZE;
    img_sqsums_step   /= sizeof(float);
    img_sqsums_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx * (int)sizeof(float));

    float sum_[2];
    float sqsum_[2];

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);
        __global float * sqsum = (__global float*)(img_sqsums);

        sum_[0] =  (float)((sum[SUMS_PTR(t_cols, t_rows)] - sum[SUMS_PTR(t_cols, 0)])-(sum[SUMS_PTR(0, t_rows)] - sum[SUMS_PTR(0, 0)]));
        sum_[1] =  (float)((sum[SUMS_PTR(t_cols, t_rows)+1] - sum[SUMS_PTR(t_cols, 0)+1])-(sum[SUMS_PTR(0, t_rows)+1] - sum[SUMS_PTR(0, 0)+1]));

        sqsum_[0] = (float)((sqsum[SQSUMS_PTR(t_cols, t_rows)] - sqsum[SQSUMS_PTR(t_cols, 0)])-(sqsum[SQSUMS_PTR(0, t_rows)] - sqsum[SQSUMS_PTR(0, 0)]));
        sqsum_[1] = (float)((sqsum[SQSUMS_PTR(t_cols, t_rows)+1] - sqsum[SQSUMS_PTR(t_cols, 0)+1])-(sqsum[SQSUMS_PTR(0, t_rows)+1] - sqsum[SQSUMS_PTR(0, 0)+1]));

        float num = sum_[0]*tpl_sum_0 + sum_[1]*tpl_sum_1;

        float denum = sqrt( tpl_sqsum * (sqsum_[0] - weight * sum_[0]* sum_[0] +
                                         sqsum_[1] - weight * sum_[1]* sum_[1]));

        __global float * result = (__global float *)(res+res_idx);
        *result = normAcc((*result) - num, denum);
    }
}

__kernel void matchTemplate_CCOEFF_NORMED_C4 (__global const uchar * img_sums, int img_sums_step, int img_sums_offset,
                                              __global const uchar * img_sqsums, int img_sqsums_step, int img_sqsums_offset,
                                              __global uchar * res, int res_step, int res_offset, int res_rows, int res_cols,
                                              int t_rows, int t_cols, float weight,
                                              float tpl_sum_0,float tpl_sum_1,float tpl_sum_2,float tpl_sum_3,
                                              float tpl_sqsum)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sums_offset   /= ELEM_SIZE;
    img_sums_step     /= ELEM_SIZE;
    img_sqsums_step   /= sizeof(float);
    img_sqsums_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx * (int)sizeof(float));

    float sum_[4];
    float sqsum_[4];

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);
        __global float * sqsum = (__global float*)(img_sqsums);

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

        float num = sum_[0]*tpl_sum_0 + sum_[1]*tpl_sum_1 + sum_[2]*tpl_sum_2 + sum_[3]*tpl_sum_3;

        float denum = sqrt( tpl_sqsum * (
                                sqsum_[0] - weight * sum_[0]* sum_[0] +
                                sqsum_[1] - weight * sum_[1]* sum_[1] +
                                sqsum_[2] - weight * sum_[2]* sum_[2] +
                                sqsum_[3] - weight * sum_[3]* sum_[3] ));

        __global float * result = (__global float *)(res+res_idx);
        *result = normAcc((*result) - num, denum);
    }
}
