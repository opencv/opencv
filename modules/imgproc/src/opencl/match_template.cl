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

#define DATA_TYPE type
#define DATA_SIZE ((int)sizeof(type))
#define ELEM_TYPE elem_type
#define ELEM_SIZE ((int)sizeof(elem_type))
#define CN cn

#define SQSUMS_PTR(ox, oy) mad24(gidy + oy, img_sqsums_step, (gidx + img_sqsums_offset + ox) * CN)
#define SQSUMS(ox, oy)     mad24(gidy + oy, img_sqsums_step, (gidx*CN + img_sqsums_offset + ox*CN))
#define SUMS_PTR(ox, oy)   mad24(gidy + oy, img_sums_step, (gidx*CN + img_sums_offset + ox*CN))

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

    res_step   /= sizeof(float);
    res_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            __global const ELEM_TYPE * img_ptr = (__global const ELEM_TYPE *)(img + mad24(gidy + i, img_step, gidx*DATA_SIZE + img_offset));
            __global const ELEM_TYPE * tpl_ptr = (__global const ELEM_TYPE *)(tpl + mad24(i, tpl_step, tpl_offset));

            for(j = 0; j < tpl_cols; j ++)

                for (int c = 0; c < CN; c++)

                    sum += (float)(img_ptr[j*CN+c] * tpl_ptr[j*CN+c]);
           
        }
        __global float * result = (__global float *)(res)+res_idx;
        *result = sum;
    }
}

__kernel void matchTemplate_CCORR_NORMED ( __global const uchar * img_sqsums, int img_sqsums_step, int img_sqsums_offset,
                                           __global uchar * res, int res_step, int res_offset, int res_rows, int res_cols,
                                           int tpl_rows, int tpl_cols, ulong tpl_sqsum)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sqsums_step /= sizeof(float);
    img_sqsums_offset /= sizeof(float);
    res_step   /= sizeof(float);
    res_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        __global float * sqsum = (__global float*)(img_sqsums);
        float image_sqsum_ = (float)(
                                 (sqsum[SQSUMS_PTR(tpl_cols, tpl_rows)] - sqsum[SQSUMS_PTR(tpl_cols, 0)]) -
                                 (sqsum[SQSUMS_PTR(0, tpl_rows)] - sqsum[SQSUMS_PTR(0, 0)]));

        __global float * result = (__global float *)(res)+res_idx;
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

    res_step   /= sizeof(float);
    res_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            __global const ELEM_TYPE * img_ptr = (__global const ELEM_TYPE *)(img + mad24(gidy + i, img_step, gidx*DATA_SIZE + img_offset));
            __global const ELEM_TYPE * tpl_ptr = (__global const ELEM_TYPE *)(tpl + mad24(i, tpl_step, tpl_offset));

            for(j = 0; j < tpl_cols; j ++)

                for (int c = 0; c < CN; c++)
                {
                    delta = (float)(img_ptr[j*CN+c] - tpl_ptr[j*CN+c]);
                    sum += delta*delta;
                }
        }
        __global float * result = (__global float *)(res)+res_idx;
        *result = sum;
    }
}

__kernel void matchTemplate_SQDIFF_NORMED ( __global const uchar * img_sqsums, int img_sqsums_step, int img_sqsums_offset,
                                            __global uchar * res, int res_step, int res_offset, int res_rows, int res_cols,
                                            int tpl_rows, int tpl_cols, ulong tpl_sqsum)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sqsums_step /= sizeof(float);
    img_sqsums_offset /= sizeof(float);
    res_step   /= sizeof(float);
    res_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        __global float * sqsum = (__global float*)(img_sqsums);
        float image_sqsum_ = (float)(
                                 (sqsum[SQSUMS_PTR(tpl_cols, tpl_rows)] - sqsum[SQSUMS_PTR(tpl_cols, 0)]) -
                                 (sqsum[SQSUMS_PTR(0, tpl_rows)] - sqsum[SQSUMS_PTR(0, 0)]));

        __global float * result = (__global float *)(res)+res_idx;

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
    res_step   /= sizeof(float);
    res_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);
    float image_sum_ = 0;

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);

        image_sum_ += (float)((sum[SUMS_PTR(tpl_cols, tpl_rows)] - sum[SUMS_PTR(tpl_cols, 0)])-
                              (sum[SUMS_PTR(0, tpl_rows)] - sum[SUMS_PTR(0, 0)])) * tpl_sum;

        __global float * result = (__global float *)(res)+res_idx;
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
    res_step   /= sizeof(float);
    res_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);
    float image_sum_ = 0;

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);

        image_sum_ += tpl_sum_0 * (float)((sum[SUMS_PTR(tpl_cols, tpl_rows)] - sum[SUMS_PTR(tpl_cols, 0)])    -(sum[SUMS_PTR(0, tpl_rows)] - sum[SUMS_PTR(0, 0)]));
        image_sum_ += tpl_sum_1 * (float)((sum[SUMS_PTR(tpl_cols, tpl_rows)+1] - sum[SUMS_PTR(tpl_cols, 0)+1])-(sum[SUMS_PTR(0, tpl_rows)+1] - sum[SUMS_PTR(0, 0)+1]));

        __global float * result = (__global float *)(res)+res_idx;

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
    res_step   /= sizeof(float);
    res_offset /= sizeof(float);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);
    float image_sum_ = 0;

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);

        image_sum_ += tpl_sum_0 * (float)((sum[SUMS_PTR(tpl_cols, tpl_rows)] - sum[SUMS_PTR(tpl_cols, 0)])    -(sum[SUMS_PTR(0, tpl_rows)] - sum[SUMS_PTR(0, 0)]));
        image_sum_ += tpl_sum_1 * (float)((sum[SUMS_PTR(tpl_cols, tpl_rows)+1] - sum[SUMS_PTR(tpl_cols, 0)+1])-(sum[SUMS_PTR(0, tpl_rows)+1] - sum[SUMS_PTR(0, 0)+1]));
        image_sum_ += tpl_sum_2 * (float)((sum[SUMS_PTR(tpl_cols, tpl_rows)+2] - sum[SUMS_PTR(tpl_cols, 0)+2])-(sum[SUMS_PTR(0, tpl_rows)+2] - sum[SUMS_PTR(0, 0)+2]));
        image_sum_ += tpl_sum_3 * (float)((sum[SUMS_PTR(tpl_cols, tpl_rows)+3] - sum[SUMS_PTR(tpl_cols, 0)+3])-(sum[SUMS_PTR(0, tpl_rows)+3] - sum[SUMS_PTR(0, 0)+3]));

        __global float * result = (__global float *)(res)+res_idx;

        *result -= image_sum_;
    }
}

__kernel void matchTemplate_CCOEFF_NORMED_C1 (__global const uchar * img_sums, int img_sums_step, int img_sums_offset,
                                              __global const uchar * img_sqsums, int img_sqsums_step, int img_sqsums_offset,
                                              __global float * res, int res_step, int res_offset, int res_rows, int res_cols,
                                              int t_rows, int t_cols, float weight, float tpl_sum, float tpl_sqsum)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sums_offset   /= ELEM_SIZE;
    img_sums_step     /= ELEM_SIZE;
    img_sqsums_step   /= sizeof(float);
    img_sqsums_offset /= sizeof(float);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);


    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);
        __global float * sqsum = (__global float*)(img_sqsums);

        float image_sum_ =  (float)((sum[SUMS_PTR(t_cols, t_rows)] - sum[SUMS_PTR(t_cols, 0)]) - 
                                    (sum[SUMS_PTR(0, t_rows)] - sum[SUMS_PTR(0, 0)]));

        float image_sqsum_ = (float)((sqsum[SQSUMS_PTR(t_cols, t_rows)] - sqsum[SQSUMS_PTR(t_cols, 0)]) -
                                     (sqsum[SQSUMS_PTR(0, t_rows)] - sqsum[SQSUMS_PTR(0, 0)]));

        __global float * result = (__global float *)(res)+res_idx;

        *result = normAcc((*result) - image_sum_ * tpl_sum,
                          sqrt(tpl_sqsum * (image_sqsum_ - weight * image_sum_ * image_sum_)));
    }
}

__kernel void matchTemplate_CCOEFF_NORMED_C2 (__global const uchar * img_sums, int img_sums_step, int img_sums_offset,
                                              __global const uchar * img_sqsums, int img_sqsums_step, int img_sqsums_offset,
                                              __global float * res, int res_step, int res_offset, int res_rows, int res_cols,
                                              int t_rows, int t_cols, float weight, float tpl_sum_0, float tpl_sum_1, float tpl_sqsum)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sums_offset   /= ELEM_SIZE;
    img_sums_step     /= ELEM_SIZE;
    img_sqsums_step   /= sizeof(float);
    img_sqsums_offset /= sizeof(float);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);


    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    float sum_[2];
    float sqsum_[2];

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);
        __global float * sqsum = (__global float*)(img_sqsums);

        sum_[0] =  (float)((sum[SUMS_PTR(t_cols, t_rows)] - sum[SUMS_PTR(t_cols, 0)])-(sum[SUMS_PTR(0, t_rows)] - sum[SUMS_PTR(0, 0)]));
        sum_[1] =  (float)((sum[SUMS_PTR(t_cols, t_rows)+1] - sum[SUMS_PTR(t_cols, 0)+1])-(sum[SUMS_PTR(0, t_rows)+1] - sum[SUMS_PTR(0, 0)+1]));

        sqsum_[0] = (float)((sqsum[SQSUMS(t_cols, t_rows)] - sqsum[SQSUMS(t_cols, 0)])-(sqsum[SQSUMS(0, t_rows)] - sqsum[SQSUMS(0, 0)]));
        sqsum_[1] = (float)((sqsum[SQSUMS(t_cols, t_rows)+1] - sqsum[SQSUMS(t_cols, 0)+1])-(sqsum[SQSUMS(0, t_rows)+1] - sqsum[SQSUMS(0, 0)+1]));

        float num = sum_[0]*tpl_sum_0 + sum_[1]*tpl_sum_1;

        float denum = sqrt( tpl_sqsum * (sqsum_[0] - weight * sum_[0]* sum_[0] +
                                         sqsum_[1] - weight * sum_[1]* sum_[1]));

        __global float * result = (__global float *)(res)+res_idx;
        *result = normAcc((*result) - num, denum);
    }
}

__kernel void matchTemplate_CCOEFF_NORMED_C4 (__global const uchar * img_sums, int img_sums_step, int img_sums_offset,
                                              __global const uchar * img_sqsums, int img_sqsums_step, int img_sqsums_offset,
                                              __global float * res, int res_step, int res_offset, int res_rows, int res_cols,
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
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);


    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    float sum_[4];
    float sqsum_[4];

    if(gidx < res_cols && gidy < res_rows)
    {
        __global ELEM_TYPE* sum = (__global ELEM_TYPE*)(img_sums);
        __global float * sqsum = (__global float*)(img_sqsums);

        sum_[0] =  (float)((sum[SUMS_PTR(t_cols, t_rows)] - sum[SUMS_PTR(t_cols, 0)])-(sum[SUMS_PTR(0, t_rows)] - sum[SUMS_PTR(0, 0)]));
        sum_[1] =  (float)((sum[SUMS_PTR(t_cols, t_rows)+1] - sum[SUMS_PTR(t_cols, 0)+1])-(sum[SUMS_PTR(0, t_rows)+1] - sum[SUMS_PTR(0, 0)+1]));
        sum_[2] =  (float)((sum[SUMS_PTR(t_cols, t_rows)+2] - sum[SUMS_PTR(t_cols, 0)+2])-(sum[SUMS_PTR(0, t_rows)+2] - sum[SUMS_PTR(0, 0)+2]));
        sum_[3] =  (float)((sum[SUMS_PTR(t_cols, t_rows)+3] - sum[SUMS_PTR(t_cols, 0)+3])-(sum[SUMS_PTR(0, t_rows)+3] - sum[SUMS_PTR(0, 0)+3]));

        sqsum_[0] = (float)((sqsum[SQSUMS(t_cols, t_rows)] - sqsum[SQSUMS(t_cols, 0)])-(sqsum[SQSUMS(0, t_rows)] - sqsum[SQSUMS(0, 0)]));
        sqsum_[1] = (float)((sqsum[SQSUMS(t_cols, t_rows)+1] - sqsum[SQSUMS(t_cols, 0)+1])-(sqsum[SQSUMS(0, t_rows)+1] - sqsum[SQSUMS(0, 0)+1]));
        sqsum_[2] = (float)((sqsum[SQSUMS(t_cols, t_rows)+2] - sqsum[SQSUMS(t_cols, 0)+2])-(sqsum[SQSUMS(0, t_rows)+2] - sqsum[SQSUMS(0, 0)+2]));
        sqsum_[3] = (float)((sqsum[SQSUMS(t_cols, t_rows)+3] - sqsum[SQSUMS(t_cols, 0)+3])-(sqsum[SQSUMS(0, t_rows)+3] - sqsum[SQSUMS(0, 0)+3]));

        float num = sum_[0]*tpl_sum_0 + sum_[1]*tpl_sum_1 + sum_[2]*tpl_sum_2 + sum_[3]*tpl_sum_3;

        float denum = sqrt( tpl_sqsum * (
                                sqsum_[0] - weight * sum_[0]* sum_[0] +
                                sqsum_[1] - weight * sum_[1]* sum_[1] +
                                sqsum_[2] - weight * sum_[2]* sum_[2] +
                                sqsum_[3] - weight * sum_[3]* sum_[3] ));

        __global float * result = (__global float *)(res)+res_idx;
        *result = normAcc((*result) - num, denum);
    }
}

//////////////////////////////////////////// extractFirstChannel/////////////////////////////
__kernel void extractFirstChannel( const __global float4* img, int img_step, int img_offset,
                                   __global float* res, int res_step, int res_offset, int rows, int cols)
{
    img_step   /= sizeof(float4);
    img_offset /= sizeof(float4);
    res_step   /= sizeof(float);
    res_offset /= sizeof(float);

    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    if(gidx < cols && gidy < rows)
    {
        __global const float4 * image = (__global const float4 *)(img) + mad24(gidy, img_step, img_offset + gidx);
        __global float * result = (__global float *)(res)+ mad24(gidy, res_step, res_offset + gidx);
        *result = image[0].x;
    }
}