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

#pragma OPENCL EXTENSION cl_amd_printf : enable

#if defined (DOUBLE_SUPPORT)

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif

#define TYPE_IMAGE_SQSUM double
#else
#define TYPE_IMAGE_SQSUM float
#endif

#ifndef CN4
#define CN4 1
#else
#define CN4 4
#endif

//////////////////////////////////////////////////
// utilities
#define SQSUMS_PTR(ox, oy) mad24(gidy + oy, img_sqsums_step, (gidx + img_sqsums_offset + ox) * CN4)
#define SUMS_PTR(ox, oy) mad24(gidy + oy, img_sums_step, gidx + img_sums_offset + ox)
// normAcc* are accurate normalization routines which make GPU matchTemplate
// consistent with CPU one
float normAcc(float num, float denum)
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

float normAcc_SQDIFF(float num, float denum)
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
//////////////////////////////////////////////////////////////////////
// normalize

__kernel
void normalizeKernel_C1_D0
(
    __global const float * img_sqsums,
    __global float * res,
    ulong tpl_sqsum,
    int res_rows,
    int res_cols,
    int tpl_rows,
    int tpl_cols,
    int img_sqsums_offset,
    int img_sqsums_step,
    int res_offset,
    int res_step
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);
    img_sqsums_step /= sizeof(*img_sqsums);
    img_sqsums_offset /= sizeof(*img_sqsums);
    int res_idx = mad24(gidy, res_step, res_offset + gidx);
    if(gidx < res_cols && gidy < res_rows)
    {
        float image_sqsum_ = (float)(
                                 (img_sqsums[SQSUMS_PTR(tpl_cols, tpl_rows)] - img_sqsums[SQSUMS_PTR(tpl_cols, 0)]) -
                                 (img_sqsums[SQSUMS_PTR(0, tpl_rows)] - img_sqsums[SQSUMS_PTR(0, 0)]));
        res[res_idx] = normAcc(res[res_idx], sqrt(image_sqsum_ * tpl_sqsum));
    }
}

__kernel
void matchTemplate_Prepared_SQDIFF_C1_D0
(
    __global const TYPE_IMAGE_SQSUM * img_sqsums,
    __global float * res,
    ulong tpl_sqsum,
    int res_rows,
    int res_cols,
    int tpl_rows,
    int tpl_cols,
    int img_sqsums_offset,
    int img_sqsums_step,
    int res_offset,
    int res_step
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);
    img_sqsums_step /= sizeof(*img_sqsums);
    img_sqsums_offset /= sizeof(*img_sqsums);
    int res_idx = mad24(gidy, res_step, res_offset + gidx);
    if(gidx < res_cols && gidy < res_rows)
    {
        float image_sqsum_ = (float)(
                                 (img_sqsums[SQSUMS_PTR(tpl_cols, tpl_rows)] - img_sqsums[SQSUMS_PTR(tpl_cols, 0)]) -
                                 (img_sqsums[SQSUMS_PTR(0, tpl_rows)] - img_sqsums[SQSUMS_PTR(0, 0)]));
        res[res_idx] = image_sqsum_ - 2.f * res[res_idx] + tpl_sqsum;
    }
}

__kernel
void matchTemplate_Prepared_SQDIFF_NORMED_C1_D0
(
    __global const float * img_sqsums,
    __global float * res,
    ulong tpl_sqsum,
    int res_rows,
    int res_cols,
    int tpl_rows,
    int tpl_cols,
    int img_sqsums_offset,
    int img_sqsums_step,
    int res_offset,
    int res_step
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);
    img_sqsums_step /= sizeof(*img_sqsums);
    img_sqsums_offset /= sizeof(*img_sqsums);
    int res_idx = mad24(gidy, res_step, res_offset + gidx);
    if(gidx < res_cols && gidy < res_rows)
    {
        float image_sqsum_ = (float)(
                                 (img_sqsums[SQSUMS_PTR(tpl_cols, tpl_rows)] - img_sqsums[SQSUMS_PTR(tpl_cols, 0)]) -
                                 (img_sqsums[SQSUMS_PTR(0, tpl_rows)] - img_sqsums[SQSUMS_PTR(0, 0)]));
        res[res_idx] = normAcc_SQDIFF(image_sqsum_ - 2.f * res[res_idx] + tpl_sqsum,
                                      sqrt(image_sqsum_ * tpl_sqsum));
    }
}

//////////////////////////////////////////////////
// SQDIFF
__kernel
void matchTemplate_Naive_SQDIFF_C1_D0
(
    __global const uchar * img,
    __global const uchar * tpl,
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int img_offset,
    int tpl_offset,
    int res_offset,
    int img_step,
    int tpl_step,
    int res_step
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int i,j;
    int delta;
    int sum = 0;
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);
    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            // get specific rows of img data
            __global const uchar * img_ptr = img + mad24(gidy + i, img_step, gidx + img_offset);
            __global const uchar * tpl_ptr = tpl + mad24(i, tpl_step, tpl_offset);
            for(j = 0; j < tpl_cols; j ++)
            {
                delta = img_ptr[j] - tpl_ptr[j];
                sum   = mad24(delta, delta, sum);
            }
        }
        res[res_idx] = sum;
    }
}

__kernel
void matchTemplate_Naive_SQDIFF_C1_D5
(
    __global const float * img,
    __global const float * tpl,
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int img_offset,
    int tpl_offset,
    int res_offset,
    int img_step,
    int tpl_step,
    int res_step
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int i,j;
    float delta;
    float sum = 0;
    img_step   /= sizeof(*img);
    img_offset /= sizeof(*img);
    tpl_step   /= sizeof(*tpl);
    tpl_offset /= sizeof(*tpl);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            // get specific rows of img data
            __global const float * img_ptr = img + mad24(gidy + i, img_step, gidx + img_offset);
            __global const float * tpl_ptr = tpl + mad24(i, tpl_step, tpl_offset);
            for(j = 0; j < tpl_cols; j ++)
            {
                delta = img_ptr[j] - tpl_ptr[j];
                sum   = mad(delta, delta, sum);
            }
        }
        res[res_idx] = sum;
    }
}

__kernel
void matchTemplate_Naive_SQDIFF_C4_D0
(
    __global const uchar4 * img,
    __global const uchar4 * tpl,
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int img_offset,
    int tpl_offset,
    int res_offset,
    int img_step,
    int tpl_step,
    int res_step
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int i,j;
    int4 delta;
    int4 sum = (int4)(0, 0, 0, 0);
    img_step   /= sizeof(*img);
    img_offset /= sizeof(*img);
    tpl_step   /= sizeof(*tpl);
    tpl_offset /= sizeof(*tpl);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            // get specific rows of img data
            __global const uchar4 * img_ptr = img + mad24(gidy + i, img_step, gidx + img_offset);
            __global const uchar4 * tpl_ptr = tpl + mad24(i, tpl_step, tpl_offset);
            for(j = 0; j < tpl_cols; j ++)
            {
                //delta = convert_int4(img_ptr[j] - tpl_ptr[j]); // this alternative is incorrect
                delta.x = img_ptr[j].x - tpl_ptr[j].x;
                delta.y = img_ptr[j].y - tpl_ptr[j].y;
                delta.z = img_ptr[j].z - tpl_ptr[j].z;
                delta.w = img_ptr[j].w - tpl_ptr[j].w;
                sum   = mad24(delta, delta, sum);
            }
        }
        res[res_idx] = sum.x + sum.y + sum.z + sum.w;
    }
}

__kernel
void matchTemplate_Naive_SQDIFF_C4_D5
(
    __global const float4 * img,
    __global const float4 * tpl,
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int img_offset,
    int tpl_offset,
    int res_offset,
    int img_step,
    int tpl_step,
    int res_step
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int i,j;
    float4 delta;
    float4 sum = (float4)(0, 0, 0, 0);
    img_step   /= sizeof(*img);
    img_offset /= sizeof(*img);
    tpl_step   /= sizeof(*tpl);
    tpl_offset /= sizeof(*tpl);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            // get specific rows of img data
            __global const float4 * img_ptr = img + mad24(gidy + i, img_step, gidx + img_offset);
            __global const float4 * tpl_ptr = tpl + mad24(i, tpl_step, tpl_offset);
            for(j = 0; j < tpl_cols; j ++)
            {
                //delta = convert_int4(img_ptr[j] - tpl_ptr[j]); // this alternative is incorrect
                delta.x = img_ptr[j].x - tpl_ptr[j].x;
                delta.y = img_ptr[j].y - tpl_ptr[j].y;
                delta.z = img_ptr[j].z - tpl_ptr[j].z;
                delta.w = img_ptr[j].w - tpl_ptr[j].w;
                sum   = mad(delta, delta, sum);
            }
        }
        res[res_idx] = sum.x + sum.y + sum.z + sum.w;
    }
}

//////////////////////////////////////////////////
// CCORR
__kernel
void matchTemplate_Naive_CCORR_C1_D0
(
    __global const uchar * img,
    __global const uchar * tpl,
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int img_offset,
    int tpl_offset,
    int res_offset,
    int img_step,
    int tpl_step,
    int res_step
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int i,j;
    int sum = 0;
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            // get specific rows of img data
            __global const uchar * img_ptr = img + mad24(gidy + i, img_step, gidx + img_offset);
            __global const uchar * tpl_ptr = tpl + mad24(i, tpl_step, tpl_offset);
            for(j = 0; j < tpl_cols; j ++)
            {
                sum = mad24(convert_int(img_ptr[j]), convert_int(tpl_ptr[j]), sum);
            }
        }
        res[res_idx] = (float)sum;
    }
}

__kernel
void matchTemplate_Naive_CCORR_C1_D5
(
    __global const float * img,
    __global const float * tpl,
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int img_offset,
    int tpl_offset,
    int res_offset,
    int img_step,
    int tpl_step,
    int res_step
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int i,j;
    float sum = 0;
    img_step   /= sizeof(*img);
    img_offset /= sizeof(*img);
    tpl_step   /= sizeof(*tpl);
    tpl_offset /= sizeof(*tpl);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            // get specific rows of img data
            __global const float * img_ptr = img + mad24(gidy + i, img_step, gidx + img_offset);
            __global const float * tpl_ptr = tpl + mad24(i, tpl_step, tpl_offset);
            for(j = 0; j < tpl_cols; j ++)
            {
                sum = mad(img_ptr[j], tpl_ptr[j], sum);
            }
        }
        res[res_idx] = sum;
    }
}

__kernel
void matchTemplate_Naive_CCORR_C4_D0
(
    __global const uchar4 * img,
    __global const uchar4 * tpl,
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int img_offset,
    int tpl_offset,
    int res_offset,
    int img_step,
    int tpl_step,
    int res_step
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int i,j;
    int4 sum = (int4)(0, 0, 0, 0);
    img_step   /= sizeof(*img);
    img_offset /= sizeof(*img);
    tpl_step   /= sizeof(*tpl);
    tpl_offset /= sizeof(*tpl);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            // get specific rows of img data
            __global const uchar4 * img_ptr = img + mad24(gidy + i, img_step, gidx + img_offset);
            __global const uchar4 * tpl_ptr = tpl + mad24(i, tpl_step, tpl_offset);
            for(j = 0; j < tpl_cols; j ++)
            {
                sum   = mad24(convert_int4(img_ptr[j]), convert_int4(tpl_ptr[j]), sum);
            }
        }
        res[res_idx] = (float)(sum.x + sum.y + sum.z + sum.w);
    }
}

__kernel
void matchTemplate_Naive_CCORR_C4_D5
(
    __global const float4 * img,
    __global const float4 * tpl,
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int img_offset,
    int tpl_offset,
    int res_offset,
    int img_step,
    int tpl_step,
    int res_step
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    int i,j;
    float4 sum = (float4)(0, 0, 0, 0);
    img_step   /= sizeof(*img);
    img_offset /= sizeof(*img);
    tpl_step   /= sizeof(*tpl);
    tpl_offset /= sizeof(*tpl);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        for(i = 0; i < tpl_rows; i ++)
        {
            // get specific rows of img data
            __global const float4 * img_ptr = img + mad24(gidy + i, img_step, gidx + img_offset);
            __global const float4 * tpl_ptr = tpl + mad24(i, tpl_step, tpl_offset);
            for(j = 0; j < tpl_cols; j ++)
            {
                sum = mad(convert_float4(img_ptr[j]), convert_float4(tpl_ptr[j]), sum);
            }
        }
        res[res_idx] = sum.x + sum.y + sum.z + sum.w;
    }
}

//////////////////////////////////////////////////
// CCOFF
__kernel
void matchTemplate_Prepared_CCOFF_C1_D0
(
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int res_offset,
    int res_step,
    __global const uint * img_sums,
    int img_sums_offset,
    int img_sums_step,
    float tpl_sum
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sums_offset   /= sizeof(*img_sums);
    img_sums_step     /= sizeof(*img_sums);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        float sum = (float)((img_sums[SUMS_PTR(tpl_cols, tpl_rows)] - img_sums[SUMS_PTR(tpl_cols, 0)])
                            -(img_sums[SUMS_PTR(0, tpl_rows)] - img_sums[SUMS_PTR(0, 0)]));
        res[res_idx] -= sum * tpl_sum;
    }
}
__kernel
void matchTemplate_Prepared_CCOFF_C4_D0
(
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int res_offset,
    int res_step,
    __global const uint * img_sums_c0,
    __global const uint * img_sums_c1,
    __global const uint * img_sums_c2,
    __global const uint * img_sums_c3,
    int img_sums_offset,
    int img_sums_step,
    float tpl_sum_c0,
    float tpl_sum_c1,
    float tpl_sum_c2,
    float tpl_sum_c3
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sums_offset   /= sizeof(*img_sums_c0);
    img_sums_step     /= sizeof(*img_sums_c0);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        float ccorr = res[res_idx];
        ccorr -= tpl_sum_c0*(float)(
                     (img_sums_c0[SUMS_PTR(tpl_cols, tpl_rows)] - img_sums_c0[SUMS_PTR(tpl_cols, 0)])
                     - (img_sums_c0[SUMS_PTR(0, tpl_rows)] - img_sums_c0[SUMS_PTR(0, 0)]));
        ccorr -= tpl_sum_c1*(float)(
                     (img_sums_c1[SUMS_PTR(tpl_cols, tpl_rows)] - img_sums_c1[SUMS_PTR(tpl_cols, 0)])
                     - (img_sums_c1[SUMS_PTR(0, tpl_rows)] - img_sums_c1[SUMS_PTR(0, 0)]));
        ccorr -= tpl_sum_c2*(float)(
                     (img_sums_c2[SUMS_PTR(tpl_cols, tpl_rows)] - img_sums_c2[SUMS_PTR(tpl_cols, 0)])
                     - (img_sums_c2[SUMS_PTR(0, tpl_rows)] - img_sums_c2[SUMS_PTR(0, 0)]));
        ccorr -= tpl_sum_c3*(float)(
                     (img_sums_c3[SUMS_PTR(tpl_cols, tpl_rows)] - img_sums_c3[SUMS_PTR(tpl_cols, 0)])
                     - (img_sums_c3[SUMS_PTR(0, tpl_rows)] - img_sums_c3[SUMS_PTR(0, 0)]));
        res[res_idx] = ccorr;
    }
}

__kernel
void matchTemplate_Prepared_CCOFF_NORMED_C1_D0
(
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int res_offset,
    int res_step,
    float weight,
    __global const uint * img_sums,
    int img_sums_offset,
    int img_sums_step,
    __global const float * img_sqsums,
    int img_sqsums_offset,
    int img_sqsums_step,
    float tpl_sum,
    float tpl_sqsum
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sqsums_step   /= sizeof(*img_sqsums);
    img_sqsums_offset /= sizeof(*img_sqsums);
    img_sums_offset   /= sizeof(*img_sums);
    img_sums_step     /= sizeof(*img_sums);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);


    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        float image_sum_ =  (float)(
                                (img_sums[SUMS_PTR(tpl_cols, tpl_rows)] - img_sums[SUMS_PTR(tpl_cols, 0)])
                                - (img_sums[SUMS_PTR(0, tpl_rows)] - img_sums[SUMS_PTR(0, 0)]));

        float image_sqsum_ = (float)(
                                 (img_sqsums[SQSUMS_PTR(tpl_cols, tpl_rows)] - img_sqsums[SQSUMS_PTR(tpl_cols, 0)]) -
                                 (img_sqsums[SQSUMS_PTR(0, tpl_rows)] - img_sqsums[SQSUMS_PTR(0, 0)]));
        res[res_idx] = normAcc(res[res_idx] - image_sum_ * tpl_sum,
                               sqrt(tpl_sqsum * (image_sqsum_ - weight * image_sum_ * image_sum_)));
    }
}
__kernel
void matchTemplate_Prepared_CCOFF_NORMED_C4_D0
(
    __global float * res,
    int img_rows,
    int img_cols,
    int tpl_rows,
    int tpl_cols,
    int res_rows,
    int res_cols,
    int res_offset,
    int res_step,
    float weight,
    __global const uint * img_sums_c0,
    __global const uint * img_sums_c1,
    __global const uint * img_sums_c2,
    __global const uint * img_sums_c3,
    int img_sums_offset,
    int img_sums_step,
    __global const float * img_sqsums_c0,
    __global const float * img_sqsums_c1,
    __global const float * img_sqsums_c2,
    __global const float * img_sqsums_c3,
    int img_sqsums_offset,
    int img_sqsums_step,
    float tpl_sum_c0,
    float tpl_sum_c1,
    float tpl_sum_c2,
    float tpl_sum_c3,
    float tpl_sqsum
)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    img_sqsums_step   /= sizeof(*img_sqsums_c0);
    img_sqsums_offset /= sizeof(*img_sqsums_c0);
    img_sums_offset   /= sizeof(*img_sums_c0);
    img_sums_step     /= sizeof(*img_sums_c0);
    res_step   /= sizeof(*res);
    res_offset /= sizeof(*res);

    int res_idx = mad24(gidy, res_step, res_offset + gidx);

    if(gidx < res_cols && gidy < res_rows)
    {
        float image_sum_c0 =  (float)(
                                  (img_sums_c0[SUMS_PTR(tpl_cols, tpl_rows)] - img_sums_c0[SUMS_PTR(tpl_cols, 0)])
                                  - (img_sums_c0[SUMS_PTR(0, tpl_rows)] - img_sums_c0[SUMS_PTR(0, 0)]));
        float image_sum_c1 =  (float)(
                                  (img_sums_c1[SUMS_PTR(tpl_cols, tpl_rows)] - img_sums_c1[SUMS_PTR(tpl_cols, 0)])
                                  - (img_sums_c1[SUMS_PTR(0, tpl_rows)] - img_sums_c1[SUMS_PTR(0, 0)]));
        float image_sum_c2 =  (float)(
                                  (img_sums_c2[SUMS_PTR(tpl_cols, tpl_rows)] - img_sums_c2[SUMS_PTR(tpl_cols, 0)])
                                  - (img_sums_c2[SUMS_PTR(0, tpl_rows)] - img_sums_c2[SUMS_PTR(0, 0)]));
        float image_sum_c3 =  (float)(
                                  (img_sums_c3[SUMS_PTR(tpl_cols, tpl_rows)] - img_sums_c3[SUMS_PTR(tpl_cols, 0)])
                                  - (img_sums_c3[SUMS_PTR(0, tpl_rows)] - img_sums_c3[SUMS_PTR(0, 0)]));

        float image_sqsum_c0 = (float)(
                                   (img_sqsums_c0[SQSUMS_PTR(tpl_cols, tpl_rows)] - img_sqsums_c0[SQSUMS_PTR(tpl_cols, 0)]) -
                                   (img_sqsums_c0[SQSUMS_PTR(0, tpl_rows)] - img_sqsums_c0[SQSUMS_PTR(0, 0)]));
        float image_sqsum_c1 = (float)(
                                   (img_sqsums_c1[SQSUMS_PTR(tpl_cols, tpl_rows)] - img_sqsums_c1[SQSUMS_PTR(tpl_cols, 0)]) -
                                   (img_sqsums_c1[SQSUMS_PTR(0, tpl_rows)] - img_sqsums_c1[SQSUMS_PTR(0, 0)]));
        float image_sqsum_c2 = (float)(
                                   (img_sqsums_c2[SQSUMS_PTR(tpl_cols, tpl_rows)] - img_sqsums_c2[SQSUMS_PTR(tpl_cols, 0)]) -
                                   (img_sqsums_c2[SQSUMS_PTR(0, tpl_rows)] - img_sqsums_c2[SQSUMS_PTR(0, 0)]));
        float image_sqsum_c3 = (float)(
                                   (img_sqsums_c3[SQSUMS_PTR(tpl_cols, tpl_rows)] - img_sqsums_c3[SQSUMS_PTR(tpl_cols, 0)]) -
                                   (img_sqsums_c3[SQSUMS_PTR(0, tpl_rows)] - img_sqsums_c3[SQSUMS_PTR(0, 0)]));

        float num = res[res_idx] -
                    image_sum_c0 * tpl_sum_c0 -
                    image_sum_c1 * tpl_sum_c1 -
                    image_sum_c2 * tpl_sum_c2 -
                    image_sum_c3 * tpl_sum_c3;
        float denum = sqrt( tpl_sqsum * (
                                image_sqsum_c0 - weight * image_sum_c0 * image_sum_c0 +
                                image_sqsum_c1 - weight * image_sum_c1 * image_sum_c1 +
                                image_sqsum_c2 - weight * image_sum_c2 * image_sum_c2 +
                                image_sqsum_c3 - weight * image_sum_c0 * image_sum_c3)
                          );
        res[res_idx] = normAcc(num, denum);
    }
}

//////////////////////////////////////////////////////////////////////
// extractFirstChannel
__kernel
void extractFirstChannel
(
    const __global float4* img,
    __global float* res,
    int rows,
    int cols,
    int img_offset,
    int res_offset,
    int img_step,
    int res_step
)
{
    img_step   /= sizeof(float4);
    res_step   /= sizeof(float);
    img_offset /= sizeof(float4);
    res_offset /= sizeof(float);
    img += img_offset;
    res += res_offset;
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    if(gidx < cols && gidy < rows)
    {
        res[gidx + gidy * res_step] = img[gidx + gidy * img_step].x;
    }
}
