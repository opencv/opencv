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
//    Jin Ma jin@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
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
#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif


#if defined (DOUBLE_SUPPORT)
#define DATA_TYPE double
#else
#define DATA_TYPE float
#endif

__kernel void setIdentityKernel_F1(__global float* src, int src_row, int src_col, int src_step, DATA_TYPE scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < src_col && y < src_row)
    {
        if(x == y)
            src[y * src_step + x] = scalar;
        else
            src[y * src_step + x] = 0 * scalar;
    }
}

__kernel void setIdentityKernel_D1(__global DATA_TYPE* src, int src_row, int src_col, int src_step, DATA_TYPE scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < src_col && y < src_row)
    {
        if(x == y)
            src[y * src_step + x] = scalar;
        else
            src[y * src_step + x] = 0 * scalar;
    }
}

__kernel void setIdentityKernel_I1(__global int* src, int src_row, int src_col, int src_step, int scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < src_col && y < src_row)
    {
        if(x == y)
            src[y * src_step + x] = scalar;
        else
            src[y * src_step + x] = 0 * scalar;
    }
}
