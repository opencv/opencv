
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
//    Wu Zailong, bullet@yeah.net
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
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif

#define INF_FLOAT -88.029694
#define INF_DOUBLE -709.0895657128241 


//////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////LOG/////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void arithm_log_D5(int rows, int cols, int srcStep, int dstStep, int srcOffset, int dstOffset, __global float *src, __global float *dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < cols && y < rows )
    {
      x = x << 2;
      int srcIdx = mad24( y, srcStep, x + srcOffset);
      int dstIdx = mad24( y, dstStep, x + dstOffset);

      float src_data = *((__global float *)((__global char *)src + srcIdx));
      float dst_data = (src_data == 0) ? INF_FLOAT : log(fabs(src_data));

      *((__global float *)((__global char *)dst + dstIdx)) = dst_data;
    }
}

#if defined (DOUBLE_SUPPORT)
__kernel void arithm_log_D6(int rows, int cols, int srcStep, int dstStep, int srcOffset, int dstOffset, __global double *src, __global double *dst)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x < cols && y < rows )
    {
      x = x << 3;
      int srcIdx = mad24( y, srcStep, x + srcOffset);
      int dstIdx = mad24( y, dstStep, x + dstOffset);

      double src_data = *((__global double *)((__global char *)src + srcIdx));
      double dst_data = (src_data == 0) ? INF_DOUBLE : log(fabs(src_data));
      *((__global double *)((__global char *)dst + dstIdx)) = dst_data;

    }
}
#endif
