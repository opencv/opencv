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
//    Jia Haipeng, jiahaipeng95@gmail.com
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

/**************************************PUBLICFUNC*************************************/
#if defined (DOUBLE_SUPPORT)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif

#if defined (DEPTH_0)
#define DATA_TYPE uchar
#endif
#if defined (DEPTH_2)
#define DATA_TYPE ushort
#endif

#define CV_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))
enum
{
    yuv_shift  = 14,
    xyz_shift  = 12,
    R2Y        = 4899,
    G2Y        = 9617,
    B2Y        = 1868,
    BLOCK_SIZE = 256
};

__kernel void RGB2Gray(int cols,int rows,int src_step,int dst_step,int channels,
                       int bidx, __global const DATA_TYPE* src, __global DATA_TYPE* dst)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (y < rows && x < cols)
    {
        int src_idx = y * src_step + x * channels * sizeof(DATA_TYPE);
        int dst_idx = y * dst_step + x * sizeof(DATA_TYPE);
        dst[dst_idx] = (DATA_TYPE)CV_DESCALE((src[src_idx + bidx] * B2Y + src[src_idx + 1] * G2Y + src[src_idx + (bidx^2)] * R2Y), yuv_shift);
    }
}   
