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
//    Jiang Liyuan, jlyuan001.good@163.com
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

/************************************** convolve **************************************/

__kernel void convolve_D5(__global float *src, __global float *temp1, __global float *dst,
                          int rows, int cols, int src_step, int dst_step,int k_step, int kWidth, int kHeight,
                          int src_offset, int dst_offset, int koffset)
{
    __local float smem[16 + 2 * 8][16 + 2 * 8];

    int x = get_local_id(0);
    int y = get_local_id(1);
    int gx = get_global_id(0);
    int gy = get_global_id(1);

            // x | x 0 | 0
            // -----------
            // x | x 0 | 0
            // 0 | 0 0 | 0
            // -----------
            // 0 | 0 0 | 0
    smem[y][x] = src[min(max(gy - 8, 0), rows - 1) * src_step + min(max(gx - 8, 0), cols - 1) + src_offset];

            // 0 | 0 x | x
            // -----------
            // 0 | 0 x | x
            // 0 | 0 0 | 0
            // -----------
            // 0 | 0 0 | 0
    smem[y][x + 16] = src[min(max(gy - 8, 0), rows - 1) * src_step + min(gx + 8, cols - 1) + src_offset];

            // 0 | 0 0 | 0
            // -----------
            // 0 | 0 0 | 0
            // x | x 0 | 0
            // -----------
            // x | x 0 | 0
    smem[y + 16][x] = src[min(gy + 8, rows - 1) * src_step + min(max(gx - 8, 0), cols - 1) + src_offset];

            // 0 | 0 0 | 0
            // -----------
            // 0 | 0 0 | 0
            // 0 | 0 x | x
            // -----------
            // 0 | 0 x | x
    smem[y + 16][x + 16] = src[min(gy + 8, rows - 1) * src_step + min(gx + 8, cols - 1) + src_offset];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx < cols && gy < rows)
    {
        float res = 0;

        for (int i = 0; i < kHeight; ++i)
            for (int j = 0; j < kWidth; ++j)
                res += smem[y + 8 - kHeight / 2 + i][x + 8 - kWidth / 2 + j] * temp1[i * k_step + j + koffset];

        dst[gy * dst_step + gx + dst_offset] = res;
    }
}
