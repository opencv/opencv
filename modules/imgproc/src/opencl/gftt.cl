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
//    Zhang Ying, zhangying913@gmail.com
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

__kernel void findCorners(__global const uchar * eigptr, int eig_step, int eig_offset,
#ifdef HAVE_MASK
                          __global const uchar * mask, int mask_step, int mask_offset,
#endif
                          __global const uchar * tmpptr, int tmp_step, int tmp_offset,
                          __global uchar * cornersptr, __global int * counter,
                          int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        ++x, ++y;

        int eig_index = mad24(y, eig_step, eig_offset + x * (int)sizeof(float));
        int tmp_index = mad24(y, tmp_step, tmp_offset + x * (int)sizeof(float));
#ifdef HAVE_MASK
        int mask_index = mad24(y, mask_step, mask_offset + x);
        mask += mask_index;
#endif

        float val = *(__global const float *)(eigptr + eig_index);
        float tmp = *(__global const float *)(tmpptr + tmp_index);

        if (val != 0 && val == tmp
#ifdef HAVE_MASK
            && mask[0] != 0
#endif
            )
        {
            __global float2 * corners = (__global float2 *)(cornersptr + (int)sizeof(float2) * atomic_inc(counter));
            corners[0] = (float2)(val, as_float( (x<<16) | y ));
        }
    }
}
