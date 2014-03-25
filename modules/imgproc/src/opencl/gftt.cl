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

#ifdef OP_MAX_EIGEN_VAL

__kernel void maxEigenVal(__global const uchar * srcptr, int src_step, int src_offset, int cols,
                          int total, __global uchar * dstptr
#ifdef HAVE_MASK
                          , __global const uchar * maskptr, int mask_step, int mask_offset
#endif
                          )
{
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int  id = get_global_id(0);

    __local float localmem_max[WGS2_ALIGNED];
    float maxval = -FLT_MAX;

    for (int grain = groupnum * WGS; id < total; id += grain)
    {
        int src_index = mad24(id / cols, src_step, mad24((id % cols), (int)sizeof(float), src_offset));
#ifdef HAVE_MASK
        int mask_index = mad24(id / cols, mask_step, id % cols + mask_offset);
        if (mask[mask_index])
#endif
            maxval = max(maxval, *(__global const float *)(srcptr + src_index));
    }

    if (lid < WGS2_ALIGNED)
        localmem_max[lid] = maxval;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid >= WGS2_ALIGNED && total >= WGS2_ALIGNED)
        localmem_max[lid - WGS2_ALIGNED] = max(maxval, localmem_max[lid - WGS2_ALIGNED]);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = WGS2_ALIGNED >> 1; lsize > 0; lsize >>= 1)
    {
        if (lid < lsize)
        {
           int lid2 = lsize + lid;
           localmem_max[lid] = max(localmem_max[lid], localmem_max[lid2]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        *(__global float *)(dstptr + (int)sizeof(float) * gid) = localmem_max[0];
}

__kernel void maxEigenValTask(__global float * dst, float qualityLevel)
{
    float maxval = -FLT_MAX;

    #pragma unroll
    for (int x = 0; x < groupnum; ++x)
        maxval = max(maxval, dst[x]);

    dst[0] = maxval * qualityLevel;
}

#elif OP_FIND_CORNERS

#define GET_SRC_32F(_y, _x) *(__global const float *)(eigptr + (_y) * eig_step + (_x) * (int)sizeof(float) )

__kernel void findCorners(__global const uchar * eigptr, int eig_step, int eig_offset,
#ifdef HAVE_MASK
                          __global const uchar * mask, int mask_step, int mask_offset,
#endif
                          __global uchar * cornersptr, __global int * counter,
                          int rows, int cols, __constant float * threshold, int max_corners)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < rows && x < cols
#ifdef HAVE_MASK
            && mask[mad24(y, mask_step, x + mask_offset)]
#endif
        )
    {
        ++x, ++y;
        float val = GET_SRC_32F(y, x);

        if (val > threshold[0])
        {
            float maxVal = val;
            maxVal = max(GET_SRC_32F(y - 1, x - 1), maxVal);
            maxVal = max(GET_SRC_32F(y - 1, x    ), maxVal);
            maxVal = max(GET_SRC_32F(y - 1, x + 1), maxVal);

            maxVal = max(GET_SRC_32F(y    , x - 1), maxVal);
            maxVal = max(GET_SRC_32F(y    , x + 1), maxVal);

            maxVal = max(GET_SRC_32F(y + 1, x - 1), maxVal);
            maxVal = max(GET_SRC_32F(y + 1, x    ), maxVal);
            maxVal = max(GET_SRC_32F(y + 1, x + 1), maxVal);

            if (val == maxVal)
            {
                int ind = atomic_inc(counter);
                if (ind < max_corners)
                {
                    __global float2 * corners = (__global float2 *)(cornersptr + ind * (int)sizeof(float2));

                    // pack and store eigenvalue and its coordinates
                    corners[0].x = val;
                    corners[0].y = as_float(y | (x << 16));
                }
            }
        }
    }
}

#endif
