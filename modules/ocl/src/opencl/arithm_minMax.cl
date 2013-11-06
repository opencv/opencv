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
//    Shengen Yan,yanshengen@gmail.com
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

/**************************************PUBLICFUNC*************************************/

#if defined (DOUBLE_SUPPORT)
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#ifdef DEPTH_5
#define MIN_VAL (-FLT_MAX)
#define MAX_VAL FLT_MAX
#elif defined DEPTH_6
#define MIN_VAL (-DBL_MAX)
#define MAX_VAL DBL_MAX
#endif

/**************************************Array minMax**************************************/

__kernel void arithm_op_minMax(__global const T * src, __global T * dst,
    int cols, int invalid_cols, int offset, int elemnum, int groupnum)
{
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int id = get_global_id(0);

    int idx = offset + id + (id / cols) * invalid_cols;

    __local T localmem_max[128], localmem_min[128];
    T minval = (T)(MAX_VAL), maxval = (T)(MIN_VAL), temp;

    for (int grainSize = groupnum << 8; id < elemnum; id += grainSize)
    {
        idx = offset + id + (id / cols) * invalid_cols;
        temp = src[idx];
        minval = min(minval, temp);
        maxval = max(maxval, temp);
    }

    if (lid > 127)
    {
        localmem_min[lid - 128] = minval;
        localmem_max[lid - 128] = maxval;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 128)
    {
        localmem_min[lid] = min(minval, localmem_min[lid]);
        localmem_max[lid] = max(maxval, localmem_max[lid]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = 64; lsize > 0; lsize >>= 1)
    {
        if (lid < lsize)
        {
            int lid2 = lsize + lid;
            localmem_min[lid] = min(localmem_min[lid], localmem_min[lid2]);
            localmem_max[lid] = max(localmem_max[lid], localmem_max[lid2]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        dst[gid] = localmem_min[0];
        dst[gid + groupnum] = localmem_max[0];
    }
}

__kernel void arithm_op_minMax_mask(__global const T * src, __global T * dst,
    int cols, int invalid_cols, int offset,
    int elemnum, int groupnum,
    const __global uchar * mask, int minvalid_cols, int moffset)
{
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int id = get_global_id(0);

    int idx = offset + id + (id / cols) * invalid_cols;
    int midx = moffset + id + (id / cols) * minvalid_cols;

    __local T localmem_max[128], localmem_min[128];
    T minval = (T)(MAX_VAL), maxval = (T)(MIN_VAL), temp;

    for (int grainSize = groupnum << 8; id < elemnum; id += grainSize)
    {
        idx = offset + id + (id / cols) * invalid_cols;
        midx = moffset + id + (id / cols) * minvalid_cols;

        if (mask[midx])
        {
            temp = src[idx];
            minval = min(minval, temp);
            maxval = max(maxval, temp);
        }
    }

    if (lid > 127)
    {
        localmem_min[lid - 128] = minval;
        localmem_max[lid - 128] = maxval;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < 128)
    {
        localmem_min[lid] = min(minval, localmem_min[lid]);
        localmem_max[lid] = max(maxval, localmem_max[lid]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = 64; lsize > 0; lsize >>= 1)
    {
        if (lid < lsize)
        {
            int lid2 = lsize + lid;
            localmem_min[lid] = min(localmem_min[lid], localmem_min[lid2]);
            localmem_max[lid] = max(localmem_max[lid], localmem_max[lid2]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        dst[gid] = localmem_min[0];
        dst[gid + groupnum] = localmem_max[0];
    }
}
