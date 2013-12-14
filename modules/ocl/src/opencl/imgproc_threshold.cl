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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#ifdef VECTORIZED

__kernel void threshold(__global const T * restrict src, int src_offset, int src_step,
                        __global T * dst, int dst_offset, int dst_step,
                        T thresh, T max_val, int max_index, int rows, int cols)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (gx < cols && gy < rows)
    {
        gx *= VECSIZE;
        int src_index = mad24(gy, src_step, src_offset + gx);
        int dst_index = mad24(gy, dst_step, dst_offset + gx);

#ifdef SRC_ALIGNED
        VT sdata = *((__global VT *)(src + src_index));
#else
        VT sdata = VLOADN(0, src + src_index);
#endif
        VT vthresh = (VT)(thresh);

#ifdef THRESH_BINARY
        VT vecValue = sdata > vthresh ? (VT)max_val : (VT)(0);
#elif defined THRESH_BINARY_INV
        VT vecValue = sdata > vthresh ? (VT)(0) : (VT)max_val;
#elif defined THRESH_TRUNC
        VT vecValue = sdata > vthresh ? (VT)thresh : sdata;
#elif defined THRESH_TOZERO
        VT vecValue = sdata > vthresh ? sdata : (VT)(0);
#elif defined THRESH_TOZERO_INV
        VT vecValue = sdata > vthresh ? (VT)(0) : sdata;
#endif

        if (gx + VECSIZE <= max_index)
#ifdef DST_ALIGNED
            *(__global VT*)(dst + dst_index) = vecValue;
#else
            VSTOREN(vecValue, 0, dst + dst_index);
#endif
        else
        {
            __attribute__(( aligned(sizeof(VT)) )) T array[VECSIZE];
            *((VT*)array) = vecValue;
            #pragma unroll
            for (int i = 0; i < VECSIZE; ++i)
                if (gx + i < max_index)
                    dst[dst_index + i] = array[i];
        }
    }
}

#else

__kernel void threshold(__global const T * restrict src, int src_offset, int src_step,
                        __global T * dst, int dst_offset, int dst_step,
                        T thresh, T max_val, int rows, int cols)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);

    if (gx < cols && gy < rows)
    {
        int src_index = mad24(gy, src_step, src_offset + gx);
        int dst_index = mad24(gy, dst_step, dst_offset + gx);

        T sdata = src[src_index];

#ifdef THRESH_BINARY
        dst[dst_index] = sdata > thresh ? max_val : (T)(0);
#elif defined THRESH_BINARY_INV
        dst[dst_index] = sdata > thresh ? (T)(0) : max_val;
#elif defined THRESH_TRUNC
        dst[dst_index] = sdata > thresh ? thresh : sdata;
#elif defined THRESH_TOZERO
        dst[dst_index] = sdata > thresh ? sdata : (T)(0);
#elif defined THRESH_TOZERO_INV
        dst[dst_index] = sdata > thresh ? (T)(0) : sdata;
#endif
    }
}

#endif
