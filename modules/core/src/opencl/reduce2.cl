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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// In no event shall the copyright holders or contributors be liable for any direct,
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

#if ddepth == 0
#define MIN_VAL 0
#define MAX_VAL 255
#elif ddepth == 1
#define MIN_VAL -128
#define MAX_VAL 127
#elif ddepth == 2
#define MIN_VAL 0
#define MAX_VAL 65535
#elif ddepth == 3
#define MIN_VAL -32768
#define MAX_VAL 32767
#elif ddepth == 4
#define MIN_VAL INT_MIN
#define MAX_VAL INT_MAX
#elif ddepth == 5
#define MIN_VAL (-FLT_MAX)
#define MAX_VAL FLT_MAX
#elif ddepth == 6
#define MIN_VAL (-DBL_MAX)
#define MAX_VAL DBL_MAX
#else
#error "Unsupported depth"
#endif

#define noconvert

#ifdef OCL_CV_REDUCE_SUM
#define INIT_VALUE 0
#define PROCESS_ELEM(acc, value) acc += value
#elif defined(OCL_CV_REDUCE_MAX)
#define INIT_VALUE MIN_VAL
#define PROCESS_ELEM(acc, value) acc = value > acc ? value : acc
#elif defined(OCL_CV_REDUCE_MIN)
#define INIT_VALUE MAX_VAL
#define PROCESS_ELEM(acc, value) acc = value < acc ? value : acc
#elif defined(OCL_CV_REDUCE_AVG)
#error "This operation should be implemented through OCL_CV_REDUCE_SUM"
#else
#error "No operation is specified"
#endif

__kernel void reduce(__global const uchar * srcptr, int src_step, int src_offset, int rows, int cols,
                     __global uchar * dstptr, int dst_step, int dst_offset)
{
#if dim == 0 // reduce to a single row
    int x = get_global_id(0);
    if (x < cols)
    {
        int src_index = x * (int)sizeof(srcT) * cn + src_offset;
        __global dstT * dst = (__global dstT *)(dstptr + dst_offset) + x * cn;
        dstT tmp[cn] = { INIT_VALUE };

        for (int y = 0; y < rows; ++y, src_index += src_step)
        {
            __global const srcT * src = (__global const srcT *)(srcptr + src_index);
            #pragma unroll
            for (int c = 0; c < cn; ++c)
            {
                dstT value = convertToDT(src[c]);
                PROCESS_ELEM(tmp[c], value);
            }
        }

        #pragma unroll
        for (int c = 0; c < cn; ++c)
            dst[c] = tmp[c];
    }
#elif dim == 1 // reduce to a single column
    int y = get_global_id(0);
    if (y < rows)
    {
        int src_index = mad24(y, src_step, src_offset);
        int dst_index = mad24(y, dst_step, dst_offset);

        __global const srcT * src = (__global const srcT *)(srcptr + src_index);
        __global dstT * dst = (__global dstT *)(dstptr + dst_index);
        dstT tmp[cn] = { INIT_VALUE };

        for (int x = 0; x < cols; ++x, src += cn)
        {
            #pragma unroll
            for (int c = 0; c < cn; ++c)
            {
                dstT value = convertToDT(src[c]);
                PROCESS_ELEM(tmp[c], value);
            }
        }

        #pragma unroll
        for (int c = 0; c < cn; ++c)
            dst[c] = tmp[c];
    }
#else
#error "Dims must be either 0 or 1"
#endif
}
