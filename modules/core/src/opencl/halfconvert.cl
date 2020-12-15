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

#ifdef HALF_SUPPORT
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16:enable
#endif
#endif

__kernel void
#ifdef FLOAT_TO_HALF
    convertFp16_FP32_to_FP16
#else
    convertFp16_FP16_to_FP32
#endif
(
    __global const uchar * srcptr, int src_step, int src_offset,
    __global uchar * dstptr, int dst_step, int dst_offset,
    int dst_rows, int dst_cols
)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * rowsPerWI;

    if (x < dst_cols)
    {
        int src_index = mad24(y0, src_step, mad24(x, (int)sizeof(srcT), src_offset));
        int dst_index = mad24(y0, dst_step, mad24(x, (int)sizeof(dstT), dst_offset));

        for (int y = y0, y1 = min(dst_rows, y0 + rowsPerWI); y < y1; ++y, src_index += src_step, dst_index += dst_step)
        {
            __global const srcT * src = (__global const srcT *)(srcptr + src_index);
            __global dstT * dst = (__global dstT *)(dstptr + dst_index);

#ifdef FLOAT_TO_HALF
            vstore_half(src[0], 0, dst);
#else
            dst[0] = vload_half(0, src);
#endif
        }
    }
}
