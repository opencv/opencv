//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Rock Li, Rock.li@amd.com
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
//

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void LUT_C1( __global const srcT * src, __global const dstT *lut,
      __global dstT *dst,
      int cols1, int rows,
      int src_offset1,
      int lut_offset1,
      int dst_offset1,
      int src_step1, int dst_step1)
{
    int x1 = get_global_id(0);
    int y = get_global_id(1);

    if (x1 < cols1 && y < rows)
    {
        int src_index = mad24(y, src_step1, src_offset1 + x1);
        int dst_index = mad24(y, dst_step1, dst_offset1 + x1);

        dst[dst_index] = lut[lut_offset1 + src[src_index]];
    }
}

__kernel void LUT_C2( __global const srcT * src, __global const dstT *lut,
      __global dstT *dst,
      int cols1, int rows,
      int src_offset1,
      int lut_offset1,
      int dst_offset1,
      int src_step1, int dst_step1)
{
    int x1 = get_global_id(0) << 1;
    int y = get_global_id(1);

    if (x1 < cols1 && y < rows)
    {
        int src_index = mad24(y, src_step1, src_offset1 + x1);
        int dst_index = mad24(y, dst_step1, dst_offset1 + x1);

        dst[dst_index    ] =                  lut[lut_offset1 + (src[src_index    ] << 1)    ];
        dst[dst_index + 1] = x1 + 1 < cols1 ? lut[lut_offset1 + (src[src_index + 1] << 1) + 1] : dst[dst_index + 1];
    }
}

__kernel void LUT_C4( __global const srcT * src, __global const dstT *lut,
      __global dstT *dst,
      int cols1, int rows,
      int src_offset1,
      int lut_offset1,
      int dst_offset1,
      int src_step1, int dst_step1)
{
    int x1 = get_global_id(0) << 2;
    int y = get_global_id(1);

    if (x1 < cols1 && y < rows)
    {
        int src_index = mad24(y, src_step1, src_offset1 + x1);
        int dst_index = mad24(y, dst_step1, dst_offset1 + x1);

        dst[dst_index    ] =                  lut[lut_offset1 + (src[src_index    ] << 2)    ];
        dst[dst_index + 1] = x1 + 1 < cols1 ? lut[lut_offset1 + (src[src_index + 1] << 2) + 1] : dst[dst_index + 1];
        dst[dst_index + 2] = x1 + 2 < cols1 ? lut[lut_offset1 + (src[src_index + 2] << 2) + 2] : dst[dst_index + 2];
        dst[dst_index + 3] = x1 + 3 < cols1 ? lut[lut_offset1 + (src[src_index + 3] << 2) + 3] : dst[dst_index + 3];
    }
}
