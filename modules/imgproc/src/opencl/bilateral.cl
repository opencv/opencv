//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Rock Li, Rock.li@amd.com
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

__kernel void bilateral(__global const uchar * src, int src_step, int src_offset,
                        __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                        __constant float * color_weight, __constant float * space_weight, __constant int * space_ofs)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < dst_rows && x < dst_cols)
    {
        int src_index = mad24(y + radius, src_step, x + radius + src_offset);
        int dst_index = mad24(y, dst_step, x + dst_offset);
        float sum = 0.f, wsum = 0.f;
        int val0 = convert_int(src[src_index]);

        #pragma unroll
        for (int k = 0; k < maxk; k++ )
        {
            int val = convert_int(src[src_index + space_ofs[k]]);
            float w = space_weight[k] * color_weight[abs(val - val0)];
            sum += (float)(val) * w;
            wsum += w;
        }
        dst[dst_index] = convert_uchar_rtz(sum / wsum + 0.5f);
    }
}
