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

#if cn != 3
#define loadpix(addr) *(__global const uchar_t *)(addr)
#define storepix(val, addr)  *(__global uchar_t *)(addr) = val
#define TSIZE cn
#else
#define loadpix(addr) vload3(0, (__global const uchar *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global uchar *)(addr))
#define TSIZE 3
#endif

#if cn == 1
#define SUM(a) a
#elif cn == 2
#define SUM(a) a.x + a.y
#elif cn == 3
#define SUM(a) a.x + a.y + a.z
#elif cn == 4
#define SUM(a) a.x + a.y + a.z + a.w
#else
#error "cn should be <= 4"
#endif

__kernel void bilateral(__global const uchar * src, int src_step, int src_offset,
                        __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                        __constant float * color_weight, __constant float * space_weight, __constant int * space_ofs)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < dst_rows && x < dst_cols)
    {
        int src_index = mad24(y + radius, src_step, mad24(x + radius, TSIZE, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, TSIZE, dst_offset));

        float_t sum = (float_t)(0.0f);
        float wsum = 0.0f;
        int_t val0 = convert_int_t(loadpix(src + src_index));

        #pragma unroll
        for (int k = 0; k < maxk; k++ )
        {
            int_t val = convert_int_t(loadpix(src + src_index + space_ofs[k]));
            uint_t diff = abs(val - val0);
            float w = space_weight[k] * color_weight[SUM(diff)];
            sum += convert_float_t(val) * (float_t)(w);
            wsum += w;
        }

        storepix(convert_uchar_t(sum / (float_t)(wsum)), dst + dst_index);
    }
}
