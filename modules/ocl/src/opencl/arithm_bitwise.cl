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
//    Peng Xiao,    pengxiao@outlook.com
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

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// bitwise_binary //////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

__kernel void arithm_bitwise(__global uchar * src1ptr, int src1_step, int src1_offset,
#ifdef OP_BINARY
                             __global uchar * src2ptr, int src2_step, int src2_offset,
#elif defined HAVE_SCALAR
                             T scalar,
#endif
#ifdef HAVE_MASK
                             __global uchar * mask, int mask_step, int mask_offset,
#endif
                             __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
#ifdef HAVE_MASK
        mask += mad24(y, mask_step, x + mask_offset);
        if (mask[0])
#endif
        {
            int src1_index = mad24(y, src1_step, mad24(x, (int)sizeof(T), src1_offset));
#ifdef OP_BINARY
            int src2_index = mad24(y, src2_step, mad24(x, (int)sizeof(T), src2_offset));
#endif
            int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(T), dst_offset));

            __global const T * src1 = (__global const T *)(src1ptr + src1_index);
#ifdef OP_BINARY
            __global const T * src2 = (__global const T *)(src2ptr + src2_index);
#endif
            __global T * dst = (__global T *)(dstptr + dst_index);

#ifdef OP_BINARY
            dst[0] = src1[0] Operation src2[0];
#elif defined HAVE_SCALAR
            dst[0] = src1[0] Operation scalar;
#else
            dst[0] = Operation src1[0];
#endif
        }
    }
}
