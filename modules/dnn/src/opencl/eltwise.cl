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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
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
// This software is provided by the copyright holders and contributors "as is" and
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

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

__kernel void op_sum4(__global const Dtype * A,
                      __global const Dtype * B,
                      unsigned int A_col_size,
                      const float coeff1,
                      const float coeff2,
                      __global Dtype * C)
{
    unsigned int row_gid = get_group_id(0);
    unsigned int lid = get_local_id(0);
    const __global Dtype *src0_read = A + row_gid * 4 * A_col_size;
    const __global Dtype *src1_read = B + row_gid * 4 * A_col_size;
    __global Dtype *dst0_read = C + row_gid * 4 * A_col_size;

    Dtype4 a0, a1, a2, a3;
    Dtype4 dot0, dot1, dot2, dot3;
    unsigned int i = lid;
    while( i < A_col_size / 4)
    {
        const Dtype4 b0 = vload4(i, src1_read);
        const Dtype4 b1 = vload4(i, src1_read + A_col_size);
        const Dtype4 b2 = vload4(i, src1_read + 2 * A_col_size);
        const Dtype4 b3 = vload4(i, src1_read + 3 * A_col_size);

#if LOOP == 0
        a0 = vload4(i, src0_read);
        a1 = vload4(i, src0_read + A_col_size);
        a2 = vload4(i, src0_read + 2 * A_col_size);
        a3 = vload4(i, src0_read + 3 * A_col_size);

        dot0 = a0 * (Dtype4)coeff1 + b0 * (Dtype4)coeff2;
        dot1 = a1 * (Dtype4)coeff1 + b1 * (Dtype4)coeff2;
        dot2 = a2 * (Dtype4)coeff1 + b2 * (Dtype4)coeff2;
        dot3 = a3 * (Dtype4)coeff1 + b3 * (Dtype4)coeff2;
#else
        a0 = vload4(i, dst0_read);
        a1 = vload4(i, dst0_read + A_col_size);
        a2 = vload4(i, dst0_read + 2 * A_col_size);
        a3 = vload4(i, dst0_read + 3 * A_col_size);

        dot0 = a0 + b0 * (Dtype4)coeff2;
        dot1 = a1 + b1 * (Dtype4)coeff2;
        dot2 = a2 + b2 * (Dtype4)coeff2;
        dot3 = a3 + b3 * (Dtype4)coeff2;
#endif
        vstore4(dot0, i, dst0_read);
        vstore4(dot1, i, dst0_read + A_col_size);
        vstore4(dot2, i, dst0_read + 2 * A_col_size);
        vstore4(dot3, i, dst0_read + 3 * A_col_size);

        i += get_local_size(0);
    }
}
