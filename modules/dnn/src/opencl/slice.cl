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

#define Dtype float
#define Dtype4 float4
#define Dtype8 float8

__kernel void slice(__global const Dtype* src,
                    const int src_plane_size,
                    const int dst_plane_size,
                    const int src_cols,
                    const int dst_cols,
                    const int row_offset,
                    const int col_offset,
                    __global Dtype* dst)
{
    unsigned int row_gid = get_group_id(0);
    unsigned int lid = get_local_id(0);
    const __global Dtype *src_read = src + row_gid * 4 * src_plane_size;
    __global Dtype *dst_read = dst + row_gid * 4 * dst_plane_size;
    Dtype4 a0, a1, a2, a3;

    int i = lid;
    while( i < dst_plane_size / 4)
    {
        int row = (4 * i) / dst_cols + row_offset;
        int col = (4 * i) % dst_cols + col_offset;
        int src_index = row * src_cols + col;

        a0 = vload4(0, src_read + src_index);
        a1 = vload4(0, src_read + src_index + src_plane_size);
        a2 = vload4(0, src_read + src_index + 2 * src_plane_size);
        a3 = vload4(0, src_read + src_index + 3 * src_plane_size);

        vstore4(a0, i, dst_read);
        vstore4(a1, i, dst_read + dst_plane_size);
        vstore4(a2, i, dst_read + 2 * dst_plane_size);
        vstore4(a3, i, dst_read + 3 * dst_plane_size);

        i += get_local_size(0);
    }
}
