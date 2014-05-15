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

__kernel void LUTC4(__global const uchar * srcptr, int src_step, int src_offset,
                  __global const uchar * lutptr, int lut_step, int lut_offset,
                  __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols)
{
    int x = get_global_id(0);
    int y = 2 * get_global_id(1);

    __global const dstT * lut = (__global const dstT *)(lutptr + lut_offset);

    __local dstT lut_l[256 * lcn];
    int init = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
    int step = get_local_size(0) * get_local_size(1);

    for (int i = init; i < 256 * lcn; i += step)
    {
        lut_l[i + 0] = lut[i + 0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < cols && y < rows)
    {
        int src_index = mad24(y, src_step, mad24(x, (int)sizeof(srcT) * dcn, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(dstT) * dcn, dst_offset));

        __global const uchar4 * src = (__global const uchar4 *)(srcptr + src_index);
        int4 idx = convert_int4(src[0]) * lcn + (int4)(0, 1, 2, 3);
        __global dstT * dst = (__global dstT *)(dstptr + dst_index);

        dst[0] = lut_l[idx.x];
        dst[1] = lut_l[idx.y];
        dst[2] = lut_l[idx.z];
        dst[3] = lut_l[idx.w];

        if (y < rows - 1)
        {
            src = (__global const uchar4 *)(srcptr + src_index + src_step);
            idx = convert_int4(src[0]) * lcn + (int4)(0, 1, 2, 3);
            dst = (__global dstT *)(dstptr + dst_index + dst_step);

            dst[0] = lut_l[idx.x];
            dst[1] = lut_l[idx.y];
            dst[2] = lut_l[idx.z];
            dst[3] = lut_l[idx.w];
        }
    }
}

__kernel void LUTC3(__global const uchar * srcptr, int src_step, int src_offset,
                  __global const uchar * lutptr, int lut_step, int lut_offset,
                  __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols)
{
    int x = get_global_id(0);
    int y = 2 * get_global_id(1);

    __global const dstT * lut = (__global const dstT *)(lutptr + lut_offset);

    __local dstT lut_l[256 * lcn];
    int init = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
    int step = get_local_size(0) * get_local_size(1);

    for (int i = init; i < 256 * lcn; i += step)
    {
        lut_l[i + 0] = lut[i + 0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < cols && y < rows)
    {
        int src_index = mad24(y, src_step, mad24(x, (int)sizeof(srcT) * dcn, src_offset));
        int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(dstT) * dcn, dst_offset));

        uchar3 src_pixel = vload3(0, (__global const uchar *)(srcptr + src_index));
        int3 idx = convert_int3(src_pixel) * lcn + (int3)(0, 1, 2);
        __global dstT * dst = (__global dstT *)(dstptr + dst_index);

        dst[0] = lut_l[idx.x];
        dst[1] = lut_l[idx.y];
        dst[2] = lut_l[idx.z];
        if (y < rows - 1)
        {
            uchar3 src_pixel = vload3(0, (__global const uchar *)(srcptr + src_index + src_step));
            idx = convert_int3(src_pixel) * lcn + (int3)(0, 1, 2);
            dst = (__global dstT *)(dstptr + dst_index + dst_step);

            dst[0] = lut_l[idx.x];
            dst[1] = lut_l[idx.y];
            dst[2] = lut_l[idx.z];
        }
    }
}

__kernel void LUT(__global const uchar * srcptr, int src_step, int src_offset,
                  __global const uchar * lutptr, int lut_step, int lut_offset,
                  __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols)
{
    __global const dstT * lut = (__global const dstT *)(lutptr + lut_offset);

    __local dstT lut_l[256 * lcn];
    int init = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
    int step = get_local_size(0) * get_local_size(1);

    for (int i = init; i < 256 * lcn; i += step)
    {
        lut_l[i + 0] = lut[i + 0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = get_global_id(0);
    int y = 2 * get_global_id(1);

    if (x < cols && y < rows)
    {
        int src_index = mad24(y, src_step, mad24(x, (int)sizeof(srcT) * dcn, src_offset));

        __global const srcT * src = (__global const srcT *)(srcptr + src_index);
        __global const dstT * lut = (__global const dstT *)(lutptr + lut_offset);

        int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(dstT) * dcn, dst_offset));
        __global dstT * dst = (__global dstT *)(dstptr + dst_index);

#if lcn == 1
        #pragma unroll
        for (int cn = 0; cn < dcn; ++cn)
            dst[cn] = lut_l[src[cn]];
#else //lcn == scn == dcn
        #pragma unroll
        for (int cn = 0; cn < dcn; ++cn)
            dst[cn] = lut_l[mad24(src[cn], lcn, cn)];
#endif
        if (y < rows - 1)
        {
            src = (__global const srcT *)(srcptr + src_index + src_step);
            dst = (__global dstT *)(dstptr + dst_index + dst_step);

#if lcn == 1
            #pragma unroll
            for (int cn = 0; cn < dcn; ++cn)
                dst[cn] = lut_l[src[cn]];
#else //lcn == scn == dcn
            #pragma unroll
            for (int cn = 0; cn < dcn; ++cn)
                dst[cn] = lut_l[mad24(src[cn], lcn, cn)];
#endif
        }
    }
}
