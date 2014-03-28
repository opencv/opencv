// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#define noconvert

__kernel void sumConvert(__global const uchar * src1ptr, int src1_step, int src1_offset,
                         __global const uchar * src2ptr, int src2_step, int src2_offset,
                         __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                         coeffT scale, coeffT delta)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (y < dst_rows && x < dst_cols)
    {
        int src1_index = mad24(y, src1_step, mad24(x, (int)sizeof(srcT), src1_offset));
        int src2_index = mad24(y, src2_step, mad24(x, (int)sizeof(srcT), src2_offset));
        int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(dstT), dst_offset));

        __global const srcT * src1 = (__global const srcT *)(src1ptr + src1_index);
        __global const srcT * src2 = (__global const srcT *)(src2ptr + src2_index);
        __global dstT * dst = (__global dstT *)(dstptr + dst_index);

#if wdepth <= 4
        dst[0] = convertToDT( mad24((WT)(scale), convertToWT(src1[0]) + convertToWT(src2[0]), (WT)(delta)) );
#else
        dst[0] = convertToDT( mad((WT)(scale), convertToWT(src1[0]) + convertToWT(src2[0]), (WT)(delta)) );
#endif
    }
}
