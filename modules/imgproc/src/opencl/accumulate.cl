// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void accumulate(__global const uchar * srcptr, int src_step, int src_offset,
#ifdef ACCUMULATE_PRODUCT
                         __global const uchar * src2ptr, int src2_step, int src2_offset,
#endif
                         __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols
#ifdef ACCUMULATE_WEIGHTED
                         , dstT alpha
#endif
#ifdef HAVE_MASK
                         , __global const uchar * mask, int mask_step, int mask_offset
#endif
                         )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int src_index = mad24(y, src_step, src_offset + x * cn * (int)sizeof(srcT));
#ifdef HAVE_MASK
        int mask_index = mad24(y, mask_step, mask_offset + x);
        mask += mask_index;
#endif
        int dst_index = mad24(y, dst_step, dst_offset + x * cn * (int)sizeof(dstT));

        __global const srcT * src = (__global const srcT *)(srcptr + src_index);
#ifdef ACCUMULATE_PRODUCT
        int src2_index = mad24(y, src2_step, src2_offset + x * cn * (int)sizeof(srcT));
        __global const srcT * src2 = (__global const srcT *)(src2ptr + src2_index);
#endif
        __global dstT * dst = (__global dstT *)(dstptr + dst_index);

        #pragma unroll
        for (int c = 0; c < cn; ++c)
#ifdef HAVE_MASK
            if (mask[0])
#endif
#ifdef ACCUMULATE
                dst[c] += src[c];
#elif defined ACCUMULATE_SQUARE
                dst[c] += src[c] * src[c];
#elif defined ACCUMULATE_PRODUCT
                dst[c] += src[c] * src2[c];
#elif defined ACCUMULATE_WEIGHTED
                dst[c] = (1 - alpha) * dst[c] + src[c] * alpha;
#else
#error "Unknown accumulation type"
#endif
    }
}
