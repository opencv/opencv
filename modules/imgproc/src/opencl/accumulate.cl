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

#define SRC_TSIZE cn * (int)sizeof(srcT1)
#define DST_TSIZE cn * (int)sizeof(dstT1)

#define noconvert

__kernel void accumulate(__global const uchar * srcptr, int src_step, int src_offset,
#ifdef ACCUMULATE_PRODUCT
                         __global const uchar * src2ptr, int src2_step, int src2_offset,
#endif
                         __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols
#ifdef ACCUMULATE_WEIGHTED
                         , dstT1 alpha
#endif
#ifdef HAVE_MASK
                         , __global const uchar * mask, int mask_step, int mask_offset
#endif
                         )
{
    int x = get_global_id(0);
    int y = get_global_id(1) * rowsPerWI;

    if (x < dst_cols)
    {
        int src_index = mad24(y, src_step, mad24(x, SRC_TSIZE, src_offset));
#ifdef HAVE_MASK
        int mask_index = mad24(y, mask_step, mask_offset + x);
        mask += mask_index;
#endif
#ifdef ACCUMULATE_PRODUCT
        int src2_index = mad24(y, src2_step, mad24(x, SRC_TSIZE, src2_offset));
#endif
        int dst_index = mad24(y, dst_step, mad24(x, DST_TSIZE, dst_offset));

        #pragma unroll
        for (int i = 0; i < rowsPerWI; ++i)
            if (y < dst_rows)
            {
                __global const srcT1 * src = (__global const srcT1 *)(srcptr + src_index);
#ifdef ACCUMULATE_PRODUCT
                __global const srcT1 * src2 = (__global const srcT1 *)(src2ptr + src2_index);
#endif
                __global dstT1 * dst = (__global dstT1 *)(dstptr + dst_index);

#ifdef HAVE_MASK
                if (mask[0])
#endif
                    #pragma unroll
                    for (int c = 0; c < cn; ++c)
                    {
#ifdef ACCUMULATE
                        dst[c] += convertToDT(src[c]);
#elif defined ACCUMULATE_SQUARE
                        dstT1 val = convertToDT(src[c]);
                        dst[c] = fma(val, val, dst[c]);
#elif defined ACCUMULATE_PRODUCT
                        dst[c] = fma(convertToDT(src[c]), convertToDT(src2[c]), dst[c]);
#elif defined ACCUMULATE_WEIGHTED
                        dst[c] = fma(1 - alpha, dst[c], src[c] * alpha);
#else
#error "Unknown accumulation type"
#endif
                    }

                src_index += src_step;
#ifdef ACCUMULATE_PRODUCT
                src2_index += src2_step;
#endif
#ifdef HAVE_MASK
                mask += mask_step;
#endif
                dst_index += dst_step;
                ++y;
            }
    }
}
