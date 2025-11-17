// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// @Authors
//    Zhang Ying, zhangying913@gmail.com
//    Pierre Chatelier, pierre@chachatelier.fr

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void threshold_mask(__global const uchar * srcptr, int src_step, int src_offset,
                             __global uchar * dstptr, int dst_step, int dst_offset, int rows, int cols,
                             __global const uchar * maskptr, int mask_step, int mask_offset,
                             T1 thresh, T1 max_val, T1 min_val)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1) * STRIDE_SIZE;

    if (gx < cols)
    {
        int src_index = mad24(gy, src_step, mad24(gx, (int)sizeof(T), src_offset));
        int dst_index = mad24(gy, dst_step, mad24(gx, (int)sizeof(T), dst_offset));
        int mask_index = mad24(gy, mask_step, mad24(gx/CN, (int)sizeof(uchar), mask_offset));

        #pragma unroll
        for (int i = 0; i < STRIDE_SIZE; i++)
        {
            if (gy < rows)
            {
                T sdata = *(__global const T *)(srcptr + src_index);
                const uchar mdata = *(maskptr + mask_index);
                if (mdata != 0)
                {
                    __global T * dst = (__global T *)(dstptr + dst_index);

                    #ifdef THRESH_BINARY
                            dst[0] = sdata > (thresh) ? (T)(max_val) : (T)(0);
                    #elif defined THRESH_BINARY_INV
                            dst[0] = sdata > (thresh) ? (T)(0) : (T)(max_val);
                    #elif defined THRESH_TRUNC
                            dst[0] = clamp(sdata, (T)min_val, (T)(thresh));
                    #elif defined THRESH_TOZERO
                            dst[0] = sdata > (thresh) ? sdata : (T)(0);
                    #elif defined THRESH_TOZERO_INV
                            dst[0] = sdata > (thresh) ? (T)(0) : sdata;
                    #endif
                }
                gy++;
                src_index += src_step;
                dst_index += dst_step;
                mask_index += mask_step;
            }
        }
    }
}
