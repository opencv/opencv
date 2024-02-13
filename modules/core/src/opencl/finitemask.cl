// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This kernel is compiled with the following possible defines:
//  - srcT, cn: source type and number of channels per pixel
//  - rowsPerWI: Intel GPU optimization
//  - DOUBLE_SUPPORT: enable double support if available

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void finiteMask(__global const uchar * srcptr, int srcstep, int srcoffset,
                         __global uchar * dstptr, int dststep, int dstoffset,
                         int rows, int cols )
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * rowsPerWI;

    if (x < cols)
    {
        int src_index = mad24(y0, srcstep, mad24(x, (int)sizeof(srcT) * cn, srcoffset));
        int dst_index = mad24(y0, dststep, x + dstoffset);

        for (int y = y0, y1 = min(rows, y0 + rowsPerWI); y < y1; ++y, src_index += srcstep, dst_index += dststep)
        {
            bool vfinite = true;

            for (int c = 0; c < cn; c++)
            {
                srcT val = *(__global srcT *)(srcptr + src_index + c * (int)sizeof(srcT));

                vfinite = vfinite && !isnan(val) & !isinf(val);
            }

            *(dstptr + dst_index) = vfinite ? 255 : 0;
        }
    }
}