// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2014, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#if cn != 3
#define loadpix(addr) *(__global const T *)(addr)
#define storepix(val, addr)  *(__global T *)(addr) = val
#define TSIZE (int)sizeof(T)
#else
#define loadpix(addr) vload3(0, (__global const T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#define TSIZE ((int)sizeof(T1)*3)
#endif

__kernel void repeat(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                     __global uchar * dstptr, int dst_step, int dst_offset)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * rowsPerWI;

    if (x < src_cols)
    {
        int src_index = mad24(y0, src_step, mad24(x, (int)sizeof(T), src_offset));
        int dst_index0 = mad24(y0, dst_step, mad24(x, (int)sizeof(T), dst_offset));

        for (int y = y0, y1 = min(src_rows, y0 + rowsPerWI); y < y1; ++y, src_index += src_step, dst_index0 += dst_step)
        {
            T srcelem = loadpix(srcptr + src_index);

            #pragma unroll
            for (int ey = 0; ey < ny; ++ey)
            {
                int dst_index = mad24(ey * src_rows, dst_step, dst_index0);

                #pragma unroll
                for (int ex = 0; ex < nx; ++ex)
                {
                    storepix(srcelem, dstptr + dst_index);
                    dst_index = mad24(src_cols, (int)sizeof(T), dst_index);
                }
            }
        }
    }
}
