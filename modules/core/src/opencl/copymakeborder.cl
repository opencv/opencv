//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Zero Lin zero.lin@amd.com
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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#if cn != 3
#define loadpix(addr)  *(__global const T*)(addr)
#define storepix(val, addr)  *(__global T*)(addr) = val
#define TSIZE ((int)sizeof(T))
#define convertScalar(a) (a)
#else
#define loadpix(addr)  vload3(0, (__global const T1*)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1*)(addr))
#define TSIZE ((int)sizeof(T1)*3)
#define convertScalar(a) (T)(a.x, a.y, a.z)
#endif

#ifdef BORDER_CONSTANT
#define EXTRAPOLATE(x, cols) \
    ;
#elif defined BORDER_REPLICATE
#define EXTRAPOLATE(x, cols) \
    x = clamp(x, 0, cols - 1);
#elif defined BORDER_WRAP
#define EXTRAPOLATE(x, cols) \
    { \
        if (x < 0) \
            x -= ((x - cols + 1) / cols) * cols; \
        if (x >= cols) \
            x %= cols; \
    }
#elif defined(BORDER_REFLECT) || defined(BORDER_REFLECT_101)
#ifdef BORDER_REFLECT
#define DELTA int delta = 0
#else
#define DELTA int delta = 1
#endif
#define EXTRAPOLATE(x, cols) \
    { \
        DELTA; \
        if (cols == 1) \
            x = 0; \
        else \
            do \
            { \
                if( x < 0 ) \
                    x = -x - 1 + delta; \
                else \
                    x = cols - 1 - (x - cols) - delta; \
            } \
            while (x >= cols || x < 0); \
    }
#else
#error "No extrapolation method"
#endif

#define NEED_EXTRAPOLATION(x, cols) (x >= cols || x < 0)

__kernel void copyMakeBorder(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                             __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                             int top, int left, ST nVal)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * rowsPerWI;

#ifdef BORDER_CONSTANT
    T scalar = convertScalar(nVal);
#endif

    if (x < dst_cols)
    {
        int src_x = x - left, src_y;
        int dst_index = mad24(y0, dst_step, mad24(x, (int)TSIZE, dst_offset));

        if (NEED_EXTRAPOLATION(src_x, src_cols))
        {
#ifdef BORDER_CONSTANT
            for (int y = y0, y1 = min(y0 + rowsPerWI, dst_rows); y < y1; ++y, dst_index += dst_step)
                storepix(scalar, dstptr + dst_index);
            return;
#endif
            EXTRAPOLATE(src_x, src_cols)
        }
        src_x = mad24(src_x, TSIZE, src_offset);

        for (int y = y0, y1 = min(y0 + rowsPerWI, dst_rows); y < y1; ++y, dst_index += dst_step)
        {
            src_y = y - top;
            if (NEED_EXTRAPOLATION(src_y, src_rows))
            {
                EXTRAPOLATE(src_y, src_rows)
#ifdef BORDER_CONSTANT
                storepix(scalar, dstptr + dst_index);
                continue;
#endif
            }
            int src_index = mad24(src_y, src_step, src_x);
            storepix(loadpix(srcptr + src_index), dstptr + dst_index);
        }
    }
}
