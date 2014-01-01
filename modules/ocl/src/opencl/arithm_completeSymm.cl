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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// Authors:
//  * Peter Andreas Entschev, peter@entschev.com
//
//M*/

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

__kernel void completeSymm(__global T * src, int src_step, int src_offset,
                           int cols, int rows, int lower_to_upper)
                           //int cols, int rows)
{
    int i = get_global_id(0) + 1;
    int x, y;
    //int lower_to_upper = 0;

    if (i <= (cols*cols-cols)/2)
    {
        // Calculates row and column by natural number series properties
        // and triangular root of a number
        float base = 8.f*(float)i + 1.f;
        float fn = (sqrt(base) - 1.f) / 2.f;
        int n = (int)round(fn);
        if (i > (n*n+n)/2)
            n++;

        int m = i + n - (n*n+n)/2 - 1;

        int src_idx, dst_idx;

        if (lower_to_upper == 0)
        {
            x = n;
            y = m;
        }
        else
        {
            x = m;
            y = n;
        }

        src_idx = mad24(y, src_step, src_offset + x);
        dst_idx = mad24(x, src_step, src_offset + y);

        src[dst_idx] = src[src_idx];
    }
}
