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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jin Ma jin@multicorewareinc.com
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
//M*/

#if kercn != 3
#define storepix(val, addr)  *(__global T *)(addr) = val
#define TSIZE (int)sizeof(T)
#define scalar scalar_
#else
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#define TSIZE ((int)sizeof(T1)*3)
#define scalar (T)(scalar_.x, scalar_.y, scalar_.z)
#endif

__kernel void setIdentity(__global uchar * srcptr, int src_step, int src_offset, int rows, int cols,
                          ST scalar_)
{
    int x = get_global_id(0);
    int y0 = get_global_id(1) * rowsPerWI;

    if (x < cols)
    {
        int src_index = mad24(y0, src_step, mad24(x, TSIZE, src_offset));

#if kercn == cn
        #pragma unroll
        for (int y = y0, i = 0, y1 = min(rows, y0 + rowsPerWI); i < rowsPerWI; ++y, ++i, src_index += src_step)
            if (y < y1)
                storepix(x == y ? scalar : (T)(0), srcptr + src_index);
#elif kercn == 4 && cn == 1
        if (y0 < rows)
        {
            storepix(x == y0 >> 2 ? (T)(scalar, 0, 0, 0) : (T)(0), srcptr + src_index);
            if (++y0 < rows)
            {
                src_index += src_step;
                storepix(x == y0 >> 2 ? (T)(0, scalar, 0, 0) : (T)(0), srcptr + src_index);

                if (++y0 < rows)
                {
                    src_index += src_step;
                    storepix(x == y0 >> 2 ? (T)(0, 0, scalar, 0) : (T)(0), srcptr + src_index);

                    if (++y0 < rows)
                    {
                        src_index += src_step;
                        storepix(x == y0 >> 2 ? (T)(0, 0, 0, scalar) : (T)(0), srcptr + src_index);
                    }
                }
            }
        }
#else
#error "Incorrect combination of cn && kercn"
#endif
    }
}
