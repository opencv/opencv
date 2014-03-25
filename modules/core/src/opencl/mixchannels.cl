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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
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
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#define DECLARE_INPUT_MAT(i) \
    __global const uchar * src##i##ptr, int src##i##_step, int src##i##_offset,
#define DECLARE_OUTPUT_MAT(i) \
    __global uchar * dst##i##ptr, int dst##i##_step, int dst##i##_offset,
#define PROCESS_ELEM(i) \
    int src##i##_index = mad24(src##i##_step, y, mad24(x, (int)sizeof(T) * scn##i, src##i##_offset)); \
    __global const T * src##i = (__global const T *)(src##i##ptr + src##i##_index); \
    int dst##i##_index = mad24(dst##i##_step, y, mad24(x, (int)sizeof(T) * dcn##i, dst##i##_offset)); \
    __global T * dst##i = (__global T *)(dst##i##ptr + dst##i##_index); \
    dst##i[0] = src##i[0];

__kernel void mixChannels(DECLARE_INPUT_MATS DECLARE_OUTPUT_MATS int rows, int cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        PROCESS_ELEMS
    }
}
