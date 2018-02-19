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
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
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
//M*/

#define Dtype float
#define Dtype4 float4

__kernel void prior_box(const int nthreads,
                        const Dtype stepX,
                        const Dtype stepY,
                        __global const Dtype* _offsetsX,
                        __global const Dtype* _offsetsY,
                        const int offsetsX_size,
                        __global const Dtype* _widths,
                        __global const Dtype* _heights,
                        const int widths_size,
                        __global Dtype* dst,
                        const int _layerHeight,
                        const int _layerWidth,
                        const int imgHeight,
                        const int imgWidth)
{
    for (int index = get_global_id(0); index < nthreads; index += get_global_size(0))
    {
        int w = index % _layerWidth;
        int h = index / _layerWidth;
        __global Dtype* outputPtr;

        outputPtr = dst + index * 4 * offsetsX_size * widths_size;

        Dtype _boxWidth, _boxHeight;
        Dtype4 vec;
        for (int i = 0; i < widths_size; ++i)
        {
            _boxWidth = _widths[i];
            _boxHeight = _heights[i];
            for (int j = 0; j < offsetsX_size; ++j)
            {
                float center_x = (w + _offsetsX[j]) * stepX;
                float center_y = (h + _offsetsY[j]) * stepY;

                vec.x = (center_x - _boxWidth * 0.5f) / imgWidth;    // xmin
                vec.y = (center_y - _boxHeight * 0.5f) / imgHeight;  // ymin
                vec.z = (center_x + _boxWidth * 0.5f) / imgWidth;    // xmax
                vec.w = (center_y + _boxHeight * 0.5f) / imgHeight;  // ymax
                vstore4(vec, 0, outputPtr);

                outputPtr += 4;
            }
        }
    }
}

__kernel void set_variance(const int nthreads,
                           const int offset,
                           const int variance_size,
                           __global const Dtype* variance,
                           __global Dtype* dst)
{
    for (int index = get_global_id(0); index < nthreads; index += get_global_size(0))
    {
        Dtype4 var_vec;

        if (variance_size == 1)
            var_vec = (Dtype4)(variance[0]);
        else
            var_vec = vload4(0, variance);

        vstore4(var_vec, 0, dst + offset + index * 4);
    }
}
