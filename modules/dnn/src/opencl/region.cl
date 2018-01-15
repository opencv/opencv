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

__kernel void logistic_activ(const int count,
                             __global const Dtype* src,
                             const int cell_size,
                             __global Dtype* dst)
{
    for (int i = get_global_id(0); i < count; i += get_global_size(0))
    {
        int index = cell_size * i;
        Dtype x = src[index + 4];
        dst[index + 4] = 1.f / (1.f + exp(-x));
    }
}

__kernel void softmax_activ(const int count,
                            __global const Dtype* src,
                            __global const Dtype* biasData,
                            const int cell_size,
                            const int classes,
                            const int classfix,
                            const int rows,
                            const int cols,
                            const int anchors,
                            const float thresh,
                            __global Dtype* dst)
{
    for (int index = get_global_id(0); index < count; index += get_global_size(0))
    {
        int box_index = index * cell_size;
        float largest = -FLT_MAX;
        __global const Dtype *input = src + box_index + 5;
        __global Dtype *output = dst + box_index + 5;

        for (int i = 0; i < classes; ++i)
            largest = fmax(largest, input[i]);

        float sum = 0;
        for (int i = 0; i < classes; ++i)
        {
            float e = exp((input[i] - largest));
            sum += e;
            output[i] = e;
        }

        int y = index / anchors / cols;
        int x = index / anchors % cols;
        int a = index - anchors * (x + y * cols);
        float scale = dst[box_index + 4];
        if (classfix == -1 && scale < .5) scale = 0;

        float v1 = src[box_index + 0];
        float v2 = src[box_index + 1];
        float l1 = 1.f / (1.f + exp(-v1));
        float l2 = 1.f / (1.f + exp(-v2));

        dst[box_index + 0] = (x + l1) / cols;
        dst[box_index + 1] = (y + l2) / rows;
        dst[box_index + 2] = exp(src[box_index + 2]) * biasData[2 * a] / cols;
        dst[box_index + 3] = exp(src[box_index + 3]) * biasData[2 * a + 1] / rows;

        for (int i = 0; i < classes; ++i)
        {
            float prob = scale * output[i] / sum;
            output[i] = (prob > thresh) ? prob : 0;
        }
    }
}
