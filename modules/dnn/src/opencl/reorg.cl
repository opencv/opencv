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

__kernel void reorg(const int count,
                    __global const Dtype* src,
                    const int channels,
                    const int height,
                    const int width,
                    const int reorgStride,
                    __global Dtype* dst)
{
    for (int index = get_global_id(0); index < count; index += get_global_size(0))
    {
        int k = index / (height * width);
        int j = (index - (k * height * width)) / width;
        int i = (index - (k * height * width)) % width;
        int out_c = channels / (reorgStride*reorgStride);
        int c2 = k % out_c;
        int offset = k / out_c;
        int w2 = i*reorgStride + offset % reorgStride;
        int h2 = j*reorgStride + offset / reorgStride;
        int in_index = w2 + width*reorgStride*(h2 + height*reorgStride*c2);
        dst[index] = src[in_index];
    }
}
