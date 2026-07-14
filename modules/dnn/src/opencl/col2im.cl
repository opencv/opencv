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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
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

__kernel void col2im(const int n, __global const T* data_col,
                     const int data_col_offset,
                     const int channels,
                     const int height, const int width,
                     const int height_col, const int width_col,
                     const int coeff_h, const int coeff_w,
                     __global const T* biasvec,
                     const int bias_offset,
                     __global T* data_im,
                     const int data_im_offset)
{
    data_col = data_col + data_col_offset;
    biasvec = biasvec + bias_offset;
    data_im = data_im + data_im_offset;
    int index = get_global_id(0);

    if(index < n)
    {
        T val = 0.f;
        int w = index % width + PAD_W;
        int h = (index / width) % height + PAD_H;
        int c = index / (width * height);
        int h_col_start = (h < KERNEL_H) ? 0 : (h - KERNEL_H) / STRIDE_H + 1;
        int h_col_end = min(h / STRIDE_H + 1, height_col);
        int plane_size_col = height_col * width_col;
        int offset = (c * KERNEL_H * KERNEL_W + h * KERNEL_W + w) * plane_size_col;

        int w_col_start = (w < KERNEL_W) ? 0 : (w - KERNEL_W) / STRIDE_W + 1;
        int w_col_end = min(w / STRIDE_W + 1, width_col);

        for (int h_col = h_col_start; h_col < h_col_end; ++h_col)
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col)
                val += data_col[offset + h_col * coeff_h + w_col * coeff_w];

        data_im[index] = val + biasvec[c];
    }
}
