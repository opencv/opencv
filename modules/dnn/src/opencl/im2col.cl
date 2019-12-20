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
//M*/

__kernel void im2col(__global const T *im_src, int im_src_offset,
                     int channels, int height_inp, int width_inp,
                     int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w,
                     int height_out, int width_out,
                     __global T *im_col, int im_col_offset
                    )
{
    int index = get_global_id(0);
    if (index >= height_out * width_out * channels)
        return;
    int j_out = index % width_out;
    int i_out = (index / width_out) % height_out;
    int c_inp = (index / width_out) / height_out;

    int c_out = c_inp * kernel_h * kernel_w;
    int i_inp = i_out * stride_h - pad_h;
    int j_inp = j_out * stride_w - pad_w;

    im_src += (c_inp * height_inp + i_inp) * width_inp + j_inp + im_src_offset;
    im_col += (c_out * height_out + i_out) * width_out + j_out + im_col_offset;

    for (int ki = 0; ki < kernel_h; ++ki)
        for (int kj = 0; kj < kernel_w; ++kj) {
            int i = i_inp + ki;
            int j = j_inp + kj;
            *im_col = (i >= 0 && j >= 0 && i < height_inp && j < width_inp) ?
                im_src[ki * width_inp + kj] : 0;
            im_col += height_out * width_out;
      }
}
