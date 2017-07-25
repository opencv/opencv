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

#ifndef _OPENCV_GREENTEA_MATH_FUNCTIONS_HPP_
#define _OPENCV_GREENTEA_MATH_FUNCTIONS_HPP_
#include "../../precomp.hpp"
#include "common.hpp"


#ifdef HAVE_OPENCL
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

template<typename Dtype>
void libdnnGEMM(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                const CBLAS_TRANSPOSE TransB, const int_tp M,
                const int_tp N, const int_tp K, const Dtype alpha,
                const cl_mem A, const int_tp offA, const cl_mem B,
                const int_tp offB, const Dtype beta, cl_mem C,
                const int_tp offC , const bool is_image_a = false,
                const bool is_image_b = false);

void libdnnGEMMCopyBufferToImage(int_tp ctx_id,
                                 cl_mem *image, cl_mem buffer, int offset,
                                 bool is_matrix_a, bool transpose,
                                 bool padding, int padded_height,
                                 int padded_width, int height,
                                 int width, int wait_list_size,
                                 cl_event *wait_list, cl_event *event);

template<typename Dtype>
void libdnnGEMV(const int_tp ctx_id, const CBLAS_TRANSPOSE TransA,
                const int_tp M, const int_tp N, const Dtype alpha,
                const cl_mem A, const int_tp offA, const cl_mem x,
                const int_tp offx, const Dtype beta, cl_mem y,
                const int_tp offy);

template<typename Dtype>
void libdnnAXPY(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                const cl_mem x, const int_tp offx, cl_mem y,
                const int_tp offy);

template<typename Dtype>
void libdnnSet(const int_tp ctx_id, const int_tp N, const Dtype alpha,
                      cl_mem Y, const int_tp offY);
#endif  // HAVE_OPENCL
#endif
