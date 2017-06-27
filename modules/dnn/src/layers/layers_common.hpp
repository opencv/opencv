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

#ifndef __OPENCV_DNN_LAYERS_LAYERS_COMMON_HPP__
#define __OPENCV_DNN_LAYERS_LAYERS_COMMON_HPP__
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

void getConvolutionKernelParams(const LayerParams &params, int &kernelH, int &kernelW, int &padH, int &padW,
                                int &strideH, int &strideW, int &dilationH, int &dilationW, cv::String& padMode);

void getPoolingKernelParams(const LayerParams &params, int &kernelH, int &kernelW, bool &globalPooling,
                            int &padH, int &padW, int &strideH, int &strideW, cv::String& padMode);

void getConvPoolOutParams(const Size& inp, const Size &kernel,
                          const Size &stride, const String &padMode,
                          Size& out);

void getConvPoolPaddings(const Size& inp, const Size& out,
                         const Size &kernel, const Size &stride,
                         const String &padMode, Size &pad);

#if CV_SSE2
#define CV_DNN_TRY_AVX2 1

void fastConv_avx2(const float* weights, size_t wstep, const float* bias,
                   const float* rowbuf, float* output, const int* outShape,
                   int blockSize, int vecsize, int vecsize_aligned,
                   const float* relu, bool initOutput);
void fastGEMM1T_avx2( const float* vec, const float* weights,
                     size_t wstep, const float* bias,
                     float* dst, int nvecs, int vecsize );
void fastGEMM_avx2( const float* aptr, size_t astep, const float* bptr0,
                   size_t bstep, float* cptr, size_t cstep,
                   int ma, int na, int nb );

#else
#define CV_DNN_TRY_AVX2 0
#endif

}
}

#endif
