// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_MLAS_CONV_HPP
#define OPENCV_DNN_MLAS_CONV_HPP

#include "../conv2_common.hpp"

namespace cv { namespace dnn {

#ifdef HAVE_MLAS

enum MlasConvGemmMode
{
    MLAS_CONV_GEMM_OFF = 0,     //!< use the NCHWc kernels
    MLAS_CONV_GEMM_FORCED = 1,  //!< OPENCV_DNN_MLAS_CONV_GEMM=1: any MLAS CPU and any conv size (for tests)
    MLAS_CONV_GEMM_AUTO = 2     //!< default on SME CPUs, where MLAS runs KleidiAI
};

// OPENCV_DNN_MLAS_CONV_GEMM: 0 = off, 1 = forced on, unset = on only for SME CPUs.
MlasConvGemmMode mlasConvGemmMode();

// Packs the Conv2 weights (Cout, Cin, kh, kw) for the SGEMM. Returns false if MLAS refuses.
bool mlasConvGemmPackWeights(const Mat& wpack, const MatShape& wshape0, int C0, Mat& packed);

// 2D convolution on NCHWc8 input as im2col + SGEMM, then scale, bias, residual and
// activation in the same order as the Conv2 kernels.
void mlasConvGemmForward(const Mat& inp, Mat& out, const float* residual,
                         const ConvState& cs, const Mat& packed, int Cout, int Cin,
                         const float* scale, const float* bias,
                         Mat& scratchA, Mat& scratchC);

#else

enum MlasConvGemmMode { MLAS_CONV_GEMM_OFF = 0 };
inline MlasConvGemmMode mlasConvGemmMode() { return MLAS_CONV_GEMM_OFF; }

#endif  // HAVE_MLAS

}}  // cv::dnn

#endif  // OPENCV_DNN_MLAS_CONV_HPP
