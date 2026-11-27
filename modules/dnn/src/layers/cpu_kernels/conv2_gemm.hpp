// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_CONV2_GEMM_HPP
#define OPENCV_DNN_CONV2_GEMM_HPP

#include "../conv2_common.hpp"
#include "gemm_select.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv { namespace dnn {

enum Conv2GemmMode
{
    CONV2_GEMM_OFF = 0,     //!< use the NCHWc kernels
    CONV2_GEMM_FORCED = 1,  //!< OPENCV_DNN_CONV_GEMM=1: any AArch64 CPU and any conv size (for tests)
    CONV2_GEMM_AUTO = 2     //!< default when the core GEMM engine runs on SME
};

#if CV_NEON && CV_NEON_AARCH64

// OPENCV_DNN_CONV_GEMM: 0 = off, 1 = forced on, unset = on when the GEMM engine uses SME. Read on every call.
Conv2GemmMode conv2GemmMode();

// Packs the Conv2 weights (Cout, Cin, kh, kw) as the GEMM's B, rows ordered tap by tap like im2col.
void conv2GemmPackWeights(const Mat& wpack, const MatShape& wshape0, int C0, EngineGemmB& packed);

// 2D convolution on NCHWc8 input as im2col + GEMM, then scale, bias, residual and activation
// in the same order as the Conv2 kernels.
void conv2GemmForward(const Mat& inp, Mat& out, const float* residual, const ConvState& cs,
                      const EngineGemmB& packed, int Cout, int Cin, const float* scale, const float* bias,
                      Mat& scratchA, Mat& scratchC);

#else

inline Conv2GemmMode conv2GemmMode() { return CONV2_GEMM_OFF; }

#endif

}} // cv::dnn

#endif // OPENCV_DNN_CONV2_GEMM_HPP
