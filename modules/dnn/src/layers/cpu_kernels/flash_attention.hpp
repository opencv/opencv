// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_FLASH_ATTENTION_HPP
#define OPENCV_DNN_FLASH_ATTENTION_HPP

#include <opencv2/core.hpp>

namespace cv { namespace dnn {

// True when flashAttention() should run instead of MLAS flash attention (AArch64 only, see OPENCV_DNN_GEMM_PATH).
bool flashAttentionEnabled();

// out[b, i, h, :] = softmax(scale * Q[b, h, i, :] * K[b, h]^T) * V[b, h], without forming the full score matrix.
// Layouts as in mlasFlashAttention: Q [B, H, Sq, D], V [B, H, Skv, Dv], out [B, Sq, H, Dv];
// K is [B, H, Skv, D], or [B, H, D, Skv] when kTransposed is set.
void flashAttention(const float* Q, const float* K, const float* V, float* out,
                    int B, int H, int Sq, int Skv, int D, int Dv, float scale, bool kTransposed = false);

}} // cv::dnn

#endif // OPENCV_DNN_FLASH_ATTENTION_HPP
