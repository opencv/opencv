// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_FLASH_ATTENTION_HPP
#define OPENCV_DNN_FLASH_ATTENTION_HPP

#include <cstddef>

namespace cv { namespace dnn {

// out = softmax(scale * Q * K^T + mask) * V, computed tile by tile without the full score matrix.
// Q: [B*H, Sq, D], K: [B*H, Skv, D] (or [B*H, D, Skv] if k_transposed), V: [B*H, Skv, Dv], out: [B, Sq, H, Dv].
// mask is optional and additive; mask_steps are its element steps over (batch, head, query, key),
// 0 for a broadcast axis.
void flashAttention(const float* Q, const float* K, const float* V, float* out,
                    int B, int H, int Sq, int Skv, int D, int Dv, float scale,
                    const float* mask = nullptr, const size_t* mask_steps = nullptr,
                    bool k_transposed = false);

}}  // namespace cv::dnn

#endif  // OPENCV_DNN_FLASH_ATTENTION_HPP
