// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_DYNAMIC_QUANTIZE_HPP
#define OPENCV_DNN_SRC_DYNAMIC_QUANTIZE_HPP

namespace cv { namespace dnn {

// ONNX DynamicQuantizeLinear to uint8 (scale and zero point from the range of x widened to 0),
// shared by the DynamicQuantizeLinear layer and by MatMulInt8 when it folds that layer in
inline void dynamicQuantizeLinearU8(const Mat& inp, uchar* dst, float& y_scale, uchar& y_zero_point)
{
    CV_Assert(inp.type() == CV_32F && inp.isContinuous());

    const float qmin = 0.f, qmax = 255.f;
    double minVal = 0.0, maxVal = 0.0;
    minMaxIdx(inp, &minVal, &maxVal);
    float xmin = std::min(0.f, (float)minVal);
    float xmax = std::max(0.f, (float)maxVal);

    y_scale = (xmax - xmin) / (qmax - qmin);
    if (y_scale == 0.f)
        y_scale = 1.f; // degenerate all-zero input: avoid divide-by-zero

    // round-half-to-even + clamp to [0, 255], same as the ONNX reference.
    float intermediate_zp = qmin - xmin / y_scale;
    y_zero_point = saturate_cast<uchar>(intermediate_zp);

    // Round first, then add the integer zero-point: folding an odd zp in before rounding
    // flips half-to-even ties.
    const float* src = inp.ptr<float>();
    const float sc = y_scale;
    int64_t total = (int64_t)inp.total();
    int zp_i = (int)y_zero_point;
    const int64_t block = 1024;
    int64_t nblocks = (total + block - 1) / block;
    parallel_for_(Range(0, (int)nblocks), [&](const Range& r) {
        for (int i = r.start; i < r.end; i++) {
            int64_t ofs = i * block;
            int64_t len = std::min(block, total - ofs);
            for (int64_t j = 0; j < len; j++)
                dst[ofs + j] = saturate_cast<uchar>(cvRound(src[ofs + j] / sc) + zp_i);
        }
    });
}

}}

#endif
