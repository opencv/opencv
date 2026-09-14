// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_RECURRENT_ACTIVATIONS_HPP
#define OPENCV_DNN_RECURRENT_ACTIVATIONS_HPP

#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>

namespace cv { namespace dnn { namespace recurrent {

// Gate activations shared by the ONNX GRU and RNN layers. CV_32F only.
inline void tanh(const Mat &src, Mat &dst)
{
    CV_Assert(src.type() == CV_32F);
    dst.create(src.size(), src.type());
    const int nrows = src.rows;
    const int cols = src.cols;
    parallel_for_(Range(0, nrows), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row)
        {
            const float* srcptr = src.ptr<float>(row);
            float* dstptr = dst.ptr<float>(row);
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int vlanes = VTraits<v_float32>::vlanes();
            v_float32 one = vx_setall_f32(1.f), two = vx_setall_f32(2.f), minus_two = vx_setall_f32(-2.f);
            for (; i <= cols - vlanes; i += vlanes)
            {
                v_float32 x = vx_load(srcptr + i);
                v_float32 e = v_exp(v_mul(minus_two, x));            // exp(-2x)
                v_float32 t = v_sub(v_div(two, v_add(one, e)), one); // 2/(1+exp(-2x)) - 1
                vx_store(dstptr + i, t);
            }
#endif
            for (; i < cols; ++i)
                dstptr[i] = std::tanh(srcptr[i]);
        }
    });
}

inline void sigmoid(const Mat &src, Mat &dst)
{
    CV_Assert(src.type() == CV_32F);
    dst.create(src.size(), src.type());
    const int nrows = src.rows;
    const int cols = src.cols;
    parallel_for_(Range(0, nrows), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row)
        {
            const float* srcptr = src.ptr<float>(row);
            float* dstptr = dst.ptr<float>(row);
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int vlanes = VTraits<v_float32>::vlanes();
            v_float32 one = vx_setall_f32(1.f), zero = vx_setzero_f32();
            for (; i <= cols - vlanes; i += vlanes)
            {
                v_float32 x = vx_load(srcptr + i);
                v_float32 t = v_exp(v_sub(zero, x));  // exp(-x)
                t = v_div(one, v_add(one, t));        // 1 / (1 + exp(-x))
                vx_store(dstptr + i, t);
            }
#endif
            for (; i < cols; ++i)
                dstptr[i] = 1.f / (1.f + std::exp(-srcptr[i]));
        }
    });
}

}}} // namespace cv::dnn::recurrent

#endif
