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

// Gate activations shared by the ONNX GRU and RNN layers. CV_32F and CV_64F.
// CV_64F takes the scalar path; the universal intrinsics here are single precision.
template<typename T, typename Op>
inline void applyRowwise(const Mat &src, Mat &dst, Op op)
{
    dst.create(src.size(), src.type());
    const int nrows = src.rows, cols = src.cols;
    parallel_for_(Range(0, nrows), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row)
        {
            const T* srcptr = src.ptr<T>(row);
            T* dstptr = dst.ptr<T>(row);
            for (int i = 0; i < cols; ++i)
                dstptr[i] = op(srcptr[i]);
        }
    });
}

template<typename VecOp, typename ScalarOp>
inline void applyFloatRowwise(const Mat &src, Mat &dst, VecOp vop, ScalarOp sop)
{
    dst.create(src.size(), src.type());
    const int nrows = src.rows, cols = src.cols;
    parallel_for_(Range(0, nrows), [&](const Range& range) {
        for (int row = range.start; row < range.end; ++row)
        {
            const float* srcptr = src.ptr<float>(row);
            float* dstptr = dst.ptr<float>(row);
            int i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int vlanes = VTraits<v_float32>::vlanes();
            for (; i <= cols - vlanes; i += vlanes)
                vx_store(dstptr + i, vop(vx_load(srcptr + i)));
#endif
            for (; i < cols; ++i)
                dstptr[i] = sop(srcptr[i]);
        }
    });
}

inline void tanh(const Mat &src, Mat &dst)
{
    CV_Assert(src.type() == CV_32F || src.type() == CV_64F);
    if (src.type() == CV_64F)
    {
        applyRowwise<double>(src, dst, [](double x) { return std::tanh(x); });
        return;
    }
    applyFloatRowwise(src, dst,
        [](const v_float32& x) -> v_float32 {
            // 2/(1+exp(-2x)) - 1
            v_float32 one = vx_setall_f32(1.f), two = vx_setall_f32(2.f);
            return v_sub(v_div(two, v_add(one, v_exp(v_mul(vx_setall_f32(-2.f), x)))), one);
        },
        [](float x) { return std::tanh(x); });
}

inline void sigmoid(const Mat &src, Mat &dst)
{
    CV_Assert(src.type() == CV_32F || src.type() == CV_64F);
    if (src.type() == CV_64F)
    {
        applyRowwise<double>(src, dst, [](double x) { return 1.0 / (1.0 + std::exp(-x)); });
        return;
    }
    applyFloatRowwise(src, dst,
        [](const v_float32& x) -> v_float32 {
            // 1/(1+exp(-x))
            v_float32 one = vx_setall_f32(1.f);
            return v_div(one, v_add(one, v_exp(v_sub(vx_setzero_f32(), x))));
        },
        [](float x) { return 1.f / (1.f + std::exp(-x)); });
}

}}} // namespace cv::dnn::recurrent

#endif
