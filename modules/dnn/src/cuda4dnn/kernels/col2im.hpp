// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_COL2IM_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_COL2IM_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    struct Col2ImParams {
        int batch, channels, total_channels, channel_offset;
        int in_h, in_w, out_h, out_w;
        int kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w;
    };

    // cols is [batch][channels * kernel_h * kernel_w][in_h * in_w]; each output pixel gathers the
    // column entries that land on it, plus bias[channel_offset + c] when bias is not empty
    template <class T>
    void col2im_2d(const csl::Stream& stream, csl::Span<T> output, csl::View<T> cols, csl::View<T> bias,
                   const Col2ImParams& params);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_COL2IM_HPP */
