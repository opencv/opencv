// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ROI_POOLING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ROI_POOLING_HPP

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"
#include "../csl/span.hpp"

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void roi_pooling(const csl::Stream& stream, csl::TensorSpan<T> output, csl::TensorView<T> input, csl::View<T> rois, T spatial_scale);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_ROI_POOLING_HPP */
