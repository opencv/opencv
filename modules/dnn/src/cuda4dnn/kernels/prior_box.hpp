// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_PRIOR_BOX_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_PRIOR_BOX_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    template <class T>
    void generate_prior_boxes(
        const csl::Stream& stream,
        csl::Span<T> output,
        csl::View<float> boxWidth, csl::View<float> boxHeight, csl::View<float> offsetX, csl::View<float> offsetY, float stepX, float stepY,
        std::vector<float> variance,
        std::size_t numPriors,
        std::size_t layerWidth, std::size_t layerHeight,
        std::size_t imageWidth, std::size_t imageHeight,
        bool normalize, bool clip);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_PRIOR_BOX_HPP */
