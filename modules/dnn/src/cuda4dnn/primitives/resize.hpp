// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RESIZE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RESIZE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/resize.hpp"

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    enum class interpolation_type {
        nearest_neighbour,
        bilinear
    };

    template <class T>
    class ResizeOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ResizeOp(csl::Stream stream_, interpolation_type type_, float scaleHeight_, float scaleWidth_)
            : stream(std::move(stream_)), type{ type_ }, scaleHeight{ scaleHeight_ }, scaleWidth{ scaleWidth_ }
        {
        }

        void forward(
            std::vector<cv::Ptr<BackendWrapper>>& inputs,
            std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            if (type == interpolation_type::nearest_neighbour)
                kernels::resize_nn<T>(stream, output, input);
            else if (type == interpolation_type::bilinear)
                kernels::resize_bilinear<T>(stream, output, input, scaleHeight, scaleWidth);
        }

    private:
        csl::Stream stream;
        interpolation_type type;
        float scaleHeight, scaleWidth; /* for bilinear interpolation */
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RESIZE_HPP */
