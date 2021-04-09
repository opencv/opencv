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

    enum class InterpolationType {
        NEAREST_NEIGHBOUR,
        BILINEAR
    };

    struct ResizeConfiguration {
        InterpolationType type;
        bool align_corners;
        bool half_pixel_centers;
    };

    template <class T>
    class ResizeOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ResizeOp(csl::Stream stream_, const ResizeConfiguration& config)
            : stream(std::move(stream_))
        {
            type = config.type;
            align_corners = config.align_corners;
            half_pixel_centers = config.half_pixel_centers;
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            // sometimes the target shape is taken from the second input; we don't use it however
            CV_Assert((inputs.size() == 1 || inputs.size() == 2) && outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            const auto compute_scale = [this](std::size_t input_size, std::size_t output_size) {
                return (align_corners && output_size > 1) ?
                            static_cast<float>(input_size - 1) / (output_size - 1) :
                            static_cast<float>(input_size) / output_size;
            };

            auto out_height = output.get_axis_size(-2), out_width = output.get_axis_size(-1);
            auto in_height = input.get_axis_size(-2), in_width = input.get_axis_size(-1);
            float scale_height = compute_scale(in_height, out_height),
                  scale_width = compute_scale(in_width, out_width);

            if (type == InterpolationType::NEAREST_NEIGHBOUR)
                kernels::resize_nn<T>(stream, output, input, scale_height, scale_width, align_corners, half_pixel_centers);
            else if (type == InterpolationType::BILINEAR)
                kernels::resize_bilinear<T>(stream, output, input, scale_height, scale_width, half_pixel_centers);
        }

    private:
        csl::Stream stream;
        InterpolationType type;
        bool align_corners, half_pixel_centers;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RESIZE_HPP */
