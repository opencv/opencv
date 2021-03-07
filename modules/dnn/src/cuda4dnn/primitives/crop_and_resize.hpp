// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CROP_AND_RESIZE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CROP_AND_RESIZE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include "../kernels/crop_and_resize.hpp"

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class CropAndResizeOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        CropAndResizeOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 2 && outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto box_wrapper = inputs[1].dynamicCast<wrapper_type>();
            auto boxes = box_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            kernels::crop_and_resize(stream, output, input, static_cast<csl::View<T>>(boxes));
        }

    private:
        csl::Stream stream;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CROP_AND_RESIZE_HPP */
