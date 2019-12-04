// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SHORTCUT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SHORTCUT_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"

#include "../kernels/shortcut.hpp"

#include <opencv2/core.hpp>

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ShortcutOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ShortcutOp(csl::Stream stream_) : stream(std::move(stream_)) { }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 2);
            CV_Assert(outputs.size() == 1);

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto from_wrapper_y = inputs[1].dynamicCast<wrapper_type>();
            auto from = from_wrapper_y->getView();

            /* output shape is determined by the input shape */
            CV_Assert(is_shape_same(output, input));

            /* the shape of `from` and `output` must be same except for channels */
            CV_Assert(output.rank() == from.rank());
            for (int i = 0; i < output.rank(); i++) {
                if (i != 1) {
                    CV_Assert(from.get_axis_size(i) == output.get_axis_size(i));
                }
            }

            kernels::input_shortcut<T>(stream, output, input, from);
        }

    private:
        csl::Stream stream;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SHORTCUT_HPP */
