// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_BATCH_NORM_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_BATCH_NORM_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include "../kernels/scale_shift.hpp"

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class BatchNormOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        BatchNormOp(csl::Stream stream_, const cv::Mat& weights, const cv::Mat& bias)
            : stream(std::move(stream_))
        {
            biasTensor = csl::makeTensorHeader<T>(bias);
            csl::copyMatToTensor<T>(bias, biasTensor, stream);

            weightsTensor = csl::makeTensorHeader<T>(weights);
            csl::copyMatToTensor<T>(weights, weightsTensor, stream);
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            std::size_t inner_size = input.size_range(2, input.rank());
            kernels::scaleN_with_biasN<T>(stream, output, input, inner_size, weightsTensor, biasTensor);
        }

    private:
        csl::Stream stream;
        csl::Tensor<T> weightsTensor, biasTensor;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_BATCH_NORM_HPP */
