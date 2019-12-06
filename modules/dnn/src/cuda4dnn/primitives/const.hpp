// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONST_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONST_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"

#include <opencv2/core.hpp>

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ConstOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ConstOp(csl::Stream stream_, const cv::Mat& data)
            : stream(std::move(stream_))
        {
            constTensor = csl::makeTensorHeader<T>(data);
            csl::copyMatToTensor<T>(data, constTensor, stream);
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(outputs.size() == 1 && inputs.size() == 0);

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();
            csl::tensor_ops::copy<T>(stream, output, constTensor);
        }

    private:
        csl::Stream stream;
        csl::Tensor<T> constTensor;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONST_HPP */
