// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_INNER_PRODUCT_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_INNER_PRODUCT_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/cublas.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"

#include "../kernels/scale_shift.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class InnerProductOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        InnerProductOp(csl::Stream stream_, csl::cublas::Handle handle, std::size_t axis, const Mat& weights, const Mat& bias)
            : stream(std::move(stream_)), cublasHandle(std::move(handle)), axis{ axis }
        {
            weightsTensor = csl::makeTensorHeader<T>(weights);
            CV_Assert(get_effective_rank(weightsTensor) <= 2);
            csl::copyMatToTensor<T>(weights, weightsTensor, stream);

            if (!bias.empty())
            {
                biasTensor = csl::makeTensorHeader<T>(bias);
                csl::copyMatToTensor<T>(bias, biasTensor, stream);
                CV_Assert(weightsTensor.get_axis_size(-2) == biasTensor.size());
            }
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            for (int i = 0; i < inputs.size(); i++)
            {
                auto input_wrapper = inputs[i].dynamicCast<wrapper_type>();
                auto input = input_wrapper->getView();

                auto output_wrapper = outputs[i].dynamicCast<wrapper_type>();
                auto output = output_wrapper->getSpan();

                std::size_t batch_size = input.size_range(0, axis);

                auto input_size = input.size() / batch_size;
                CV_Assert(input_size == weightsTensor.get_axis_size(-1));

                auto output_size = output.size() / batch_size;
                CV_Assert(output_size == weightsTensor.get_axis_size(-2));

                /* we treat the input and output as a matrix with dimensions (batch_size, input_size)
                 * and (batch_size, output_size) respectively
                 *
                 * weight matrix dimensions: (output_size, input_size)
                 *
                 * I(W^T) = O
                 * (batch_size, input_size) * (input_size, output_size) = (batch_size, output_size)
                 */
                input.reshape(batch_size, input_size);
                output.reshape(batch_size, output_size);
                csl::tensor_ops::gemm<T>(cublasHandle, 0.0, output, 1.0, false, input, true, weightsTensor);

                if (!biasTensor.empty())
                    kernels::biasN<T>(stream, output, output, 1, biasTensor);
            }
        }

    private:
        csl::Stream stream;
        csl::cublas::Handle cublasHandle;
        csl::Tensor<T> weightsTensor, biasTensor;
        std::size_t axis;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_INNER_PRODUCT_HPP */
