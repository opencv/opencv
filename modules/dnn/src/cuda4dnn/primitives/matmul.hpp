// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MATMUL_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MATMUL_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/cublas.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"

#include <opencv2/core.hpp>

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class MatMulOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        MatMulOp(csl::Stream stream_, csl::cublas::Handle handle, const Mat& constInp)
            : stream(std::move(stream_)), cublasHandle(std::move(handle))
        {
            if (!constInp.empty())
            {
                constTensor = csl::makeTensorHeader<T>(constInp);
                csl::copyMatToTensor<T>(constInp, constTensor, stream);
            }
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert((inputs.size() == 2 && constTensor.empty() ||
                       inputs.size() == 1 && !constTensor.empty()) && outputs.size() == 1);

            auto input1_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input1 = input1_wrapper->getView();

            csl::TensorView<T> input2;
            if (constTensor.empty())
            {
                auto input2_wrapper = inputs[1].dynamicCast<wrapper_type>();
                input2 = input2_wrapper->getView();
            }
            else
                input2 = csl::TensorView<T>(constTensor);

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            auto rank = output.rank();
            CV_Assert(rank == input1.rank());
            CV_Assert(rank == input2.rank());
            CV_Assert(rank >= 2); // 1D MatMul not supported

            for (int i = 0; i < rank - 2; i++)
            {
                // broadcasting not supported
                auto size = output.get_axis_size(i);
                CV_Assert(input1.get_axis_size(i) == size);
                CV_Assert(input2.get_axis_size(i) == size);
            }

            auto m = input1.get_axis_size(-2);
            auto n = input1.get_axis_size(-1);
            auto b = input1.size() / m / n;
            int k;
            if (constTensor.empty())
            {
                k = input2.get_axis_size(-1);
                CV_Assert(input2.get_axis_size(-2) == n);
            }
            else
            {
                k = input2.get_axis_size(-2);
                CV_Assert(input2.get_axis_size(-1) == n);
            }
            CV_Assert(output.get_axis_size(-2) == m);
            CV_Assert(output.get_axis_size(-1) == k);

            if (get_effective_rank(output) <= 2)
            {
                CV_Assert(b == 1);
                CV_Assert(get_effective_rank(input1) <= 2);
                CV_Assert(get_effective_rank(input2) <= 2);
                csl::tensor_ops::gemm<T>(cublasHandle, 0.0, output, 1.0, false, input1, !constTensor.empty(), input2);
            }
            else
            {
                CV_Assert(rank >= 3);
                input1.reshape(b, m, n);
                if (constTensor.empty())
                    input2.reshape(b, n, k);
                else
                    input2.reshape(b, k, n);
                output.reshape(b, m, k);
                input1.squeeze_to(3);
                input2.squeeze_to(3);
                output.squeeze_to(3);
                csl::tensor_ops::gemmStridedBatched<T>(cublasHandle, 0.0, output, 1.0, false, input1, !constTensor.empty(), input2);
            }
        }

    private:
        csl::Stream stream;
        csl::cublas::Handle cublasHandle;
        csl::Tensor<T> constTensor;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MATMUL_HPP */
