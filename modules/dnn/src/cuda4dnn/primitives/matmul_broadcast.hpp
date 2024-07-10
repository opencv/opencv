// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MATMUL_BROADCAST_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MATMUL_BROADCAST_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/cublas.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"

#include "../kernels/eltwise_ops.hpp" // for adding bias

#include <opencv2/core.hpp>

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class MatMulBroadcastOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        MatMulBroadcastOp(csl::Stream stream_, csl::cublas::Handle handle, const Mat &B, const Mat &bias, bool _transA, bool _transB,
                 const std::vector<size_t> &A_offsets_, const std::vector<size_t> &B_offsets_, std::vector<size_t> &C_offsets_,
                 size_t batch_)
            : stream(std::move(stream_)), cublasHandle(std::move(handle)), A_offsets(A_offsets_), B_offsets(B_offsets_), C_offsets(C_offsets_), batch(batch_)
        {
            if (!B.empty()) {
                input_B_tensor = csl::makeTensorHeader<T>(B);
                csl::copyMatToTensor<T>(B, input_B_tensor, stream);
            }

            if (!bias.empty()) {
                bias_tensor = csl::makeTensorHeader<T>(bias);
                csl::copyMatToTensor<T>(bias, bias_tensor, stream);
            }

            transA = _transA;
            transB = _transB;
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            auto input_A_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input_A = input_A_wrapper->getView();

            csl::TensorView<T> input_B;
            if (input_B_tensor.empty()) {
                auto input_B_wrapper = inputs[1].dynamicCast<wrapper_type>();
                input_B = input_B_wrapper->getView();
            } else {
                input_B = csl::TensorView<T>(input_B_tensor);
            }

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            csl::tensor_ops::gemmBatched<T>(cublasHandle, batch, 0.f, output, C_offsets, 1.f, transA, input_A, A_offsets, transB, input_B, B_offsets);

            // add bias if exists
            if (!bias_tensor.empty() || inputs.size() >= 3) {
                csl::TensorView<T> bias;
                if (bias_tensor.empty()) {
                    auto bias_wrapper = inputs[2].dynamicCast<wrapper_type>();
                    bias = bias_wrapper->getView();
                } else {
                    bias = csl::TensorView<T>(bias_tensor);
                }

                kernels::eltwise_sum_2<T>(stream, output, output, bias);
            }
        }

    private:
        csl::Stream stream;
        csl::cublas::Handle cublasHandle;
        csl::Tensor<T> input_B_tensor;
        csl::Tensor<T> bias_tensor;
        bool transA, transB;

        std::vector<size_t> A_offsets;
        std::vector<size_t> B_offsets;
        std::vector<size_t> C_offsets;
        size_t batch;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MATMUL_BROADCAST_HPP */
