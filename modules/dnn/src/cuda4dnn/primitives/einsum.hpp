// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_EINSUM_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_EINSUM_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/cublas.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"
#include "../csl/workspace.hpp"

#include "../kernels/permute.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    /** A two-operand einsum as permute -> strided batched GEMM -> permute.
     *
     * A is read as [batch, M, K] (or [batch, K, M] with transA) and B as [batch, K, N]
     * (or [batch, N, K] with transB); the GEMM writes [batch, M, N]. An empty perm means
     * the operand already has that layout, so it is used in place.
     */
    struct EinsumPlan {
        std::vector<std::size_t> permA, permB, permC;     // permC maps [batch, M, N] to the output
        std::vector<std::size_t> shapeA, shapeB, shapeC;  // full shapes after permA/permB, and of the GEMM result
        bool transA = false, transB = false;
        std::size_t batch = 1, M = 1, N = 1, K = 1;
    };

    template <class T>
    class EinsumOp final : public CUDABackendNode {
    public:
        EinsumOp(csl::Stream stream_, csl::cublas::Handle handle_, EinsumPlan plan_)
            : stream(std::move(stream_)), cublasHandle(std::move(handle_)), plan(std::move(plan_))
        {
            csl::WorkspaceBuilder builder;
            if (!plan.permA.empty())
                builder.require<T>(plan.batch * plan.M * plan.K);
            if (!plan.permB.empty())
                builder.require<T>(plan.batch * plan.K * plan.N);
            if (!plan.permC.empty())
                builder.require<T>(plan.batch * plan.M * plan.N);
            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 2 && outputs.size() == 1);

            auto allocator = csl::WorkspaceAllocator(workspace);
            csl::TensorView<T> a = operand(allocator, csl::viewOf<T>(inputs[0]), plan.permA, plan.shapeA);
            csl::TensorView<T> b = operand(allocator, csl::viewOf<T>(inputs[1]), plan.permB, plan.shapeB);

            const std::size_t rowsA = plan.transA ? plan.K : plan.M, colsA = plan.transA ? plan.M : plan.K;
            const std::size_t rowsB = plan.transB ? plan.N : plan.K, colsB = plan.transB ? plan.K : plan.N;

            auto output = csl::spanOf<T>(outputs[0]);
            csl::TensorSpan<T> c = plan.permC.empty() ? output
                : allocator.get_tensor_span<T>(plan.shapeC.begin(), plan.shapeC.end());

            // reshape() cannot raise the rank, so the rank-3 GEMM views are built over the same memory
            const std::size_t shapeA3[] = { plan.batch, rowsA, colsA };
            const std::size_t shapeB3[] = { plan.batch, rowsB, colsB };
            const std::size_t shapeC3[] = { plan.batch, plan.M, plan.N };
            csl::TensorView<T> a3(a.get(), std::begin(shapeA3), std::end(shapeA3));
            csl::TensorView<T> b3(b.get(), std::begin(shapeB3), std::end(shapeB3));
            csl::TensorSpan<T> c3(c.get(), std::begin(shapeC3), std::end(shapeC3));
            csl::tensor_ops::gemmStridedBatched<T>(cublasHandle, 0.0, c3, 1.0, plan.transA, a3, plan.transB, b3);

            if (!plan.permC.empty())
                kernels::permute<T>(stream, output, c, plan.permC);
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        // the operand itself, or a permuted copy in the workspace
        csl::TensorView<T> operand(csl::WorkspaceAllocator& allocator, csl::TensorView<T> input,
                                   const std::vector<std::size_t>& perm, const std::vector<std::size_t>& shape)
        {
            if (perm.empty())
                return input;
            auto permuted = allocator.get_tensor_span<T>(shape.begin(), shape.end());
            kernels::permute<T>(stream, permuted, input, perm);
            return permuted;
        }

        csl::Stream stream;
        csl::cublas::Handle cublasHandle;
        EinsumPlan plan;
        std::size_t scratch_mem_in_bytes;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_EINSUM_HPP */
