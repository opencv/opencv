// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ATTENTION_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ATTENTION_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/cublas.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"
#include "../csl/workspace.hpp"

#include "../kernels/scale_shift.hpp"
#include "../kernels/permute.hpp"
#include "../kernels/softmax.hpp"
#include "../kernels/eltwise_ops.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct AttentionConfiguration {
        std::size_t num_heads;
        std::vector<std::size_t> qkv_hidden_sizes;
        std::size_t input_hidden_size;
        std::size_t batch_size;
        std::size_t seq_len;
        float scale;
    };

    template <class T>
    class AttentionOp final : public CUDABackendNode {
    public:
        AttentionOp(csl::Stream stream_, csl::cublas::Handle handle,
                    const Mat& weight, const Mat& bias, const AttentionConfiguration& cfg)
            : stream(std::move(stream_)), cublasHandle(std::move(handle)), config(cfg)
        {
            const int K = static_cast<int>(cfg.input_hidden_size);
            std::size_t offset = 0;
            for (int i = 0; i < 3; i++) {
                const int width = static_cast<int>(cfg.qkv_hidden_sizes[i]);
                Mat part = weight.rowRange(0, K).colRange(static_cast<int>(offset), static_cast<int>(offset) + width).clone();
                weightTensor[i] = csl::makeTensorHeader<T>(part);
                csl::copyMatToTensor<T>(part, weightTensor[i], stream);

                Mat biasPart = bias.reshape(1, 1).colRange(static_cast<int>(offset),
                                                           static_cast<int>(offset) + width).clone();
                biasTensor[i] = csl::makeTensorHeader<T>(biasPart);
                csl::copyMatToTensor<T>(biasPart, biasTensor[i], stream);
                offset += cfg.qkv_hidden_sizes[i];
            }

            const std::size_t B = cfg.batch_size, S = cfg.seq_len, N = cfg.num_heads;
            const std::size_t headV = cfg.qkv_hidden_sizes[2] / N;

            // one require() per span forward() acquires: the allocator rounds each to 256, so the
            // total must be a sum of rounded sizes, not a rounded sum
            csl::WorkspaceBuilder builder;
            for (int i = 0; i < 3; i++) {
                builder.require<T>(B * S * cfg.qkv_hidden_sizes[i]);                 // proj[i]
                builder.require<T>(B * N * S * (cfg.qkv_hidden_sizes[i] / N));       // headMajor[i]
            }
            builder.require<T>(B * N * S * S);          // scores
            builder.require<T>(B * N * S * headV);      // context
            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Check(inputs.size(), inputs.size() == 1 || inputs.size() == 2,
                     "DNN/CUDA Attention: rotary and non-constant weight inputs are not supported");
            CV_Assert(outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            const std::size_t B = config.batch_size, S = config.seq_len, N = config.num_heads;
            const std::size_t K = config.input_hidden_size;
            const std::size_t headSize[3] = { config.qkv_hidden_sizes[0] / N,
                                              config.qkv_hidden_sizes[1] / N,
                                              config.qkv_hidden_sizes[2] / N };

            csl::WorkspaceAllocator allocator(workspace);

            csl::TensorSpan<T> proj[3], headMajor[3];
            for (int i = 0; i < 3; i++) {
                std::vector<std::size_t> flat = { B * S, config.qkv_hidden_sizes[i] };
                proj[i] = allocator.get_tensor_span<T>(std::begin(flat), std::end(flat));
                std::vector<std::size_t> hm = { B, N, S, headSize[i] };
                headMajor[i] = allocator.get_tensor_span<T>(std::begin(hm), std::end(hm));
            }
            std::vector<std::size_t> scoreShape = { B * N, S, S };
            auto scores = allocator.get_tensor_span<T>(std::begin(scoreShape), std::end(scoreShape));
            std::vector<std::size_t> ctxShape = { B * N, S, headSize[2] };
            auto context = allocator.get_tensor_span<T>(std::begin(ctxShape), std::end(ctxShape));

            auto input2d = csl::TensorView<T>(input);
            while (input2d.rank() < 2) input2d.unsqueeze();
            input2d.reshape(B * S, K);

            for (int i = 0; i < 3; i++) {
                csl::tensor_ops::gemm<T>(cublasHandle, 0.0, proj[i], 1.0,
                                         false, input2d, false, csl::TensorView<T>(weightTensor[i]));
                kernels::biasN<T>(stream, proj[i], csl::TensorView<T>(proj[i]), 1,
                                  csl::TensorView<T>(biasTensor[i]));

                auto asHeads = csl::TensorView<T>(proj[i]);
                while (asHeads.rank() < 4) asHeads.unsqueeze();
                asHeads.reshape(B, S, N, headSize[i]);
                kernels::permute<T>(stream, headMajor[i], asHeads, { 0, 2, 1, 3 });
            }

            auto q3 = csl::TensorView<T>(headMajor[0]); while (q3.rank() < 3) q3.unsqueeze(); q3.reshape(B * N, S, headSize[0]);
            auto k3 = csl::TensorView<T>(headMajor[1]); while (k3.rank() < 3) k3.unsqueeze(); k3.reshape(B * N, S, headSize[1]);
            auto v3 = csl::TensorView<T>(headMajor[2]); while (v3.rank() < 3) v3.unsqueeze(); v3.reshape(B * N, S, headSize[2]);

            csl::tensor_ops::gemmStridedBatched<T>(cublasHandle, 0.0, scores,
                                                   static_cast<T>(config.scale), false, q3, true, k3);

            // additive mask, broadcast right-aligned onto [B, N, S, S] before the softmax, as on CPU
            if (inputs.size() == 2) {
                auto mask = csl::viewOf<T>(inputs[1]);
                if (mask.rank() > 4)
                    mask.squeeze_to(4);  // extra leading dims must be 1
                auto scores4 = csl::TensorSpan<T>(scores);
                while (scores4.rank() < 4) scores4.unsqueeze();
                scores4.reshape(B, N, S, S);
                kernels::eltwise_sum_2<T>(stream, scores4, csl::TensorView<T>(scores4), mask);
            }

            kernels::softmax<T>(stream, csl::Span<T>(scores), csl::View<T>(csl::TensorView<T>(scores)),
                                static_cast<int>(S), static_cast<int>(B * N * S), 1, false);

            csl::tensor_ops::gemmStridedBatched<T>(cublasHandle, 0.0, context, 1.0,
                                                   false, csl::TensorView<T>(scores), false, v3);

            auto ctx4 = csl::TensorView<T>(context);
            while (ctx4.rank() < 4) ctx4.unsqueeze();
            ctx4.reshape(B, N, S, headSize[2]);
            auto out4 = csl::TensorSpan<T>(output);
            while (out4.rank() < 4) out4.unsqueeze();
            out4.reshape(B, S, N, headSize[2]);
            kernels::permute<T>(stream, out4, ctx4, { 0, 2, 1, 3 });
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        csl::Stream stream;
        csl::cublas::Handle cublasHandle;
        csl::Tensor<T> weightTensor[3], biasTensor[3];
        AttentionConfiguration config;
        std::size_t scratch_mem_in_bytes = 0;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ATTENTION_HPP */
