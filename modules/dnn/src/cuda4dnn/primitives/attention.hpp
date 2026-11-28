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

#include <algorithm>
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
        static constexpr std::size_t ATTN_Q_TILE = 256;

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
            const std::size_t maxProjWidth = std::max(cfg.qkv_hidden_sizes[0],
                                                      std::max(cfg.qkv_hidden_sizes[1], cfg.qkv_hidden_sizes[2]));
            const std::size_t qTile = std::min<std::size_t>(S, ATTN_Q_TILE);

            // one require() per span forward() acquires: the allocator rounds each to 256, so the
            // total must be a sum of rounded sizes, not a rounded sum
            csl::WorkspaceBuilder builder;
            builder.require<T>(B * S * maxProjWidth);   // proj scratch reused across Q, K, V
            for (int i = 0; i < 3; i++)
                builder.require<T>(B * N * S * (cfg.qkv_hidden_sizes[i] / N)); // headMajor[i]
            builder.require<T>(B * N * qTile * S);      // tiled scores [B*N, qTile, S]
            builder.require<T>(B * N * S * headV);      // context [B*N, S, headV]
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

            const std::size_t B = config.batch_size, S = config.seq_len, N = config.num_heads;
            if (B == 0 || S == 0)
                return;

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            const std::size_t K = config.input_hidden_size;
            const std::size_t headSize[3] = { config.qkv_hidden_sizes[0] / N,
                                              config.qkv_hidden_sizes[1] / N,
                                              config.qkv_hidden_sizes[2] / N };
            const std::size_t maxProjWidth = std::max(config.qkv_hidden_sizes[0],
                                                      std::max(config.qkv_hidden_sizes[1], config.qkv_hidden_sizes[2]));
            const std::size_t qTile = std::min<std::size_t>(S, ATTN_Q_TILE);

            csl::WorkspaceAllocator allocator(workspace);

            auto projBuf = allocator.get_span<T>(B * S * maxProjWidth);
            csl::TensorSpan<T> headMajor[3];
            for (int i = 0; i < 3; i++) {
                std::vector<std::size_t> hm = { B, N, S, headSize[i] };
                headMajor[i] = allocator.get_tensor_span<T>(std::begin(hm), std::end(hm));
            }
            auto scoresBuf = allocator.get_span<T>(B * N * qTile * S);
            std::vector<std::size_t> ctxShape = { B * N, S, headSize[2] };
            auto context = allocator.get_tensor_span<T>(std::begin(ctxShape), std::end(ctxShape));

            auto input2d = csl::TensorView<T>(input);
            while (input2d.rank() < 2) input2d.unsqueeze();
            input2d.reshape(B * S, K);

            for (int i = 0; i < 3; i++) {
                std::size_t flat[2] = { B * S, config.qkv_hidden_sizes[i] };
                csl::TensorSpan<T> proj(projBuf.data(), flat, flat + 2);
                csl::tensor_ops::gemm<T>(cublasHandle, 0.0, proj, 1.0,
                                         false, input2d, false, csl::TensorView<T>(weightTensor[i]));
                kernels::biasN<T>(stream, proj, csl::TensorView<T>(proj), 1,
                                  csl::TensorView<T>(biasTensor[i]));

                auto asHeads = csl::TensorView<T>(proj);
                while (asHeads.rank() < 4) asHeads.unsqueeze();
                asHeads.reshape(B, S, N, headSize[i]);
                kernels::permute<T>(stream, headMajor[i], asHeads, { 0, 2, 1, 3 });
            }

            csl::TensorView<T> mask;
            std::size_t mB = 1, mN = 1, mQ = 1, mK = 1;
            if (inputs.size() == 2) {
                mask = csl::viewOf<T>(inputs[1]);
                if (mask.rank() > 4)
                    mask.squeeze_to(4);  // extra leading dims must be 1
                while (mask.rank() < 4)
                    mask.unsqueeze();
                mB = mask.get_axis_size(0);
                mN = mask.get_axis_size(1);
                mQ = mask.get_axis_size(2);
                mK = mask.get_axis_size(3);
            }

            const std::size_t dQK = headSize[0], dV = headSize[2];
            const std::size_t batchCount = B * N;

            // Process query positions in ATTN_Q_TILE blocks so scores stay O(B*N*TILE*S) instead of O(B*N*S^2).
            for (std::size_t q0 = 0; q0 < S; q0 += qTile) {
                const std::size_t qlen = std::min(qTile, S - q0);

                // Row-major Q_tile [B*N, qlen, dQK] * K^T [B*N, dQK, S] -> scoresTile [B*N, qlen, S]
                csl::cublas::gemmStridedBatched<T>(
                    cublasHandle,
                    true, false,
                    S, qlen, dQK,
                    static_cast<T>(config.scale),
                    csl::DevicePtr<const T>(headMajor[1].get()), dQK, S * dQK,
                    csl::DevicePtr<const T>(headMajor[0].get()) + static_cast<std::ptrdiff_t>(q0 * dQK), dQK, S * dQK,
                    static_cast<T>(0.0f),
                    scoresBuf.data(), S, qlen * S,
                    batchCount);

                if (inputs.size() == 2) {
                    if (mQ == 1) {
                        std::size_t s4[4] = { B, N, qlen, S };
                        csl::TensorSpan<T> scores4(scoresBuf.data(), s4, s4 + 4);
                        kernels::eltwise_sum_2<T>(stream, scores4, csl::TensorView<T>(scores4), mask);
                    } else if (mB == 1 && mN == 1) {
                        std::size_t s4[4] = { B, N, qlen, S };
                        std::size_t m4[4] = { 1, 1, qlen, mK };
                        csl::TensorSpan<T> scores4(scoresBuf.data(), s4, s4 + 4);
                        csl::TensorView<T> maskTile(mask.get() + static_cast<std::ptrdiff_t>(q0 * mK), m4, m4 + 4);
                        kernels::eltwise_sum_2<T>(stream, scores4, csl::TensorView<T>(scores4), maskTile);
                    } else if (mN == 1) {
                        std::size_t s3[3] = { N, qlen, S };
                        std::size_t m3[3] = { 1, qlen, mK };
                        for (std::size_t b = 0; b < B; b++) {
                            const std::size_t mb = (mB == 1) ? 0 : b;
                            csl::TensorSpan<T> scoresB(scoresBuf.data() + static_cast<std::ptrdiff_t>(b * N * qlen * S), s3, s3 + 3);
                            csl::TensorView<T> maskB(mask.get() + static_cast<std::ptrdiff_t>((mb * mQ + q0) * mK), m3, m3 + 3);
                            kernels::eltwise_sum_2<T>(stream, scoresB, csl::TensorView<T>(scoresB), maskB);
                        }
                    } else {
                        std::size_t s2[2] = { qlen, S };
                        std::size_t m2[2] = { qlen, mK };
                        for (std::size_t b = 0; b < B; b++) {
                            const std::size_t mb = (mB == 1) ? 0 : b;
                            for (std::size_t n = 0; n < N; n++) {
                                csl::TensorSpan<T> scoresBN(scoresBuf.data() + static_cast<std::ptrdiff_t>((b * N + n) * qlen * S), s2, s2 + 2);
                                csl::TensorView<T> maskBN(mask.get() + static_cast<std::ptrdiff_t>(((mb * mN + n) * mQ + q0) * mK), m2, m2 + 2);
                                kernels::eltwise_sum_2<T>(stream, scoresBN, csl::TensorView<T>(scoresBN), maskBN);
                            }
                        }
                    }
                }

                kernels::softmax<T>(stream,
                                    csl::Span<T>(scoresBuf.data(), batchCount * qlen * S),
                                    csl::View<T>(scoresBuf.data(), batchCount * qlen * S),
                                    static_cast<int>(S), static_cast<int>(batchCount * qlen), 1, false);

                // Row-major scoresTile [B*N, qlen, S] * V [B*N, S, dV] -> context[:, q0:q0+qlen, :]
                csl::cublas::gemmStridedBatched<T>(
                    cublasHandle,
                    false, false,
                    dV, qlen, S,
                    static_cast<T>(1.0f),
                    csl::DevicePtr<const T>(headMajor[2].get()), dV, S * dV,
                    csl::DevicePtr<const T>(scoresBuf.data()), S, qlen * S,
                    static_cast<T>(0.0f),
                    context.get() + static_cast<std::ptrdiff_t>(q0 * dV), dV, S * dV,
                    batchCount);
            }

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
