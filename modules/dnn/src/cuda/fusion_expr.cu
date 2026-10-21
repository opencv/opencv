// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"
#include "../cuda4dnn/kernels/fusion_expr.hpp"

#include <opencv2/core.hpp>

#include <cstddef>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, int MaxNodes>
        __global__ void fusion_expr(Span<T> output, View<T> input,
                                    array<FusionOpNode, MaxNodes> nodes, int nnodes,
                                    View<float> cbufs, size_type nch)
        {
            for (auto i : grid_stride_range(output.size())) {
                const int c = static_cast<int>(i % nch);
                const float x = static_cast<float>(input[i]);

                float v[MaxNodes];
                for (int k = 0; k < nnodes; k++) {
                    const FusionOpNode n = nodes[k];
                    const float a = n.in0 >= 0 ? v[n.in0] : 0.f;
                    const float b = n.in1 >= 0 ? v[n.in1] : 0.f;

                    switch (n.op) {
                    case FUSION_EXPR_INPUT:             v[k] = x; break;
                    case FUSION_EXPR_CONST:             v[k] = n.s0; break;
                    case FUSION_EXPR_PER_CHANNEL_CONST: v[k] = cbufs[n.cbuf + c]; break;
                    case FUSION_EXPR_ADD:               v[k] = a + b; break;
                    case FUSION_EXPR_SUB:               v[k] = a - b; break;
                    case FUSION_EXPR_MUL:               v[k] = a * b; break;
                    case FUSION_EXPR_MAX:               v[k] = a < b ? b : a; break;
                    case FUSION_EXPR_MIN:               v[k] = b < a ? b : a; break;
                    case FUSION_EXPR_ERF:               v[k] = erff(a); break;
                    case FUSION_EXPR_TANH:              v[k] = tanhf(a); break;
                    case FUSION_EXPR_EXP:               v[k] = expf(a); break;
                    case FUSION_EXPR_SQRT:              v[k] = sqrtf(a); break;
                    case FUSION_EXPR_CLAMP:             v[k] = a < n.s0 ? n.s0 : (a > n.s1 ? n.s1 : a); break;
                    case FUSION_EXPR_RECIP:             v[k] = 1.f / a; break;
                    default:                            v[k] = 0.f; break;
                    }
                }

                output[i] = static_cast<T>(v[nnodes - 1]);
            }
        }
    }

    template <class T, int MaxNodes> static
    void launch_fusion_expr(const Stream& stream, Span<T> output, View<T> input,
                            const FusionOpNode* nodes, std::size_t nnodes,
                            View<float> cbufs, std::size_t nch)
    {
        CV_Assert(nnodes > 0 && nnodes <= static_cast<std::size_t>(MaxNodes));

        array<FusionOpNode, MaxNodes> nodes_k;
        for (int i = 0; i < MaxNodes; i++) {
            if (static_cast<std::size_t>(i) < nnodes)
                nodes_k[i] = nodes[i];
            else
                nodes_k[i] = FusionOpNode{ FUSION_EXPR_CONST, -1, -1, -1, 0.f, 0.f };
        }

        auto kernel = raw::fusion_expr<T, MaxNodes>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, input, nodes_k, static_cast<int>(nnodes),
                      cbufs, static_cast<size_type>(nch));
    }

    template <class T>
    void fusion_expr(const Stream& stream, Span<T> output, View<T> input,
                     const FusionOpNode* nodes, std::size_t nnodes,
                     View<float> cbufs, std::size_t nch)
    {
        CV_Assert(output.size() == input.size());
        CV_Assert(nnodes > 0 && nnodes <= static_cast<std::size_t>(FUSION_EXPR_MAX_NODES));
        CV_Assert(nch > 0);

        if (nnodes <= 8)
            launch_fusion_expr<T, 8>(stream, output, input, nodes, nnodes, cbufs, nch);
        else if (nnodes <= 16)
            launch_fusion_expr<T, 16>(stream, output, input, nodes, nnodes, cbufs, nch);
        else if (nnodes <= 32)
            launch_fusion_expr<T, 32>(stream, output, input, nodes, nnodes, cbufs, nch);
        else
            launch_fusion_expr<T, 64>(stream, output, input, nodes, nnodes, cbufs, nch);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void fusion_expr(const Stream&, Span<__half>, View<__half>, const FusionOpNode*,
                              std::size_t, View<float>, std::size_t);
#endif
    template void fusion_expr(const Stream&, Span<float>, View<float>, const FusionOpNode*,
                              std::size_t, View<float>, std::size_t);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
