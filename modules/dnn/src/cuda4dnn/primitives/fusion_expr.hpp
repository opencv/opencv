// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_FUSION_EXPR_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_FUSION_EXPR_HPP

#include "../../op_cuda.hpp"
#include "../../adjacency_graph.hpp"

#include "../csl/stream.hpp"
#include "../csl/span.hpp"
#include "../csl/tensor.hpp"

#include "../kernels/fusion_expr.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    /* A fused pointwise expression, flattened into the form the device evaluator wants.
     *
     * runnable() is the predicate supportBackend() asks before claiming CUDA; build() does the
     * same checks and then packs. Anything refused here stays on the host exactly as before.
     */
    class FusionExprPlan {
    public:
        FusionExprPlan() = default;

        bool empty() const noexcept { return nodes.empty(); }

        /* `nch` is the length of the last axis, the axis PER_CHANNEL_CONST indexes. Pass 0 when
         * it is not known yet -- supportBackend() runs before shapes are settled -- and the
         * per-channel length check is left to build().
         */
        static bool runnable(const AdjacencyGraph& g, std::size_t nch)
        {
            const std::vector<FusionNode>& src = g.nodes();
            if (src.empty() || src.size() > static_cast<std::size_t>(kernels::FUSION_EXPR_MAX_NODES))
                return false;

            for (const FusionNode& n : src) {
                if (n.inputs.size() > 2)
                    return false;
                for (int in : n.inputs)
                    if (in < 0 || in >= static_cast<int>(src.size()))
                        return false;
                if (n.op != FusionEltwiseOp::PER_CHANNEL_CONST)
                    continue;
                if (n.constBufferId < 0 || n.constBufferId >= static_cast<int>(g.constBufs.size()))
                    return false;
                const Mat& m = g.constBufs[n.constBufferId];
                if (m.empty() || m.type() != CV_32F || !m.isContinuous())
                    return false;
                if (nch != 0 && m.total() != nch)
                    return false;
            }
            return true;
        }

        bool build(const AdjacencyGraph& g, std::size_t nch, const csl::Stream& stream)
        {
            checkOpcodes();
            if (nch == 0 || !runnable(g, nch))
                return false;

            const std::vector<FusionNode>& src = g.nodes();
            nodes.resize(src.size());

            std::vector<float> packed;
            std::vector<int> offsetOf(g.constBufs.size(), -1);

            for (std::size_t i = 0; i < src.size(); i++) {
                const FusionNode& n = src[i];
                kernels::FusionOpNode& d = nodes[i];
                d.op = static_cast<int>(n.op);
                d.in0 = !n.inputs.empty() ? n.inputs[0] : -1;
                d.in1 = n.inputs.size() > 1 ? n.inputs[1] : -1;
                d.cbuf = -1;
                d.s0 = n.scalar;
                d.s1 = n.scalar2;

                if (n.op != FusionEltwiseOp::PER_CHANNEL_CONST)
                    continue;

                int& off = offsetOf[n.constBufferId];
                if (off < 0) {
                    const Mat& m = g.constBufs[n.constBufferId];
                    off = static_cast<int>(packed.size());
                    packed.insert(packed.end(), m.ptr<float>(), m.ptr<float>() + m.total());
                }
                d.cbuf = off;
            }

            if (!packed.empty()) {
                // the upload is queued on `stream`, so the staging Mat has to outlive this call
                channelConstsHost = Mat(1, static_cast<int>(packed.size()), CV_32F, packed.data()).clone();
                channelConsts = csl::makeTensorHeader<float>(channelConstsHost);
                csl::copyMatToTensor<float>(channelConstsHost, channelConsts, stream);
            }
            return true;
        }

        template <class T>
        void run(const csl::Stream& stream, csl::Span<T> inout, std::size_t nch) const
        {
            if (nodes.empty())
                return;
            csl::View<float> consts;
            if (!channelConsts.empty())
                consts = channelConsts;
            kernels::fusion_expr<T>(stream, inout, inout, nodes.data(), nodes.size(), consts, nch);
        }

    private:
        /* The device enum is a copy; drift between the two would silently evaluate the wrong op. */
        static void checkOpcodes()
        {
            static_assert((int)FusionEltwiseOp::INPUT             == kernels::FUSION_EXPR_INPUT, "");
            static_assert((int)FusionEltwiseOp::CONST             == kernels::FUSION_EXPR_CONST, "");
            static_assert((int)FusionEltwiseOp::PER_CHANNEL_CONST == kernels::FUSION_EXPR_PER_CHANNEL_CONST, "");
            static_assert((int)FusionEltwiseOp::ADD               == kernels::FUSION_EXPR_ADD, "");
            static_assert((int)FusionEltwiseOp::SUB               == kernels::FUSION_EXPR_SUB, "");
            static_assert((int)FusionEltwiseOp::MUL               == kernels::FUSION_EXPR_MUL, "");
            static_assert((int)FusionEltwiseOp::MAX               == kernels::FUSION_EXPR_MAX, "");
            static_assert((int)FusionEltwiseOp::MIN               == kernels::FUSION_EXPR_MIN, "");
            static_assert((int)FusionEltwiseOp::ERF               == kernels::FUSION_EXPR_ERF, "");
            static_assert((int)FusionEltwiseOp::TANH              == kernels::FUSION_EXPR_TANH, "");
            static_assert((int)FusionEltwiseOp::EXP               == kernels::FUSION_EXPR_EXP, "");
            static_assert((int)FusionEltwiseOp::SQRT              == kernels::FUSION_EXPR_SQRT, "");
            static_assert((int)FusionEltwiseOp::CLAMP             == kernels::FUSION_EXPR_CLAMP, "");
            static_assert((int)FusionEltwiseOp::RECIP             == kernels::FUSION_EXPR_RECIP, "");
        }

        std::vector<kernels::FusionOpNode> nodes;
        csl::Tensor<float> channelConsts;
        Mat channelConstsHost;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_FUSION_EXPR_HPP */
