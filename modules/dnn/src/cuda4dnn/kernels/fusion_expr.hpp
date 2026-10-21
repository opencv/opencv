// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_KERNELS_FUSION_EXPR_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_KERNELS_FUSION_EXPR_HPP

#include "../csl/stream.hpp"
#include "../csl/span.hpp"

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    enum {
        FUSION_EXPR_MAX_NODES = 64
    };

    /* Mirrors FusionEltwiseOp in adjacency_graph.hpp. The device side cannot include that
     * header, so the values are duplicated here and checked against it in FusionExprPlan.
     */
    enum FusionExprOp {
        FUSION_EXPR_INPUT = 0,
        FUSION_EXPR_CONST,
        FUSION_EXPR_PER_CHANNEL_CONST,
        FUSION_EXPR_ADD,
        FUSION_EXPR_SUB,
        FUSION_EXPR_MUL,
        FUSION_EXPR_MAX,
        FUSION_EXPR_MIN,
        FUSION_EXPR_ERF,
        FUSION_EXPR_TANH,
        FUSION_EXPR_EXP,
        FUSION_EXPR_SQRT,
        FUSION_EXPR_CLAMP,
        FUSION_EXPR_RECIP
    };

    /* One node of the flattened expression. `in0`/`in1` index earlier nodes, -1 when the op
     * takes fewer operands; `cbuf` is an element offset into the packed per-channel buffer.
     */
    struct FusionOpNode {
        int op;
        int in0;
        int in1;
        int cbuf;
        float s0;
        float s1;
    };

    template <class T>
    void fusion_expr(const csl::Stream& stream, csl::Span<T> output, csl::View<T> input,
                     const FusionOpNode* nodes, std::size_t nnodes,
                     csl::View<float> channelConsts, std::size_t nch);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_KERNELS_FUSION_EXPR_HPP */
