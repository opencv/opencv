// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_SRC_LAYERS_CPU_KERNELS_FUSION_APPLY_HPP__
#define __OPENCV_DNN_SRC_LAYERS_CPU_KERNELS_FUSION_APPLY_HPP__

#include "opencv2/core.hpp"
#include "opencv2/dnn/all_layers.hpp"
#include "../../adjacency_graph.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

/** @brief Fused math a layer took on, ready to run over its output. Either a fast
 *  activation kernel when we recognize the shape, or the expression for the interpreter.
 */
struct PreparedFusion
{
    Ptr<AdjacencyGraph> expr;                     //!< set once prepared, either way
    ActivationFunc activationFn = nullptr;     //!< non-null picks the fast kernel
    std::vector<float> activationParams;       //!< arguments for activationFn
    std::vector<const float*> channelBufs;     //!< per-channel constants, indexed by bufferId

    //! Takes on `e` if this can run it. Refusing leaves the object untouched.
    bool take(const Ptr<AdjacencyGraph>& e);
    //! Runs the taken math over a layer's output. Called on every inference.
    void run(Mat& Y) const;
};

inline bool PreparedFusion::take(const Ptr<AdjacencyGraph>& e)
{
    if (expr)
        return false;
    if (!e || e->size() == 0)
        return false;
    if (e->outputNode != (int)e->size() - 1)
        return false;
    if (e->size() > (size_t)FUSION_MAX_EXPR_NODES)
        return false;

    // Built aside and assigned only on success: a caller that refuses must be left
    // exactly as it was, or its later shorter offers get rejected by its own guard
    // and forward() runs math the layer never took on.
    PreparedFusion prepared;
    prepared.expr = e;

    // The layer may already own a kernel for exactly this math; use it rather than
    // decomposing and then recognising the pieces again.
    if (e->kernel) {
        prepared.activationFn = e->kernel;
        prepared.activationParams.assign(e->kernelParams,
                                         e->kernelParams + e->kernelParamCount);
        *this = prepared;
        return true;
    }

    prepared.channelBufs.resize(e->constBufs.size());
    for (size_t i = 0; i < e->constBufs.size(); i++) {
        const Mat& m = e->constBufs[i];
        if (m.empty() || m.type() != CV_32F || !m.isContinuous())
            return false;
        prepared.channelBufs[i] = m.ptr<float>();
    }
    for (const FusionNode& n : e->nodes()) {
        if (n.op != FusionEltwiseOp::PER_CHANNEL_CONST)
            continue;
        if (n.constBufferId < 0 || n.constBufferId >= (int)prepared.channelBufs.size())
            return false;
        const Mat& m = e->constBufs[n.constBufferId];
        if (m.dims < 1 || (int)m.total() != m.size[m.dims - 1])
            return false;
    }

    *this = prepared;
    return true;
}

inline void PreparedFusion::run(Mat& Y) const
{
    if (!expr)
        return;
    CV_CheckTypeEQ(Y.type(), CV_32F, "DNN/fusion: output must be CV_32F");
    CV_Assert(Y.isContinuous());

    float* p = Y.ptr<float>();
    const size_t n = Y.total();
    if (n == 0)
        return;
    CV_CheckLE(n, (size_t)INT_MAX, "DNN/fusion: output too large to split");

    if (activationFn) {
        const float* pr = activationParams.empty() ? nullptr : activationParams.data();
        parallel_for_(Range(0, (int)n), [&](const Range& r) {
            activationFn(p + r.start, p + r.start, (size_t)(r.end - r.start), pr);
        });
        return;
    }

    const int nch = Y.dims > 0 ? std::max(Y.size[Y.dims - 1], 1) : 1;
    for (const FusionNode& n : expr->nodes()) {
        if (n.op == FusionEltwiseOp::PER_CHANNEL_CONST)
            CV_CheckEQ((int)expr->constBufs[n.constBufferId].total(), nch,
                       "DNN/fusion: per-channel constant length must match the channel axis");
    }

    const AdjacencyGraph& g = *expr;
    const std::vector<const float*>& bufs = channelBufs;
    parallel_for_(Range(0, (int)n), [&](const Range& r) {
        int c = r.start % nch;
        for (int k = r.start; k < r.end; k++) {
            p[k] = fusion::evalElement(g, p[k], bufs, c);
            if (++c == nch) c = 0;
        }
    });
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif
