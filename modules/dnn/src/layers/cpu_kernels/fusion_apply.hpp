// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_SRC_LAYERS_CPU_KERNELS_FUSION_APPLY_HPP__
#define __OPENCV_DNN_SRC_LAYERS_CPU_KERNELS_FUSION_APPLY_HPP__

#include "opencv2/core.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "opencv2/dnn/all_layers.hpp"
#include "../../adjacency_graph.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

/** @brief Fused math a layer took on, ready to run over its output: a fast activation
 *  kernel, or a per-channel scale and/or shift along the last axis. */
struct PreparedFusion
{
    Ptr<AdjacencyGraph> expr;                     //!< set once prepared, either way
    ActivationFunc activationFn = nullptr;     //!< non-null picks the fast kernel
    std::vector<float> activationParams;       //!< arguments for activationFn

    //! y = x * scale + shift along the last axis; an empty buffer means its scalar is used.
    //! The buffers are shared with expr->constBufs or folded from them in take().
    Mat scaleBuf, shiftBuf;
    float scale = 1.f, shift = 0.f;

    //! Takes on `e` if this can run it. Refusing leaves the object untouched.
    bool take(const Ptr<AdjacencyGraph>& e);
    //! Runs the taken math over a layer's output. Called on every inference.
    void run(Mat& Y) const;
};

namespace fusion { namespace detail {

//! An affine coefficient: a scalar, or a last-axis buffer when `buf` is set.
struct ChannelCoef
{
    float val = 0.f;
    Mat buf;
};

//! Reads a CONST node as a scalar, or a PER_CHANNEL_CONST node as a last-axis buffer.
inline bool constantOperand(const AdjacencyGraph& g, int idx, ChannelCoef& c)
{
    const FusionNode& n = g.nodes()[idx];
    if (n.op == FusionEltwiseOp::CONST) {
        c.val = n.scalar;
        c.buf.release();
        return true;
    }
    if (n.op != FusionEltwiseOp::PER_CHANNEL_CONST)
        return false;
    const int id = n.constBufferId;
    if (id < 0 || id >= (int)g.constBufs.size())
        return false;
    // run() walks the last axis, but a flagged buffer varies along the channel one.
    // Refusing lets the caller fall back to a chain that leaves it alone.
    if (id < (int)g.constBufPerChannel.size() && g.constBufPerChannel[id])
        return false;
    const Mat& m = g.constBufs[id];
    if (m.empty() || m.type() != CV_32F || !m.isContinuous() ||
        m.dims < 1 || (int)m.total() != m.size[m.dims - 1])
        return false;
    c.val = 0.f;
    c.buf = m;
    return true;
}

enum class CoefOp
{
    ADD = 0,
    SUB = 1,
    MUL = 2
};

//! out = a op b elementwise, a scalar broadcasting over a buffer; fails on unequal lengths.
inline bool combine(const ChannelCoef& a, const ChannelCoef& b, CoefOp op, ChannelCoef& out)
{
    auto apply = [op](float x, float y) {
        return op == CoefOp::ADD ? x + y : op == CoefOp::SUB ? x - y : x * y;
    };
    if (a.buf.empty() && b.buf.empty()) {
        out.val = apply(a.val, b.val);
        out.buf.release();
        return true;
    }
    const int len = (int)(a.buf.empty() ? b.buf.total() : a.buf.total());
    if (!a.buf.empty() && !b.buf.empty() && (int)b.buf.total() != len)
        return false;
    Mat res(1, &len, CV_32F);
    const float* pa = a.buf.empty() ? nullptr : a.buf.ptr<float>();
    const float* pb = b.buf.empty() ? nullptr : b.buf.ptr<float>();
    float* pr = res.ptr<float>();
    for (int i = 0; i < len; i++)
        pr[i] = apply(pa ? pa[i] : a.val, pb ? pb[i] : b.val);
    out.val = 0.f;
    out.buf = res;
    return true;
}

//! Writes node `idx` as x*scale + shift, folding any chain of MUL, ADD and (x - c) by a
//! constant into the two coefficients. Fails on anything else.
inline bool affineOf(const AdjacencyGraph& g, int idx, ChannelCoef& scale, ChannelCoef& shift)
{
    const FusionNode& n = g.nodes()[idx];
    if (n.op == FusionEltwiseOp::INPUT) {
        scale = ChannelCoef();
        scale.val = 1.f;
        shift = ChannelCoef();
        return true;
    }
    const bool isSub = n.op == FusionEltwiseOp::SUB;
    if ((n.op != FusionEltwiseOp::MUL && n.op != FusionEltwiseOp::ADD && !isSub) ||
        n.inputs.size() != 2)
        return false;
    // One operand is a constant and the other depends on x; SUB only as (x - c).
    for (int k = 0; k < (isSub ? 1 : 2); k++) {
        ChannelCoef c;
        if (!constantOperand(g, n.inputs[1 - k], c) || !affineOf(g, n.inputs[k], scale, shift))
            continue;
        if (n.op == FusionEltwiseOp::MUL)
            return combine(scale, c, CoefOp::MUL, scale) && combine(shift, c, CoefOp::MUL, shift);
        return combine(shift, c, isSub ? CoefOp::SUB : CoefOp::ADD, shift);
    }
    return false;
}

}} // namespace fusion::detail

inline bool PreparedFusion::take(const Ptr<AdjacencyGraph>& e)
{
    if (expr)
        return false;
    if (!e || e->size() == 0)
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
    if (e->kernel.fn) {
        prepared.activationFn = e->kernel.fn;
        prepared.activationParams.assign(e->kernel.params,
                                         e->kernel.params + e->kernel.nparams);
        *this = prepared;
        return true;
    }

    // Without a kernel only a per-channel scale/shift is taken; anything else is refused so
    // it stays its own vectorized layer instead of running through the per-value interpreter.
    fusion::detail::ChannelCoef s, b;
    if (!fusion::detail::affineOf(*e, e->outputNode(), s, b))
        return false;
    prepared.scale = s.val;
    prepared.scaleBuf = s.buf;
    prepared.shift = b.val;
    prepared.shiftBuf = b.buf;

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

    // One contiguous range per thread. Without a stripe count, parallel_for_ hands the
    // range out in shrinking chunks, down to single elements at the end.
    const int nthreads = std::max(1, getNumThreads());

    if (activationFn) {
        const float* pr = activationParams.empty() ? nullptr : activationParams.data();
        parallel_for_(Range(0, (int)n), [&](const Range& r) {
            activationFn(p + r.start, p + r.start, (size_t)(r.end - r.start), pr);
        }, nthreads);
        return;
    }

    // take() accepts nothing else, so this is a per-channel scale and/or shift.
    const int nch = Y.dims > 0 ? std::max(Y.size[Y.dims - 1], 1) : 1;
    if (!scaleBuf.empty())
        CV_CheckEQ((int)scaleBuf.total(), nch, "DNN/fusion: per-channel constant length must match the channel axis");
    if (!shiftBuf.empty())
        CV_CheckEQ((int)shiftBuf.total(), nch, "DNN/fusion: per-channel constant length must match the channel axis");

    // A scalar coefficient is expanded into a row so one SIMD loop covers every form.
    AutoBuffer<float> expanded(2 * (size_t)nch);
    float* tmp = expanded.data();
    if (scaleBuf.empty())
        std::fill_n(tmp, nch, scale);
    if (shiftBuf.empty())
        std::fill_n(tmp + nch, nch, shift);
    const float* sb = scaleBuf.empty() ? tmp : scaleBuf.ptr<float>();
    const float* bb = shiftBuf.empty() ? tmp + nch : shiftBuf.ptr<float>();

    const int rows = (int)(n / (size_t)nch);
    parallel_for_(Range(0, rows), [&](const Range& r) {
        for (int i = r.start; i < r.end; i++) {
            float* row = p + (size_t)i * nch;
            int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int vl = VTraits<v_float32>::vlanes();
            for (; j <= nch - vl; j += vl)
                v_store(row + j, v_fma(vx_load(row + j), vx_load(sb + j), vx_load(bb + j)));
#endif
            for (; j < nch; j++)
                row[j] = row[j] * sb[j] + bb[j];
        }
    }, nthreads);
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif
