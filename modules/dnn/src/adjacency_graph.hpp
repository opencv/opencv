// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_SRC_ADJACENCY_GRAPH_HPP__
#define __OPENCV_DNN_SRC_ADJACENCY_GRAPH_HPP__

#include <cmath>
#include <unordered_map>
#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/dnn/all_layers.hpp"   // ActivationFunc

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

enum {
    //! Cap on one extracted expression, and the size of evalElement()'s stack array.
    FUSION_MAX_EXPR_NODES  = 64,
    //! Cap on one layer's math, and the size of LayerMath::nodes_.
    FUSION_MAX_MATH_NODES  = 16,
    //! Cap on the shared arena for a whole graph. Chain growth stops on it; internNode()
    //! only checks it under CV_DbgAssert, so treat it as advisory in release builds.
    FUSION_MAX_ARENA_NODES = 1 << 16
};

enum class FusionEltwiseOp
{
    INPUT,
    CONST,
    PER_CHANNEL_CONST,
    //! A second runtime tensor: another layer's live output, not a baked buffer.
    TENSOR,
    ADD,
    SUB,
    MUL,
    MAX,
    MIN,
    ERF,
    TANH,
    EXP,
    SQRT,
    CLAMP,
    RECIP
};

namespace fusion { namespace detail {

inline int arity(FusionEltwiseOp op)
{
    switch (op) {
    case FusionEltwiseOp::INPUT:
    case FusionEltwiseOp::CONST:
    case FusionEltwiseOp::PER_CHANNEL_CONST:
    case FusionEltwiseOp::TENSOR:
        return 0;
    case FusionEltwiseOp::ADD:
    case FusionEltwiseOp::SUB:
    case FusionEltwiseOp::MUL:
    case FusionEltwiseOp::MAX:
    case FusionEltwiseOp::MIN:
        return 2;
    case FusionEltwiseOp::ERF:
    case FusionEltwiseOp::TANH:
    case FusionEltwiseOp::EXP:
    case FusionEltwiseOp::SQRT:
    case FusionEltwiseOp::CLAMP:
    case FusionEltwiseOp::RECIP:
        return 1;
    }
    CV_Error(Error::StsBadArg, "DNN/fusion: unknown FusionEltwiseOp");
}

inline unsigned bits(float f) { Cv32suf s; s.f = f; return s.u; }

}} // namespace fusion::detail

/** @brief One constant operand: a scalar, or a per-channel buffer. */
struct FusionConst
{
    float value    = 0.f;   //!< the scalar, meaningful only while bufferId and tensorId are < 0
    int   bufferId = -1;    //!< >=0 selects a per-channel buffer, indexes constBufs
    int   tensorId = -1;    //!< >=0 selects a live tensor, indexes tensorArgs

    bool isBuffer() const { return bufferId >= 0; }
    bool isTensor() const { return tensorId >= 0; }
};

/** @brief The constant operands a layer has alongside the value flowing into it, kept in
 *  the layer's own input order with the flowing one removed: the 3 in `x + 3`, Clip's two
 *  bounds, BatchNorm's scale and bias. A layer with none gets count 0.
 */
struct ConstOperand
{
    enum { MAX_CONSTS = 2 };

    bool flowIsFirstInput = true;   //!< the flowing value is the layer's input 0
    int  count            = 0;      //!< how many entries of consts[] are filled
    FusionConst consts[MAX_CONSTS];

    const FusionConst& at(int i) const
    {
        CV_DbgAssert(i >= 0 && i < count);
        return consts[i];
    }
};

/** @brief A ready-made kernel for exactly one expression, when the layer that described
 *  it already owns one. A sink that recognizes it runs it directly instead of walking
 *  the decomposed form. The two must agree: fn(x, params) is the same function the
 *  expression evaluates to.
 */
struct FusionKernel
{
    enum { MAX_PARAMS = 4 };

    ActivationFunc fn = nullptr;
    float params[MAX_PARAMS] = { 0.f, 0.f, 0.f, 0.f };
    int   nparams = 0;

    void set(ActivationFunc f, const std::vector<float>& p)
    {
        CV_Assert(p.size() <= (size_t)MAX_PARAMS);
        fn = f;
        nparams = (int)p.size();
        for (int i = 0; i < nparams; i++)
            params[i] = p[i];
    }
};

struct LayerMathNode
{
    FusionEltwiseOp op = FusionEltwiseOp::CONST;
    int   left = -1, right = -1;
    float scalar = 0.f, scalar2 = 0.f;
    int   bufferId = -1;
};

//! Per-channel buffer ids for a layer whose data an earlier pass folded into members,
//! leaving no Arg. Ids come in request order and hold for the rest of the chain.
struct FusionBufferSink
{
    virtual ~FusionBufferSink() {}
    virtual int add(const Mat& buf) = 0;
};

/** @brief Straight-line description of one layer's elementwise math.
 *
 * A layer builds its math by appending nodes; each call returns the index of
 * the node it added, and an operand is either one of those indices or
 * INPUT_VALUE, standing for the value arriving from the layer being fused into.
 * Since nodes can only be added this way, a math is always well formed.
 */
struct LayerMath
{
    enum { INPUT_VALUE = -1 };

    //! The layer's own kernel for exactly this math, when it has one.
    FusionKernel kernel;

    //! Set by the pass before unfold; null when a math is built standalone.
    FusionBufferSink* buffers = nullptr;

    void setKernel(ActivationFunc fn, const std::vector<float>& params)
    { kernel.set(fn, params); }

    int nodeCount() const { return nodeCount_; }
    const LayerMathNode& nodeAt(int i) const
    {
        CV_DbgAssert(i >= 0 && i < nodeCount_);
        return nodes_[i];
    }

    int constant(float value)
    { return appendNode(FusionEltwiseOp::CONST, INPUT_VALUE, INPUT_VALUE, value, 0.f, -1); }

    int perChannelConstant(int bufferId)
    {
        CV_Assert(bufferId >= 0);
        return appendNode(FusionEltwiseOp::PER_CHANNEL_CONST, INPUT_VALUE, INPUT_VALUE,
                          0.f, 0.f, bufferId);
    }

    //! For a live tensor, at the slot the pass assigned it in the chain's tensorArgs.
    int tensorOperand(int tensorId)
    {
        CV_Assert(tensorId >= 0);
        return appendNode(FusionEltwiseOp::TENSOR, INPUT_VALUE, INPUT_VALUE,
                          0.f, 0.f, tensorId);
    }

    //! For per-channel data the layer owns; @p buf is kept by reference.
    int perChannelConstant(const Mat& buf)
    {
        CV_Assert(buffers != nullptr);
        CV_Assert(!buf.empty() && buf.type() == CV_32F && buf.isContinuous());
        return perChannelConstant(buffers->add(buf));
    }

    int unary(FusionEltwiseOp op, int operand)
    {
        CV_Assert(fusion::detail::arity(op) == 1);
        return appendNode(op, operand, INPUT_VALUE, 0.f, 0.f, -1);
    }

    int binary(FusionEltwiseOp op, int left, int right)
    {
        CV_Assert(fusion::detail::arity(op) == 2);
        return appendNode(op, left, right, 0.f, 0.f, -1);
    }

    int clamp(int operand, float lo, float hi)
    { return appendNode(FusionEltwiseOp::CLAMP, operand, INPUT_VALUE, lo, hi, -1); }

private:
    int appendNode(FusionEltwiseOp op, int a, int b, float s0, float s1, int buf)
    {
        CV_Assert(nodeCount_ < FUSION_MAX_MATH_NODES);
        CV_Assert(a >= INPUT_VALUE && a < nodeCount_);
        CV_Assert(b >= INPUT_VALUE && b < nodeCount_);
        LayerMathNode& nd = nodes_[nodeCount_];
        nd.op = op; nd.left = a; nd.right = b; nd.scalar = s0; nd.scalar2 = s1; nd.bufferId = buf;
        return nodeCount_++;
    }

    int nodeCount_ = 0;
    LayerMathNode nodes_[FUSION_MAX_MATH_NODES];
};

namespace fusion { namespace detail {

inline int sigmoid(LayerMath& r)
{
    const int one      = r.constant(1.f);
    const int minusOne = r.constant(-1.f);
    const int negated  = r.binary(FusionEltwiseOp::MUL, LayerMath::INPUT_VALUE, minusOne);
    const int exponent = r.unary(FusionEltwiseOp::EXP, negated);
    const int denom    = r.binary(FusionEltwiseOp::ADD, one, exponent);
    return r.unary(FusionEltwiseOp::RECIP, denom);
}

inline int abs(LayerMath& r, int operand)
{
    const int minusOne = r.constant(-1.f);
    const int negated  = r.binary(FusionEltwiseOp::MUL, operand, minusOne);
    return r.binary(FusionEltwiseOp::MAX, operand, negated);
}

//! max(x,0) + alpha*(exp(min(x,0)*xScale) - 1). The arms never overlap because
//! exp(0) - 1 is exactly 0, so the one that does not apply contributes nothing.
inline int expLinearUnit(LayerMath& r, float alpha, float xScale)
{
    const int zero    = r.constant(0.f);
    const int one     = r.constant(1.f);
    const int pos     = r.binary(FusionEltwiseOp::MAX, LayerMath::INPUT_VALUE, zero);
    const int clipped = r.binary(FusionEltwiseOp::MIN, LayerMath::INPUT_VALUE, zero);
    int neg = clipped;
    if (xScale != 1.f) {
        const int s   = r.constant(xScale);
        neg           = r.binary(FusionEltwiseOp::MUL, clipped, s);
    }
    const int expNeg  = r.unary(FusionEltwiseOp::EXP, neg);
    const int decayed = r.binary(FusionEltwiseOp::SUB, expNeg, one);
    const int a       = r.constant(alpha);
    const int scaled  = r.binary(FusionEltwiseOp::MUL, a, decayed);
    return r.binary(FusionEltwiseOp::ADD, pos, scaled);
}

inline int gelu(LayerMath& r)
{
    const int half     = r.constant(0.5f);
    const int one      = r.constant(1.f);
    const int invSqrt2 = r.constant(0.70710678118654752440f);
    const int scaled   = r.binary(FusionEltwiseOp::MUL, LayerMath::INPUT_VALUE, invSqrt2);
    const int erfTerm  = r.unary(FusionEltwiseOp::ERF, scaled);
    const int gate     = r.binary(FusionEltwiseOp::ADD, one, erfTerm);
    const int halfX    = r.binary(FusionEltwiseOp::MUL, half, LayerMath::INPUT_VALUE);
    return r.binary(FusionEltwiseOp::MUL, halfX, gate);
}

}} // namespace fusion::detail

struct FusionNode
{
    FusionEltwiseOp op = FusionEltwiseOp::INPUT;
    std::vector<int> inputs;
    float scalar = 0.f;
    float scalar2 = 0.f;
    int constBufferId = -1;

    bool operator==(const FusionNode& o) const noexcept
    {
        return op == o.op && inputs == o.inputs
            && fusion::detail::bits(scalar)  == fusion::detail::bits(o.scalar)
            && fusion::detail::bits(scalar2) == fusion::detail::bits(o.scalar2)
            && constBufferId   == o.constBufferId;
    }
};

struct FusionNodeHash
{
    size_t operator()(const FusionNode& n) const noexcept
    {
        size_t h = std::hash<int>()((int)n.op);
        for (int i : n.inputs) h = h * 1000003u ^ (size_t)i;
        h = h * 1000003u ^ (size_t)fusion::detail::bits(n.scalar);
        h = h * 1000003u ^ (size_t)fusion::detail::bits(n.scalar2);
        h = h * 1000003u ^ (size_t)n.constBufferId;
        return h;
    }
};

class AdjacencyGraph
{
public:
    const std::vector<FusionNode>& nodes() const { return nodes_; }
    size_t size() const { return nodes_.size(); }

    /** @brief The node holding the result, always the last one.
     *
     * internNode() only ever appends, and every node's inputs precede it, so the graph
     * is topologically ordered and the root cannot be anywhere else. extract() rebuilds
     * the same way. -1 while the graph is empty.
     */
    int outputNode() const { return (int)nodes_.size() - 1; }

    std::vector<Mat> constBufs;

    //! Live tensor operands, indexed by each TENSOR node's constBufferId; extract() remaps them.
    std::vector<Arg> tensorArgs;
    //! Parallel to constBufs: the layer declared it per-channel, so do not infer the axis.
    std::vector<uchar> constBufPerChannel;

    //! Set when this whole expression is one layer that already has a kernel.
    FusionKernel kernel;

private:
    std::vector<FusionNode> nodes_;
    int append(FusionNode n)
    {
        nodes_.push_back(std::move(n));
        return (int)nodes_.size() - 1;
    }
    friend class AdjacencyGraphBuilder;
};

class AdjacencyGraphBuilder
{
public:
    AdjacencyGraphBuilder() : gp_(makePtr<AdjacencyGraph>()) {}

    /** @brief Adds one node, reusing an identical existing one if there is
     *  already one in this graph. Returns the node's index either way. */
    int internNode(FusionEltwiseOp op, std::vector<int> inputs, float scalar = 0.f,
                   float scalar2 = 0.f, int constBufferId = -1)
    {
        CV_Assert((gp_->size() == 0) == (op == FusionEltwiseOp::INPUT));
        CV_Assert((int)inputs.size() == fusion::detail::arity(op));
        CV_DbgAssert(gp_->size() < (size_t)FUSION_MAX_ARENA_NODES);
        for (int i : inputs)
            CV_Assert(i >= 0 && i < (int)gp_->size());

        if ((op == FusionEltwiseOp::ADD || op == FusionEltwiseOp::MUL) && inputs[0] > inputs[1])
            std::swap(inputs[0], inputs[1]);

        if (op != FusionEltwiseOp::CONST && op != FusionEltwiseOp::CLAMP) scalar = 0.f;
        if (op != FusionEltwiseOp::CLAMP)                                 scalar2 = 0.f;
        if (op != FusionEltwiseOp::PER_CHANNEL_CONST && op != FusionEltwiseOp::TENSOR)
            constBufferId = -1;

        FusionNode cand;
        cand.op = op;
        cand.inputs = std::move(inputs);
        cand.scalar = scalar;
        cand.scalar2 = scalar2;
        cand.constBufferId = constBufferId;

        std::unordered_map<FusionNode, int, FusionNodeHash>::const_iterator it = interned_.find(cand);
        if (it != interned_.end())
            return it->second;

        const int idx = gp_->append(cand);
        interned_.emplace(std::move(cand), idx);
        return idx;
    }

    size_t size() const { return gp_->size(); }
    const AdjacencyGraph& graph() const { return *gp_; }

    //! Hands over the built graph, checking that @p root really is its output node.
    Ptr<AdjacencyGraph> finish(int root)
    {
        CV_Assert(root == gp_->outputNode());
        return gp_;
    }

    //! The graph as built so far; the builder keeps writing to it.
    Ptr<AdjacencyGraph> sharedGraph() { return gp_; }

private:
    Ptr<AdjacencyGraph> gp_;
    std::unordered_map<FusionNode, int, FusionNodeHash> interned_;
};

namespace fusion {

inline int instantiate(AdjacencyGraphBuilder& builder, int inputNode,
                             const LayerMath& math)
{
    if (inputNode < 0 || math.nodeCount() <= 0)
        return -1;

    int graphIndex[FUSION_MAX_MATH_NODES];
    for (int i = 0; i < math.nodeCount(); i++) {
        const LayerMathNode& nd = math.nodeAt(i);
        const int k = detail::arity(nd.op);
        std::vector<int> inputs;
        inputs.reserve((size_t)k);
        if (k >= 1) inputs.push_back(nd.left  < 0 ? inputNode : graphIndex[nd.left]);
        if (k >= 2) inputs.push_back(nd.right < 0 ? inputNode : graphIndex[nd.right]);
        graphIndex[i] = builder.internNode(nd.op, inputs, nd.scalar, nd.scalar2, nd.bufferId);
    }
    return graphIndex[math.nodeCount() - 1];
}

inline float evalElement(const AdjacencyGraph& g, float x,
                             const std::vector<const float*>& constBufs, int channelIdx)
{
    const std::vector<FusionNode>& nodes = g.nodes();
    const int out = g.outputNode();
    CV_Assert(nodes.size() <= (size_t)FUSION_MAX_EXPR_NODES);

    float v[FUSION_MAX_EXPR_NODES];

    for (int i = 0; i <= out; i++) {
        const FusionNode& n = nodes[i];
        const float a = n.inputs.size() > 0 ? v[n.inputs[0]] : 0.f;
        const float b = n.inputs.size() > 1 ? v[n.inputs[1]] : 0.f;

        switch (n.op) {
        case FusionEltwiseOp::INPUT: v[i] = x; break;
        case FusionEltwiseOp::CONST: v[i] = n.scalar; break;
        case FusionEltwiseOp::PER_CHANNEL_CONST:
            CV_DbgAssert(n.constBufferId >= 0 && n.constBufferId < (int)constBufs.size());
            v[i] = constBufs[n.constBufferId][channelIdx];
            break;
        case FusionEltwiseOp::ADD:   v[i] = a + b; break;
        case FusionEltwiseOp::SUB:   v[i] = a - b; break;
        case FusionEltwiseOp::MUL:   v[i] = a * b; break;
        case FusionEltwiseOp::MAX:   v[i] = a < b ? b : a; break;
        case FusionEltwiseOp::MIN:   v[i] = b < a ? b : a; break;
        case FusionEltwiseOp::ERF:   v[i] = std::erf(a); break;
        case FusionEltwiseOp::TANH:  v[i] = std::tanh(a); break;
        case FusionEltwiseOp::EXP:   v[i] = std::exp(a); break;
        case FusionEltwiseOp::SQRT:  v[i] = std::sqrt(a); break;
        case FusionEltwiseOp::RECIP: v[i] = 1.f / a; break;
        case FusionEltwiseOp::CLAMP:
            v[i] = a < n.scalar ? n.scalar : (a > n.scalar2 ? n.scalar2 : a);
            break;
        case FusionEltwiseOp::TENSOR:
            // Only a sink that opted in is offered one, and it reads the nodes itself.
            CV_Error(Error::StsNotImplemented,
                     "DNN/fusion: TENSOR operand reached the generic interpreter");
        }
    }
    return v[out];
}

namespace detail {

inline int markLive(const AdjacencyGraph& g, int root, std::vector<char>& live)
{
    const std::vector<FusionNode>& nodes = g.nodes();
    CV_Assert(root >= 0 && root < (int)nodes.size());
    live.assign((size_t)root + 1, 0);
    live[root] = 1;
    int nlive = 0;
    for (int i = root; i >= 0; i--) {
        if (!live[i]) continue;
        nlive++;
        for (int in : nodes[i].inputs) {
            CV_DbgAssert(in >= 0 && in < i);
            live[in] = 1;
        }
    }
    return nlive;
}

} // namespace detail

inline Ptr<AdjacencyGraph> extract(const AdjacencyGraph& arena, int root,
                                   const std::vector<Mat>& constBufs,
                                   const std::vector<uchar>& perChannel = std::vector<uchar>(),
                                   const std::vector<Arg>& tensorArgs = std::vector<Arg>())
{
    const std::vector<FusionNode>& src = arena.nodes();
    if (root < 0 || root >= (int)src.size())
        return Ptr<AdjacencyGraph>();

    std::vector<char> live;
    detail::markLive(arena, root, live);
    CV_DbgAssert(src[0].op == FusionEltwiseOp::INPUT);
    if (!live[0])
        return Ptr<AdjacencyGraph>();

    std::vector<int> remap(live.size(), -1);
    // tensorId -> slot in outTensorArgs, so a shorter offer only carries the tensors it reads.
    std::vector<int> tensorRemap(tensorArgs.size(), -1);
    std::vector<Arg> outTensorArgs;
    AdjacencyGraphBuilder out;
    remap[0] = out.internNode(FusionEltwiseOp::INPUT, {});

    std::vector<int> pending(1, root);
    while (!pending.empty()) {
        const int i = pending.back();
        if (remap[i] >= 0) {
            pending.pop_back();
            continue;
        }
        const FusionNode& n = src[i];
        bool waiting = false;
        for (size_t k = n.inputs.size(); k > 0; k--) {
            if (remap[n.inputs[k - 1]] < 0) {
                pending.push_back(n.inputs[k - 1]);
                waiting = true;
            }
        }
        if (waiting)
            continue;
        pending.pop_back();

        if (out.size() >= (size_t)FUSION_MAX_EXPR_NODES)
            return Ptr<AdjacencyGraph>();
        if (n.op == FusionEltwiseOp::PER_CHANNEL_CONST &&
            (n.constBufferId < 0 || n.constBufferId >= (int)constBufs.size()))
            return Ptr<AdjacencyGraph>();
        int constBufferId = n.constBufferId;
        if (n.op == FusionEltwiseOp::TENSOR) {
            if (n.constBufferId < 0 || n.constBufferId >= (int)tensorArgs.size())
                return Ptr<AdjacencyGraph>();
            if (tensorRemap[n.constBufferId] < 0) {
                tensorRemap[n.constBufferId] = (int)outTensorArgs.size();
                outTensorArgs.push_back(tensorArgs[n.constBufferId]);
            }
            constBufferId = tensorRemap[n.constBufferId];
        }
        std::vector<int> ins(n.inputs.size());
        for (size_t k = 0; k < ins.size(); k++)
            ins[k] = remap[n.inputs[k]];
        remap[i] = out.internNode(n.op, ins, n.scalar, n.scalar2, constBufferId);
    }
    if (remap[root] != (int)out.size() - 1)
        return Ptr<AdjacencyGraph>();

    Ptr<AdjacencyGraph> g = out.finish(remap[root]);
    g->constBufs = constBufs;
    g->constBufPerChannel = perChannel;
    g->tensorArgs = outTensorArgs;
    return g;
}


} // namespace fusion

/** @brief What one layer class contributes to fusion. A layer opts in by registering
 *  a table for its own concrete type; the pass looks the table up, so nothing about
 *  fusion has to appear on Layer itself. Either slot may be null.
 */
struct FusionOps
{
    /** @brief States the layer's math as a small expression, so a fusion pass can
     *  absorb it into whatever produces its input.
     *
     *  @param self the layer, known to be of the registered type.
     *  @param out receives the expression, built through LayerMath's emitters.
     *  @param side describes the layer's non-flowing inputs when it has any, e.g.
     *         the constant operand of an Add. Empty for a plain unary op.
     *  @return false if the layer cannot be expressed, which also means it can
     *          never be absorbed. A null slot means the layer never can.
     */
    bool (*unfold)(const Layer* self, LayerMath& out, const ConstOperand& side);

    /** @brief Offers a trailing expression for this layer to absorb into its own
     *  computation. The layer decides; refusing is always safe.
     *
     *  The pass offers the longest chain first and retries with shorter ones, so
     *  an implementation must finish validating before it mutates any state.
     *  @return true if the expression was taken on, in which case the layer is now
     *          responsible for computing it. A null slot means it never takes one.
     */
    bool (*absorb)(Layer* self, const Ptr<AdjacencyGraph>& expr);

    //! Opt-in: lets a chain grow through a live-tensor (TENSOR) side operand. The pass wires
    //! the accepted expression's tensorArgs on as extra layer inputs, in order.
    bool acceptsTensorOperands = false;
};

/** @brief Binds @p ops to one concrete layer class. Last registration for a type wins. */
CV_EXPORTS void registerFusionOps(const std::type_info& layerType, const FusionOps& ops);

/** @brief The table registered for @p layer's most-derived type, or null if it has none. */
CV_EXPORTS const FusionOps* fusionOpsFor(const Layer* layer);

/** @brief Registers @p ops for LayerT the first time it is called for that type.
 *
 * Call it from LayerT's constructor: registration then costs one guarded branch per
 * construction, needs no list of participating types to keep in step, and cannot run
 * before main() the way a namespace-scope registrar object would.
 */
template<typename LayerT>
inline void registerFusionOpsOnce(const FusionOps& ops)
{
    static const bool registered = (registerFusionOps(typeid(LayerT), ops), true);
    CV_UNUSED(registered);
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn

#endif
