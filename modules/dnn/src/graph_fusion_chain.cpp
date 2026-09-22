// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "net_impl.hpp"
#include "adjacency_graph.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;

namespace {

void firstConsumerOf(const vector<Ptr<LayerInfo> >& prog, int nargs,
                     vector<int>& firstConsumer)
{
    firstConsumer.assign((size_t)nargs, -1);
    for (size_t j = 0; j < prog.size(); j++) {
        if (!prog[j]) continue;
        for (Arg in : prog[j]->inputs) {
            if (in.idx > 0 && in.idx < nargs && firstConsumer[in.idx] < 0)
                firstConsumer[in.idx] = (int)j;
        }
    }
}

void producerOf(const vector<Ptr<LayerInfo> >& prog, int nargs, vector<int>& producer)
{
    producer.assign((size_t)nargs, -1);
    for (size_t j = 0; j < prog.size(); j++) {
        if (!prog[j]) continue;
        for (Arg out : prog[j]->outputs) {
            if (out.idx > 0 && out.idx < nargs)
                producer[out.idx] = (int)j;
        }
    }
}

class ChainFuser
{
public:
    ChainFuser(Net::Impl& net, const Ptr<Graph>& graph, const vector<int>& usecounts)
        : net_(net), graph_(graph), usecounts_(usecounts)
    {
        CV_Assert((int)usecounts_.size() == (int)net_.args.size());
        claimed_.assign(prog().size(), false);
        const int inputNode = arena_.internNode(FusionEltwiseOp::INPUT, {});
        CV_Assert(inputNode == 0);
    }
    ChainFuser(const ChainFuser&) = delete;
    ChainFuser& operator=(const ChainFuser&) = delete;

    bool fuse()
    {
        collectChains();
        if (chains_.empty())
            return false;
        freezeArena();
        fuseLongestChains();
        if (nfused_ == 0)
            return false;
        dropAbsorbedLayers();
        return true;
    }

private:
    struct ChainCandidate
    {
        vector<int> layerIdx;
        vector<Arg> constArgs;
        vector<int> rootAfterStep;
        vector<Mat> constBufs;
        //! Parallel to constBufs: set for a buffer the layer declared per-channel outright.
        vector<uchar> constBufPerChannel;
        //! First step's kernel; a chain that lands at one step runs it instead of the DAG.
        FusionKernel singleStepKernel;
        //! Live tensors the steps read, in the order their slots were handed out.
        vector<Arg> tensorArgs;
    };

    const vector<Ptr<LayerInfo> >& prog() const { return graph_->prog(); }

    bool isFusableConstArg(Arg a, bool& isScalar, float& scalarVal) const
    {
        if (!net_.isConstArg(a))
            return false;
        Mat t = net_.argTensor(a);
        if (t.total() == 1) {
            if (t.type() == CV_32F) { isScalar = true; scalarVal = t.ptr<float>()[0]; return true; }
            if (t.type() == CV_64F) { isScalar = true; scalarVal = (float)t.ptr<double>()[0]; return true; }
            return false;
        }
        if (t.type() != CV_32F)
            return false;
        int nonUnit = 0;
        for (int d = 0; d < t.dims; d++)
            if (t.size[d] != 1) nonUnit++;
        if (nonUnit != 1)
            return false;
        isScalar = false;
        return true;
    }

    //! Releases the slots a refused step claimed, so the next step's ids start where it did.
    static void dropSlotsAfter(ChainCandidate& c, size_t keepBufs, size_t keepTensors)
    {
        c.constBufs.resize(keepBufs);
        c.constArgs.resize(keepBufs);
        c.constBufPerChannel.resize(keepBufs);
        c.tensorArgs.resize(keepTensors);
    }

    //! The slot @p a occupies in the chain's live-tensor list, adding it if new.
    static int tensorSlotFor(Arg a, ChainCandidate& c)
    {
        for (size_t k = 0; k < c.tensorArgs.size(); k++) {
            if (c.tensorArgs[k].idx == a.idx)
                return (int)k;
        }
        c.tensorArgs.push_back(a);
        return (int)c.tensorArgs.size() - 1;
    }

    //! The host reads it, so a layer before @p anchor must have produced it, and it must not
    //! be one the graph still owes its caller. Shape and type belong to the sink: they are
    //! unknown here, since an intermediate's ArgData is only filled once the graph runs.
    bool isFusableTensorArg(Arg a, size_t anchor) const
    {
        if (a.idx <= 0 || a.idx >= (int)producerOf_.size())
            return false;
        const int prod = producerOf_[a.idx];
        if (prod < 0 || prod >= (int)anchor)
            return false;
        return externalArgs_.find(a.idx) == externalArgs_.end();
    }

    //! The slot @p a occupies in the chain's per-channel buffer list, adding it if new.
    int bufferSlotFor(Arg a, ChainCandidate& c) const
    {
        for (size_t k = 0; k < c.constArgs.size(); k++) {
            if (c.constArgs[k].idx == a.idx)
                return (int)k;
        }
        c.constBufs.push_back(net_.argTensor(a));
        c.constArgs.push_back(a);
        c.constBufPerChannel.push_back(0);
        return (int)c.constBufs.size() - 1;
    }

    //! Per-channel data the layer folded into its members has no Arg, so it is slotted here
    //! and its ids join the ones readConstOperand() took from the layer's inputs.
    static bool addOwnedBuffers(const FusionOps* ops, Layer* l, ChainCandidate& c,
                                ConstOperand& out)
    {
        if (!ops->ownedBuffers)
            return true;
        std::vector<Mat> owned;
        if (!ops->ownedBuffers(l, owned))
            return false;
        if (out.count + (int)owned.size() > ConstOperand::MAX_CONSTS)
            return false;
        for (const Mat& buf : owned) {
            c.constBufs.push_back(buf);
            c.constArgs.push_back(Arg());
            c.constBufPerChannel.push_back(1);
            out.consts[out.count++].bufferId = (int)c.constBufs.size() - 1;
        }
        return true;
    }

    bool readConstOperand(const Ptr<LayerInfo>& L, Arg cur, ChainCandidate& c,
                          ConstOperand& out, size_t anchor, bool acceptsTensors) const
    {
        vector<Arg> sideInputs;
        for (Arg in : L->inputs) {
            if (in.idx == cur.idx || in.idx == 0)
                continue;
            sideInputs.push_back(in);
        }
        if (sideInputs.empty())
            return true;
        if (sideInputs.size() > (size_t)ConstOperand::MAX_CONSTS)
            return false;

        out.flowIsFirstInput = !L->inputs.empty() && L->inputs[0].idx == cur.idx;

        for (size_t i = 0; i < sideInputs.size(); i++) {
            bool isScalar = false;
            float scalarVal = 0.f;
            if (isFusableConstArg(sideInputs[i], isScalar, scalarVal)) {
                if (isScalar)
                    out.consts[i].value = scalarVal;
                else
                    out.consts[i].bufferId = bufferSlotFor(sideInputs[i], c);
                continue;
            }
            if (!acceptsTensors || !isFusableTensorArg(sideInputs[i], anchor))
                return false;
            out.consts[i].tensorId = tensorSlotFor(sideInputs[i], c);
        }
        out.count = (int)sideInputs.size();
        return true;
    }

    static bool isAbsorbableMath(Layer* l)
    {
        const FusionOps* ops = fusionOpsFor(l);
        if (!ops || !ops->unfold)
            return false;
        // The operand count and kind are not known yet, so try every combination.
        for (int n = 0; n <= ConstOperand::MAX_CONSTS; n++) {
            for (int kind = 0; kind < 3; kind++) {
                if (kind > 0 && n == 0)
                    continue;   // nothing to mark, so it repeats the probe above
                LayerMath r;
                ConstOperand probe;
                probe.count = n;
                for (int i = 0; i < n; i++) {
                    if (kind == 1)
                        probe.consts[i].bufferId = i;
                    else if (kind == 2)
                        probe.consts[i].tensorId = i;
                }
                if (ops->unfold(l, r, probe))
                    return true;
            }
        }
        return false;
    }

    void growChain(size_t anchor, bool acceptsTensors, ChainCandidate& c)
    {
        CV_Assert(!arenaPtr_);

        int chainRoot = 0;
        Arg curArg = prog()[anchor]->outputs[0];
        int producer = (int)anchor;

        while (curArg.idx > 0 && curArg.idx < (int)usecounts_.size() &&
               usecounts_[curArg.idx] == 1) {
            const int j = firstConsumer_[curArg.idx];
            if (j <= producer || j >= (int)prog().size() || claimed_[j])
                break;

            const Ptr<LayerInfo>& L = prog()[j];
            if (!L || L->outputs.size() != 1 || L->subgraphs())
                break;
            Layer* l = dynamic_cast<Layer*>(L.get());
            if (!l)
                break;

            if (arena_.size() >= (size_t)FUSION_MAX_ARENA_NODES) {
                CV_LOG_DEBUG(NULL, cv::format("[fusion] arena full (%d nodes), chain truncated",
                                              (int)arena_.size()));
                break;
            }

            const FusionOps* ops = fusionOpsFor(l);
            if (!ops || !ops->unfold)
                break;

            const size_t savedBufs = c.constBufs.size();
            const size_t savedTensors = c.tensorArgs.size();
            ConstOperand side;
            LayerMath r;
            if (!readConstOperand(L, curArg, c, side, anchor, acceptsTensors) ||
                !addOwnedBuffers(ops, l, c, side) ||
                !ops->unfold(l, r, side)) {
                dropSlotsAfter(c, savedBufs, savedTensors);
                break;
            }

            if (r.nodeCount() == 0) {
                // A kernel states the whole expression, so the step can neither extend a
                // chain nor be extended, and is taken only as the sole step. chainRoot is
                // still the arena's INPUT node, which is the expression extract() returns.
                if (!r.kernel.fn || !c.rootAfterStep.empty()) {
                    dropSlotsAfter(c, savedBufs, savedTensors);
                    break;
                }
                c.singleStepKernel = r.kernel;
                c.rootAfterStep.push_back(chainRoot);
                c.layerIdx.push_back(j);
                break;
            }

            const int next = fusion::instantiate(arena_, chainRoot, r);
            if (next < 0 || fusion::detail::markLive(arena_.graph(), next, reachScratch_) > FUSION_MAX_EXPR_NODES) {
                dropSlotsAfter(c, savedBufs, savedTensors);
                break;
            }

            CV_DbgAssert(next > chainRoot);
            if (c.rootAfterStep.empty())
                c.singleStepKernel = r.kernel;
            chainRoot = next;
            c.rootAfterStep.push_back(next);
            c.layerIdx.push_back(j);
            curArg = L->outputs[0];
            producer = j;
        }
    }

    void collectChains()
    {
        const int nargs = (int)net_.args.size();
        firstConsumerOf(prog(), nargs, firstConsumer_);
        producerOf(prog(), nargs, producerOf_);
        for (Arg out : graph_->outputs())
            externalArgs_.insert(out.idx);

        for (size_t i = 0; i < prog().size(); i++) {
            const Ptr<LayerInfo>& L = prog()[i];
            if (!L || claimed_[i])
                continue;
            if (L->outputs.size() != 1 || L->subgraphs())
                continue;
            Layer* anchor = dynamic_cast<Layer*>(L.get());
            if (!anchor || isAbsorbableMath(anchor))
                continue;
            // A layer that cannot take anything on is no use as the head of a chain.
            const FusionOps* anchorOps = fusionOpsFor(anchor);
            if (!anchorOps || !anchorOps->absorb)
                continue;

            ChainCandidate c;
            c.layerIdx.push_back((int)i);
            growChain(i, anchorOps->acceptsTensorOperands, c);

            if (c.rootAfterStep.empty())
                continue;
            for (int n : c.layerIdx)
                claimed_[n] = true;
            chains_.push_back(c);
        }
    }

    //! Carries a kernel when there is no expression. The lone INPUT node is an identity,
    //! so this is only ever built with a kernel attached.
    void freezeArena()
    {
        arenaPtr_ = arena_.sharedGraph();
    }

    void fuseLongestChains()
    {
        dropped_.assign(prog().size(), false);

        for (ChainCandidate& c : chains_) {
            const Ptr<LayerInfo>& anchorInfo = prog()[c.layerIdx[0]];
            Layer* sink = dynamic_cast<Layer*>(anchorInfo.get());
            if (!sink)
                continue;

            const FusionOps* sinkOps = fusionOpsFor(sink);
            if (!sinkOps || !sinkOps->absorb)
                continue;

            size_t accepted = 0;
            Ptr<AdjacencyGraph> expr;
            for (size_t n = c.rootAfterStep.size(); n >= 1; n--) {
                expr = fusion::extract(*arenaPtr_, c.rootAfterStep[n - 1],
                                       c.constBufs, c.constBufPerChannel, c.tensorArgs);
                if (!expr)
                    continue;
                if (n == 1)
                    expr->kernel = c.singleStepKernel;
                if (sinkOps->absorb(sink, expr)) {
                    accepted = n;
                    break;
                }
            }
            if (accepted == 0) {
                CV_LOG_DEBUG(NULL, cv::format("[fusion] refused %s (+%d)",
                                              anchorInfo->type.c_str(),
                                              (int)c.rootAfterStep.size()));
                continue;
            }

            // Wired on here rather than in absorb(), which the retry loop above may call
            // more than once per chain.
            for (Arg t : expr->tensorArgs)
                anchorInfo->inputs.push_back(t);

            anchorInfo->outputs[0] = prog()[c.layerIdx[accepted]]->outputs[0];
            for (size_t k = 1; k <= accepted; k++)
                dropped_[c.layerIdx[k]] = true;
            nfused_++;

            CV_LOG_DEBUG(NULL, cv::format("[fusion] FUSED %s +%d of %d",
                                          anchorInfo->type.c_str(),
                                          (int)accepted, (int)c.rootAfterStep.size()));
        }
    }

    void dropAbsorbedLayers()
    {
        const size_t nops = prog().size();
        vector<Ptr<LayerInfo> > newprog;
        newprog.reserve(nops);
        for (size_t i = 0; i < nops; i++) {
            if (!dropped_[i] && prog()[i])
                newprog.push_back(prog()[i]);
        }
        graph_->setProg(newprog);

        CV_LOG_DEBUG(NULL, cv::format("fuseChains: fused %d chain(s) in graph '%s', arena %d nodes",
                                      nfused_, graph_->name().c_str(), (int)arenaPtr_->size()));
    }

    Net::Impl& net_;
    const Ptr<Graph>& graph_;
    const vector<int>& usecounts_;

    AdjacencyGraphBuilder arena_;
    Ptr<AdjacencyGraph>   arenaPtr_;
    vector<int>        firstConsumer_, producerOf_;
    std::set<int>      externalArgs_;
    vector<bool>       claimed_, dropped_;
    vector<ChainCandidate>  chains_;
    vector<char>       reachScratch_;
    int nfused_ = 0;
};

bool fuseChainsInGraph(Net::Impl& net, const Ptr<Graph>& graph,
                       const vector<int>& usecounts)
{
    if (!graph)
        return false;

    bool subFused = false;
    for (const Ptr<LayerInfo>& layer : graph->prog()) {
        if (!layer) continue;
        if (vector<Ptr<Graph> >* subs = layer->subgraphs()) {
            for (Ptr<Graph>& g : *subs) {
                if (fuseChainsInGraph(net, g, usecounts))
                    subFused = true;
            }
        }
    }

    vector<int> recounted;
    if (subFused)
        net.useCounts(recounted);

    ChainFuser fuser(net, graph, subFused ? recounted : usecounts);
    const bool fusedHere = fuser.fuse();
    return fusedHere || subFused;
}

} // namespace

void Net::Impl::fuseChains()
{
    if (!mainGraph)
        return;

    // First: it deletes a layer, and if the chain pass absorbed the BatchNorm backwards into
    // the preceding conv it would find nothing left.
    fuseBN();

    // Order is load-bearing: each pass matches shapes the one before it leaves.
    fuseAttention();
    fuseMatMulConstBToGemm();
    fuseSharedInputGemm();
    fuseReshapeTranspose();
    fuseTransposeMatMul();
    fuseScaleSoftmax();

    // A sink runs the math it absorbs in its own CPU kernel. setPreferableTarget resolves
    // only CPU and CUDA while mainGraph is set, so nothing reaches here on OpenCL today.
    if (!IS_DNN_OPENCL_TARGET(preferableTarget)) {
        // Only a one-step chain carries the layer's kernel, so absorbing a step can
        // expose a fusable next one.
        vector<int> usecounts;
        for (int iter = 0; iter < 10; iter++) {
            useCounts(usecounts);
            if (!fuseChainsInGraph(*this, mainGraph, usecounts))
                break;
        }
    }

    // Leaves a whole layer behind rather than handing math to a sink, so unlike the
    // chain pass it is not restricted to the CPU path.
    fuseInstanceNormAffine();
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
