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
        ActivationFunc singleStepKernel = nullptr;
        float singleStepKernelParams[LayerMath::MAX_KERNEL_PARAMS] = { 0.f, 0.f, 0.f, 0.f };
        int   singleStepKernelParamCount = 0;
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

    bool readConstOperand(const Ptr<LayerInfo>& L, Arg cur, ChainCandidate& c, ConstOperand& out) const
    {
        vector<Arg> sideInputs;
        for (Arg in : L->inputs) {
            if (in.idx == cur.idx || in.idx == 0)
                continue;
            sideInputs.push_back(in);
        }
        if (sideInputs.empty())
            return true;
        if (sideInputs.size() > 2)
            return false;

        out.flowIsFirstInput = !L->inputs.empty() && L->inputs[0].idx == cur.idx;

        if (sideInputs.size() == 2) {
            bool s0 = false, s1 = false;
            float v0 = 0.f, v1 = 0.f;
            if (!isFusableConstArg(sideInputs[0], s0, v0) || !s0)
                return false;
            if (!isFusableConstArg(sideInputs[1], s1, v1) || !s1)
                return false;
            out.hasValue = true;
            out.value = v0;
            out.value2 = v1;
            return true;
        }

        bool isScalar = false;
        float scalarVal = 0.f;
        if (!isFusableConstArg(sideInputs[0], isScalar, scalarVal))
            return false;
        out.hasValue = true;
        if (isScalar) {
            out.value = scalarVal;
            return true;
        }
        for (size_t k = 0; k < c.constArgs.size(); k++) {
            if (c.constArgs[k].idx == sideInputs[0].idx) {
                out.bufferId = (int)k;
                return true;
            }
        }
        c.constArgs.push_back(sideInputs[0]);
        out.bufferId = (int)c.constArgs.size() - 1;
        return true;
    }

    static bool isAbsorbableMath(Layer* l)
    {
        LayerMath r;
        ConstOperand anyConstant;
        anyConstant.hasValue = true;
        return l->unfoldOp(r, anyConstant);
    }

    void growChain(size_t anchor, ChainCandidate& c)
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

            const size_t savedSlots = c.constArgs.size();
            ConstOperand side;
            LayerMath r;
            if (!readConstOperand(L, curArg, c, side) || !l->unfoldOp(r, side)) {
                c.constArgs.resize(savedSlots);
                break;
            }

            const int next = fusion::instantiate(arena_, chainRoot, r);
            if (next < 0 || fusion::detail::markLive(arena_.graph(), next, reachScratch_) > FUSION_MAX_EXPR_NODES) {
                c.constArgs.resize(savedSlots);
                break;
            }

            CV_DbgAssert(next > chainRoot);
            if (c.rootAfterStep.empty()) {
                c.singleStepKernel = r.kernel;
                c.singleStepKernelParamCount = r.kernelParamCount;
                for (int q = 0; q < r.kernelParamCount; q++)
                    c.singleStepKernelParams[q] = r.kernelParams[q];
            }
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

        for (size_t i = 0; i < prog().size(); i++) {
            const Ptr<LayerInfo>& L = prog()[i];
            if (!L || claimed_[i])
                continue;
            if (L->outputs.size() != 1 || L->subgraphs())
                continue;
            Layer* anchor = dynamic_cast<Layer*>(L.get());
            if (!anchor || isAbsorbableMath(anchor))
                continue;

            ChainCandidate c;
            c.layerIdx.push_back((int)i);
            growChain(i, c);

            if (c.rootAfterStep.empty())
                continue;
            for (int n : c.layerIdx)
                claimed_[n] = true;
            chains_.push_back(c);
        }
    }

    void freezeArena()
    {
        arenaPtr_ = arena_.sharedGraph();
        for (ChainCandidate& c : chains_) {
            c.constBufs.reserve(c.constArgs.size());
            for (Arg a : c.constArgs)
                c.constBufs.push_back(net_.argTensor(a));
        }
    }

    void fuseLongestChains()
    {
        dropped_.assign(prog().size(), false);

        for (ChainCandidate& c : chains_) {
            const Ptr<LayerInfo>& anchorInfo = prog()[c.layerIdx[0]];
            Layer* sink = dynamic_cast<Layer*>(anchorInfo.get());
            if (!sink)
                continue;

            size_t accepted = 0;
            for (size_t n = c.rootAfterStep.size(); n >= 1; n--) {
                Ptr<AdjacencyGraph> expr = fusion::extract(*arenaPtr_, c.rootAfterStep[n - 1], c.constBufs);
                if (!expr)
                    continue;
                if (n == 1) {
                    expr->kernel = c.singleStepKernel;
                    expr->kernelParamCount = c.singleStepKernelParamCount;
                    for (int q = 0; q < c.singleStepKernelParamCount; q++)
                        expr->kernelParams[q] = c.singleStepKernelParams[q];
                }
                if (sink->absorbMath(expr)) {
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
    vector<int>        firstConsumer_;
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
    // Sinks apply the fused math on the CPU path only; on an OpenCL target the
    // absorbed layers would be dropped and their math never run.
    if (IS_DNN_OPENCL_TARGET(preferableTarget))
        return;
    vector<int> usecounts;
    useCounts(usecounts);
    fuseChainsInGraph(*this, mainGraph, usecounts);
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
