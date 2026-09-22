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
        //! The one step states a kernel and no expression, so it has no arena node.
        bool kernelOnly = false;
        //! Live tensors the steps read, in the order their slots were handed out.
        vector<Arg> tensorArgs;
    };

    //! Buffer slots for per-channel data an earlier pass folded into members, leaving no Arg.
    struct BufferSink CV_FINAL : public FusionBufferSink
    {
        ChainCandidate* c = nullptr;

        int add(const Mat& buf) CV_OVERRIDE
        {
            c->constBufs.push_back(buf);
            c->constArgs.push_back(Arg());
            c->constBufPerChannel.push_back(1);
            return (int)c->constBufs.size() - 1;
        }
    };

    //! Sink for isAbsorbableMath()'s probe, which discards whatever the layer states.
    struct DiscardingSink CV_FINAL : public FusionBufferSink
    {
        int nbufs = 0;

        int add(const Mat&) CV_OVERRIDE { return nbufs++; }
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
        // The probe discards what it names, but still needs a sink or a layer that owns
        // its per-channel data cannot answer.
        DiscardingSink probeSink;
        // We don't know yet how many operands this layer will get, nor whether they will be
        // constants or live tensors, so try each count both ways.
        for (int n = 0; n <= ConstOperand::MAX_CONSTS; n++) {
            for (int asTensor = 0; asTensor < 2; asTensor++) {
                if (asTensor && n == 0)
                    continue;   // nothing to mark, so it would repeat the probe above
                LayerMath r;
                r.buffers = &probeSink;
                ConstOperand probe;
                probe.count = n;
                for (int i = 0; asTensor && i < n; i++)
                    probe.consts[i].tensorId = i;
                if (ops->unfold(l, r, probe))
                    return true;
            }
        }
        return false;
    }

    void growChain(size_t anchor, ChainCandidate& c)
    {
        CV_Assert(!arenaPtr_);

        BufferSink sink;
        sink.c = &c;

        Layer* anchorLayer = dynamic_cast<Layer*>(prog()[anchor].get());
        const FusionOps* anchorOps = anchorLayer ? fusionOpsFor(anchorLayer) : nullptr;
        const bool acceptsTensors = anchorOps && anchorOps->acceptsTensorOperands;

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
            r.buffers = &sink;
            if (!readConstOperand(L, curArg, c, side, anchor, acceptsTensors) ||
                !ops->unfold(l, r, side)) {
                dropSlotsAfter(c, savedBufs, savedTensors);
                break;
            }

            if (r.nodeCount() == 0) {
                // A kernel states the whole expression, so the step can neither extend a
                // chain nor be extended, and is taken only as the sole step.
                if (!r.kernel.fn || !c.rootAfterStep.empty()) {
                    dropSlotsAfter(c, savedBufs, savedTensors);
                    break;
                }
                c.kernelOnly = true;
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
            growChain(i, c);

            if (c.rootAfterStep.empty())
                continue;
            for (int n : c.layerIdx)
                claimed_[n] = true;
            chains_.push_back(c);
        }
    }

    //! Carries a kernel when there is no expression. The lone INPUT node is an identity,
    //! so this is only ever built with a kernel attached.
    static Ptr<AdjacencyGraph> kernelEnvelope(const FusionKernel& kernel)
    {
        CV_Assert(kernel.fn != nullptr);
        AdjacencyGraphBuilder b;
        const int root = b.internNode(FusionEltwiseOp::INPUT, {});
        Ptr<AdjacencyGraph> g = b.finish(root);
        g->kernel = kernel;
        return g;
    }

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
            Ptr<AdjacencyGraph> acceptedExpr;
            for (size_t n = c.rootAfterStep.size(); n >= 1; n--) {
                Ptr<AdjacencyGraph> expr = c.kernelOnly ? kernelEnvelope(c.singleStepKernel)
                    : fusion::extract(*arenaPtr_, c.rootAfterStep[n - 1],
                                      c.constBufs, c.constBufPerChannel, c.tensorArgs);
                if (!expr)
                    continue;
                if (n == 1)
                    expr->kernel = c.singleStepKernel;
                if (sinkOps->absorb(sink, expr)) {
                    accepted = n;
                    acceptedExpr = expr;
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
            for (Arg t : acceptedExpr->tensorArgs)
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

//! Reshape+InstanceNorm(1,0)+Reshape+Mul+Add collapsed into one layer. It matches
//! backward from the Add and can swap the survivor's type, so it is not a chain.
struct InstanceNormAffineFusion
{
    InstanceNormAffineFusion(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse()
    {
        int i, niter = 10;
        netimpl->useCounts(usecounts);
        for (i = 0; i < niter; i++) {
            bool fused_any = fuseGraph(netimpl->mainGraph);
            if (!fused_any)
                break;
        }
    }

    template<typename _LayerType> _LayerType*
    getLayer(std::vector<Ptr<LayerInfo> >& newprog, int op_idx) const
    {
        return op_idx >= 0 ? dynamic_cast<_LayerType*>(newprog.at(op_idx).get()) : 0;
    }

    bool fuseGraph(Ptr<Graph>& graph)
    {
        vector<Arg> removed_args;
        bool modified = false;
        const std::vector<Ptr<LayerInfo> >& prog = graph->prog();
        size_t i, nargs = netimpl->args.size(), nops = prog.size();
        std::vector<int> producer_of(nargs, -1);
        std::vector<Ptr<LayerInfo> > newprog;

        for (i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& layer = prog[i];
            Layer* layer_ptr = (Layer*)layer.get();
            int fused_layer_idx = -1;
            std::vector<Ptr<Graph> >* subgraphs = layer->subgraphs();
            if (subgraphs) {
                for (Ptr<Graph>& g: *subgraphs) {
                    if (fuseGraph(g))
                        modified = true;
                }
            }
            const std::vector<Arg>& inputs = layer->inputs;
            const std::vector<Arg>& outputs = layer->outputs;
            size_t ninputs = inputs.size();
            removed_args.clear();

            NaryEltwiseLayer* elemwise = dynamic_cast<NaryEltwiseLayer*>(layer_ptr);

            // fuse Reshape + InstanceNorm(scale=ones,bias=zeros) + Reshape + Mul + Add
            if (elemwise && elemwise->op == NaryEltwiseLayer::OPERATION::ADD &&
                ninputs == 2) {
                int mul_input_idx = -1;
                Arg add_bias_arg;
                for (int k = 0; k < 2; k++) {
                    int pidx = producer_of.at(inputs[k].idx);
                    NaryEltwiseLayer* mul = getLayer<NaryEltwiseLayer>(newprog, pidx);
                    if (mul && mul->op == NaryEltwiseLayer::OPERATION::PROD) {
                        mul_input_idx = k;
                        add_bias_arg = inputs[1 - k];
                        break;
                    }
                }
                if (mul_input_idx >= 0 && netimpl->isConstArg(add_bias_arg)) {
                    Arg mul_out = inputs[mul_input_idx];
                    int mul_idx = producer_of.at(mul_out.idx);
                    NaryEltwiseLayer* mul = getLayer<NaryEltwiseLayer>(newprog, mul_idx);
                    if (mul && mul->inputs.size() == 2 &&
                        usecounts.at(mul_out.idx) == 1) {
                        int reshape2_input_idx = -1;
                        Arg mul_scale_arg;
                        for (int k = 0; k < 2; k++) {
                            int pidx = producer_of.at(mul->inputs[k].idx);
                            if (getLayer<Reshape2Layer>(newprog, pidx)) {
                                reshape2_input_idx = k;
                                mul_scale_arg = mul->inputs[1 - k];
                                break;
                            }
                        }
                        if (reshape2_input_idx >= 0 && netimpl->isConstArg(mul_scale_arg)) {
                            Arg reshape2_out = mul->inputs[reshape2_input_idx];
                            int reshape2_idx = producer_of.at(reshape2_out.idx);
                            Reshape2Layer* reshape2_lyr = getLayer<Reshape2Layer>(newprog, reshape2_idx);
                            if (reshape2_lyr && reshape2_lyr->inputs.size() >= 1 &&
                                usecounts.at(reshape2_out.idx) == 1) {
                                Arg reshape2_inp = reshape2_lyr->inputs[0];
                                int instnorm_idx = producer_of.at(reshape2_inp.idx);
                                InstanceNormLayer* instnorm = getLayer<InstanceNormLayer>(newprog, instnorm_idx);
                                if (instnorm && instnorm->inputs.size() == 3 &&
                                    usecounts.at(reshape2_inp.idx) == 1) {
                                    Arg instnorm_inp = instnorm->inputs[0];
                                    int reshape1_idx = producer_of.at(instnorm_inp.idx);
                                    Reshape2Layer* reshape1_lyr = getLayer<Reshape2Layer>(newprog, reshape1_idx);
                                    if (reshape1_lyr && reshape1_lyr->outputs.size() == 1 &&
                                        usecounts.at(instnorm_inp.idx) == 1) {
                                        Mat in_scale = netimpl->isConstArg(instnorm->inputs[1]) ?
                                                       netimpl->argTensor(instnorm->inputs[1]) : Mat();
                                        Mat in_bias = netimpl->isConstArg(instnorm->inputs[2]) ?
                                                      netimpl->argTensor(instnorm->inputs[2]) : Mat();
                                        bool valid = !in_scale.empty() && !in_bias.empty() &&
                                                     in_scale.type() == CV_32F && in_bias.type() == CV_32F;
                                        if (valid) {
                                            const float* sp = in_scale.ptr<float>();
                                            const float* bp = in_bias.ptr<float>();
                                            bool all_ones = true, all_zeros = true;
                                            for (size_t k = 0; k < in_scale.total() && all_ones; k++)
                                                all_ones = (std::abs(sp[k] - 1.f) < 1e-6f);
                                            for (size_t k = 0; k < in_bias.total() && all_zeros; k++)
                                                all_zeros = (std::abs(bp[k]) < 1e-6f);
                                            if (all_ones && all_zeros) {
                                                Arg orig_inp = reshape1_lyr->inputs[0];
                                                Mat mul_scale_mat = netimpl->argTensor(mul_scale_arg);
                                                if (in_scale.total() == mul_scale_mat.total()) {
                                                    // Channel dim preserved — fuse into InstanceNorm
                                                    instnorm->inputs[0] = orig_inp;
                                                    instnorm->inputs[1] = mul_scale_arg;
                                                    instnorm->inputs[2] = add_bias_arg;
                                                } else {
                                                    // Channel dim changed (e.g. [1,C,H,W]->[1,1,C*H*W]):
                                                    // original is global norm + per-channel affine.
                                                    // Replace with GroupNorm(num_groups=reshaped_C).
                                                    int num_groups = (int)in_scale.total();
                                                    LayerParams gnparams;
                                                    gnparams.name = instnorm->name;
                                                    gnparams.type = "GroupNormalization";
                                                    gnparams.set("epsilon", instnorm->epsilon);
                                                    gnparams.set("num_groups", num_groups);
                                                    Ptr<LayerInfo> gnlayer = GroupNormLayer::create(gnparams);
                                                    gnlayer->netimpl = netimpl;
                                                    gnlayer->inputs = {orig_inp, mul_scale_arg, add_bias_arg};
                                                    newprog[instnorm_idx] = gnlayer;
                                                }
                                                fused_layer_idx = instnorm_idx;
                                                removed_args.push_back(instnorm_inp);
                                                removed_args.push_back(reshape2_inp);
                                                removed_args.push_back(reshape2_out);
                                                removed_args.push_back(mul_out);
                                                newprog[reshape1_idx] = Ptr<LayerInfo>();
                                                newprog[reshape2_idx] = Ptr<LayerInfo>();
                                                newprog[mul_idx] = Ptr<LayerInfo>();
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (fused_layer_idx >= 0) {
                modified = true;
                Layer* fused_layer = (Layer*)newprog[fused_layer_idx].get();
                fused_layer->outputs = outputs;
                for (Arg new_out: outputs)
                    producer_of[new_out.idx] = fused_layer_idx;
                for (Arg old_out: removed_args) {
                    usecounts.at(old_out.idx) = 0;
                    producer_of.at(old_out.idx) = -1;
                }
            } else {
                for (auto out: outputs)
                    producer_of[out.idx] = (int)newprog.size();
                newprog.push_back(layer);
            }
        }

        if (modified) {
            size_t i, j = 0, newops = newprog.size();
            for (i = 0; i < newops; i++) {
                if (!newprog[i].empty()) {
                    if (j < i)
                        newprog[j] = newprog[i];
                    j++;
                }
            }
            newprog.resize(j);
            //printf("fused some ops in graph %s. size before: %zu ops, size after: %zu ops\n",
            //       graph->name().data(), nops, j);
            graph->setProg(newprog);
        }

        return modified;
    }

    Net::Impl* netimpl;
    vector<int> usecounts;
};

} // namespace

void Net::Impl::fuseChains()
{
    if (!mainGraph)
        return;

    // Sinks apply the fused math on the CPU path only; on an OpenCL target the
    // absorbed layers would be dropped and their math never run.
    if (!IS_DNN_OPENCL_TARGET(preferableTarget)) {
        vector<int> usecounts;
        useCounts(usecounts);
        fuseChainsInGraph(*this, mainGraph, usecounts);
    }

    // Leaves a whole layer behind rather than handing math to a sink, so unlike the
    // chain pass it is not restricted to the CPU path.
    InstanceNormAffineFusion(this).fuse();
}

CV__DNN_INLINE_NS_END
}} // namespace cv::dnn
