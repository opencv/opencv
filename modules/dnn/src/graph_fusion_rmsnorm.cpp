// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Fuses the decomposed RMSNorm that HuggingFace/optimum exports emit:
//   X * (1 / Sqrt(ReduceMean(Pow(X, 2), [-1]) + eps)) * scale  ->  RMSNormalization(X, scale)

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

struct ModelFusionRMSNorm
{
    explicit ModelFusionRMSNorm(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse() { fuseGraph(netimpl->mainGraph); }

    static NaryEltwiseLayer* nary(const Ptr<LayerInfo>& layer, NaryEltwiseLayer::OPERATION op)
    {
        NaryEltwiseLayer* n = dynamic_cast<NaryEltwiseLayer*>(layer.get());
        return n && n->op == op ? n : nullptr;
    }

    bool constScalar(Arg a, float& v) const
    {
        if (!netimpl->isConstArg(a)) return false;
        const Mat& t = netimpl->argTensor(a);
        // A one-element tensor of higher rank would raise X's rank, which the fused layer keeps.
        if (t.total() != 1 || t.dims > 1 || t.type() != CV_32F) return false;
        v = t.ptr<float>()[0];
        return true;
    }

    // The operand of a 2-input layer that is not `known`, or an invalid Arg if neither is.
    static Arg otherInput(const Ptr<LayerInfo>& layer, Arg known)
    {
        if (layer->inputs.size() != 2) return Arg();
        if (layer->inputs[0].idx == known.idx) return layer->inputs[1];
        if (layer->inputs[1].idx == known.idx) return layer->inputs[0];
        return Arg();
    }

    bool fuseGraph(Ptr<Graph>& graph)
    {
        const vector<Ptr<LayerInfo>>& prog = graph->prog();
        size_t nops = prog.size();
        bool modified = false;

        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            vector<Ptr<Graph>>* subgraphs = prog[i]->subgraphs();
            if (subgraphs) {
                for (Ptr<Graph>& g : *subgraphs)
                    if (fuseGraph(g)) modified = true;
            }
        }

        vector<int> usecounts;
        netimpl->useCounts(usecounts);

        std::set<int> externalArgs;
        for (Arg out : graph->outputs())
            externalArgs.insert(out.idx);

        std::map<int, int> consumer, nconsumers;
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            for (Arg in : prog[i]->inputs) {
                consumer[in.idx] = (int)i;
                nconsumers[in.idx]++;
            }
        }

        vector<bool> dropped(nops, false);
        vector<Ptr<LayerInfo>> newprog = prog;

        // One link of the chain: `a` feeds exactly one surviving single-output op and is
        // not a graph output, which is what makes the removal self-contained.
        auto step = [&](Arg a, int& op) -> bool {
            if (externalArgs.count(a.idx) || usecounts[a.idx] != 1) return false;
            auto nit = nconsumers.find(a.idx);
            if (nit == nconsumers.end() || nit->second != 1) return false;
            op = consumer[a.idx];
            return prog[op] && !dropped[op] && prog[op]->outputs.size() == 1;
        };

        for (size_t i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& pow_layer = prog[i];
            if (!pow_layer || dropped[i]) continue;
            if (!nary(pow_layer, NaryEltwiseLayer::OPERATION::POW)) continue;
            if (pow_layer->inputs.size() != 2 || pow_layer->outputs.size() != 1) continue;

            Arg X = pow_layer->inputs[0];
            float exponent = 0.f;
            if (!constScalar(pow_layer->inputs[1], exponent) || exponent != 2.f) continue;

            int rm_idx = -1;
            if (!step(pow_layer->outputs[0], rm_idx)) continue;
            Reduce2Layer* rm = dynamic_cast<Reduce2Layer*>(prog[rm_idx].get());
            if (!rm || rm->reduce_type != Reduce2Layer::ReduceType::MEAN) continue;
            // Shapes are not inferred before fusion, so only -1 is known to be the last axis.
            if (!rm->keepdims || rm->axes.size() != 1 || rm->axes[0] != -1) continue;

            int add_idx = -1;
            if (!step(prog[rm_idx]->outputs[0], add_idx)) continue;
            if (!nary(prog[add_idx], NaryEltwiseLayer::OPERATION::ADD)) continue;
            Arg eps_arg = otherInput(prog[add_idx], prog[rm_idx]->outputs[0]);
            float epsilon = 0.f;
            if (eps_arg.empty() || !constScalar(eps_arg, epsilon)) continue;

            int sqrt_idx = -1;
            if (!step(prog[add_idx]->outputs[0], sqrt_idx)) continue;
            if (!dynamic_cast<SqrtLayer*>(prog[sqrt_idx].get())) continue;

            int div_idx = -1;
            if (!step(prog[sqrt_idx]->outputs[0], div_idx)) continue;
            if (!nary(prog[div_idx], NaryEltwiseLayer::OPERATION::DIV)) continue;
            // Only Div(1, sqrt) is a reciprocal; Div(sqrt, 1) would invert the scaling.
            const vector<Arg>& div_in = prog[div_idx]->inputs;
            if (div_in.size() != 2 || div_in[1].idx != prog[sqrt_idx]->outputs[0].idx) continue;
            float numerator = 0.f;
            if (!constScalar(div_in[0], numerator) || numerator != 1.f) continue;

            int mul_idx = -1;
            if (!step(prog[div_idx]->outputs[0], mul_idx)) continue;
            if (!nary(prog[mul_idx], NaryEltwiseLayer::OPERATION::PROD)) continue;
            if (otherInput(prog[mul_idx], prog[div_idx]->outputs[0]).idx != X.idx) continue;

            int scale_idx = -1;
            if (!step(prog[mul_idx]->outputs[0], scale_idx)) continue;
            if (!nary(prog[scale_idx], NaryEltwiseLayer::OPERATION::PROD)) continue;
            Arg scale = otherInput(prog[scale_idx], prog[mul_idx]->outputs[0]);
            if (scale.empty() || !netimpl->isConstArg(scale)) continue;
            // RMSNormalization needs a rank-1 scale; Mul would have broadcast anything.
            const MatShape sshape = netimpl->argTensor(scale).shape();
            if (sshape.dims != 1 || sshape[0] < 2) continue;

            LayerParams lp;
            lp.name = prog[scale_idx]->name;   // keep the name for profiling continuity
            lp.type = "RMSNormalization";
            lp.set("axis", -1);
            lp.set("epsilon", epsilon);

            Ptr<LayerInfo> rmsnorm = LayerFactory::createLayerInstance("RMSNormalization", lp);
            if (!rmsnorm) continue;
            rmsnorm->inputs  = { X, scale };
            rmsnorm->outputs = prog[scale_idx]->outputs;
            rmsnorm->netimpl = netimpl;

            newprog[i] = rmsnorm;
            for (int op : { rm_idx, add_idx, sqrt_idx, div_idx, mul_idx, scale_idx })
                dropped[op] = true;
            modified = true;
        }

        if (modified) {
            vector<Ptr<LayerInfo>> fused;
            fused.reserve(nops);
            for (size_t i = 0; i < nops; i++) {
                if (!dropped[i] && newprog[i])
                    fused.push_back(newprog[i]);
            }
            graph->setProg(fused);
        }
        return modified;
    }

    Net::Impl* netimpl;
};

void Net::Impl::fuseRMSNorm()
{
    // RMSNormalization has no CUDA exec, and one such op sends the whole graph to the CPU.
    if (preferableBackend != DNN_BACKEND_OPENCV || preferableTarget != DNN_TARGET_CPU)
        return;
    ModelFusionRMSNorm pass(this);
    pass.fuse();
}

CV__DNN_INLINE_NS_END
}}
