// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Folds a scalar Mul or Div on a layer's input into that layer's own parameters.
//
//   X -> Mul(X, c) -> L   =>   X -> L, with c folded in
//   X -> Div(X, c) -> L   =>   X -> L, with 1/c folded in
//
// Where it lands is the layer's choice: alpha for MatMul and Gemm, scale for Softmax.
// Declining is always safe, so the offer costs nothing.

#include "precomp.hpp"
#include "net_impl.hpp"
#include "adjacency_graph.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;

struct ModelFusionInputScalar
{
    explicit ModelFusionInputScalar(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse() { fuseGraph(netimpl->mainGraph); }

    bool scalarOf(Arg a, float& out) const
    {
        if (!netimpl->isConstArg(a)) return false;
        Mat t = netimpl->argTensor(a);
        if (t.total() != 1) return false;
        if (t.type() == CV_32F) { out = t.ptr<float>()[0]; return true; }
        if (t.type() == CV_64F) { out = (float)t.ptr<double>()[0]; return true; }
        if (t.type() == CV_16F) {
            Mat f; t.convertTo(f, CV_32F); out = f.ptr<float>()[0]; return true;
        }
        return false;
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

        std::map<int, int> producer;
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            for (Arg out : prog[i]->outputs)
                producer[out.idx] = (int)i;
        }

        vector<bool> dropped(nops, false);

        for (size_t i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& layer = prog[i];
            if (!layer || dropped[i]) continue;
            Layer* sink = dynamic_cast<Layer*>(layer.get());
            if (!sink) continue;
            const FusionOps* ops = fusionOpsFor(sink);
            if (!ops || !ops->foldInputScalar) continue;

            for (size_t slot = 0; slot < layer->inputs.size(); slot++) {
                Arg mid = layer->inputs[slot];
                auto it = producer.find(mid.idx);
                if (it == producer.end()) continue;
                int pidx = it->second;
                if (pidx < 0 || dropped[pidx]) continue;

                const Ptr<LayerInfo>& pl = prog[pidx];
                NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(pl.get());
                if (!elt) continue;
                const bool is_mul = elt->op == NaryEltwiseLayer::OPERATION::PROD;
                const bool is_div = elt->op == NaryEltwiseLayer::OPERATION::DIV;
                if (!is_mul && !is_div) continue;
                if (pl->inputs.size() != 2 || pl->outputs.size() != 1) continue;

                float scale = 1.f;
                int data_slot = -1;
                if (scalarOf(pl->inputs[1], scale)) data_slot = 0;
                // Mul commutes, Div does not
                else if (is_mul && scalarOf(pl->inputs[0], scale)) data_slot = 1;
                else continue;

                if (is_div) {
                    if (scale == 0.f) continue;
                    scale = 1.f / scale;
                }
                if (!std::isfinite(scale)) continue;

                // The fold deletes this arg, so nothing else may still read it.
                if (usecounts[mid.idx] != 1 || externalArgs.count(mid.idx)) continue;

                // Offered last: a layer that accepts has already changed itself.
                if (!ops->foldInputScalar(sink, (int)slot, scale)) continue;

                layer->inputs[slot] = pl->inputs[data_slot];
                dropped[pidx] = true;
                usecounts[mid.idx] = 0;
                modified = true;
            }
        }

        if (modified) {
            vector<Ptr<LayerInfo>> newprog;
            newprog.reserve(nops);
            for (size_t i = 0; i < nops; i++)
                if (!dropped[i] && prog[i])
                    newprog.push_back(prog[i]);
            graph->setProg(newprog);
        }
        return modified;
    }

    Net::Impl* netimpl;
};

void Net::Impl::fuseInputScalars()
{
    // Only the CPU kernels read the folded factor.
    if (preferableBackend != DNN_BACKEND_OPENCV || preferableTarget != DNN_TARGET_CPU)
        return;
    ModelFusionInputScalar pass(this);
    pass.fuse();
}

CV__DNN_INLINE_NS_END
}}
