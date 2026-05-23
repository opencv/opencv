// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Cleans up redundant Reshape/Transpose chains:
//   - Reshape(A) -> Reshape(B): drop the inner reshape (A's shape is irrelevant
//     once B reshapes to the final target).
//   - Transpose(p1) -> Transpose(p2): compose into a single Transpose with
//     perm[i] = p1[p2[i]].
//   - Identity Transpose (perm == [0,1,...,n-1]): bypass.
// These run iteratively, so a Reshape+Transpose+Reshape that ends up with an
// identity transpose collapses to a single Reshape, and adjacent transposes
// in attention layouts merge.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

struct ModelFusionReshapeTranspose
{
    explicit ModelFusionReshapeTranspose(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse()
    {
        for (int iter = 0; iter < 10; iter++) {
            netimpl->useCounts(usecounts);
            if (!fuseGraph(netimpl->mainGraph))
                break;
        }
    }

    static bool isIdentityPerm(const vector<int>& perm)
    {
        if (perm.empty()) return false;
        for (size_t i = 0; i < perm.size(); i++)
            if (perm[i] != (int)i) return false;
        return true;
    }

    bool fuseGraph(Ptr<Graph>& graph)
    {
        const vector<Ptr<Layer>>& prog = graph->prog();
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

        std::map<int, int> producer;
        std::set<int> externalArgs;
        for (Arg out : graph->outputs())
            externalArgs.insert(out.idx);
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            for (Arg out : prog[i]->outputs)
                producer[out.idx] = (int)i;
        }

        vector<bool> dropped(nops, false);

        for (size_t i = 0; i < nops; i++) {
            const Ptr<Layer>& layer = prog[i];
            if (!layer || dropped[i]) continue;
            if (layer->inputs.empty() || layer->outputs.empty()) continue;

            // Identity Transpose: bypass.
            TransposeLayer* tr = dynamic_cast<TransposeLayer*>(layer.get());
            if (tr && layer->outputs.size() == 1 && isIdentityPerm(tr->perm)) {
                Arg out = layer->outputs[0];
                if (externalArgs.count(out.idx) == 0) {
                    redirectConsumers(prog, dropped, i + 1, out, layer->inputs[0]);
                    dropped[i] = true;
                    usecounts[out.idx] = 0;
                    modified = true;
                    continue;
                }
            }

            // Transpose + Transpose -> Transpose (compose perms).
            if (tr && layer->outputs.size() == 1) {
                auto it = producer.find(layer->inputs[0].idx);
                if (it != producer.end()) {
                    int prod_idx = it->second;
                    if (prod_idx >= 0 && !dropped[prod_idx]) {
                        const Ptr<Layer>& pl = prog[prod_idx];
                        TransposeLayer* prevTr = dynamic_cast<TransposeLayer*>(pl.get());
                        Arg prevOut = layer->inputs[0];
                        bool single_consumer = usecounts[prevOut.idx] == 1
                                            && externalArgs.count(prevOut.idx) == 0;
                        if (prevTr && pl->outputs.size() == 1 && single_consumer
                            && !prevTr->perm.empty()
                            && prevTr->perm.size() == tr->perm.size())
                        {
                            // composed[i] = prevTr->perm[tr->perm[i]]
                            vector<int> composed(tr->perm.size());
                            bool ok = true;
                            for (size_t k = 0; k < tr->perm.size(); k++) {
                                int idx = tr->perm[k];
                                if (idx < 0 || idx >= (int)prevTr->perm.size()) {
                                    ok = false; break;
                                }
                                composed[k] = prevTr->perm[idx];
                            }
                            if (ok) {
                                tr->perm = composed;
                                layer->inputs[0] = pl->inputs[0];
                                dropped[prod_idx] = true;
                                usecounts[prevOut.idx] = 0;
                                producer[layer->outputs[0].idx] = (int)i;
                                modified = true;
                                continue;
                            }
                        }
                    }
                }
            }

            //Reshape + Reshape -> Reshape (drop the inner reshape).
            Reshape2Layer* rs = dynamic_cast<Reshape2Layer*>(layer.get());
            if (rs && layer->outputs.size() == 1) {
                auto it = producer.find(layer->inputs[0].idx);
                if (it != producer.end()) {
                    int prod_idx = it->second;
                    if (prod_idx >= 0 && !dropped[prod_idx]) {
                        const Ptr<Layer>& pl = prog[prod_idx];
                        Reshape2Layer* prevRs = dynamic_cast<Reshape2Layer*>(pl.get());
                        Arg prevOut = layer->inputs[0];
                        bool single_consumer = usecounts[prevOut.idx] == 1
                                            && externalArgs.count(prevOut.idx) == 0;
                        if (prevRs && pl->outputs.size() == 1 && single_consumer)
                        {
                            layer->inputs[0] = pl->inputs[0];
                            dropped[prod_idx] = true;
                            usecounts[prevOut.idx] = 0;
                            modified = true;
                            continue;
                        }
                    }
                }
            }
        }

        if (modified) {
            vector<Ptr<Layer>> newprog;
            newprog.reserve(nops);
            for (size_t i = 0; i < nops; i++) {
                if (!dropped[i] && prog[i])
                    newprog.push_back(prog[i]);
            }
            graph->setProg(newprog);
        }

        return modified;
    }

    void redirectConsumers(const vector<Ptr<Layer>>& prog,
                           const vector<bool>& dropped,
                           size_t start_idx, Arg from, Arg to)
    {
        for (size_t j = start_idx; j < prog.size(); j++) {
            if (!prog[j] || dropped[j]) continue;
            for (Arg& in : prog[j]->inputs) {
                if (in.idx == from.idx) {
                    in = to;
                    usecounts[to.idx]++;
                }
            }
        }
    }

    Net::Impl* netimpl;
    vector<int> usecounts;
};

void Net::Impl::fuseReshapeTranspose()
{
    ModelFusionReshapeTranspose pass(this);
    pass.fuse();
}

CV__DNN_INLINE_NS_END
}}
