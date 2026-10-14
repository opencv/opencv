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
//   - Reshape whose shape operand is Concat(Unsqueeze(Gather(Shape(s), i)), consts...):
//     replace it with a constant, writing 0 where the gathered dimension provably equals
//     the Reshape input's own at that position, which Reshape reads as "keep this axis".
// These run iteratively, so a Reshape+Transpose+Reshape that ends up with an
// identity transpose collapses to a single Reshape, and adjacent transposes
// in attention layouts merge.

#include "precomp.hpp"
#include "net_impl.hpp"
#include "layers/layers_common.hpp"

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

    bool constInts(Arg a, std::vector<int>& out) const
    {
        if (!netimpl->isConstArg(a)) return false;
        Mat t = netimpl->argTensor(a);
        if (t.empty() || t.total() > 16) return false;
        tensorToIntVec(t, out);
        return !out.empty();
    }

    // The declared length of `arg`'s axis `ax`, or 0 when it is not known.
    int declaredDim(Arg arg, int ax) const
    {
        auto it = netimpl->declaredShapes.find(arg.idx);
        if (it == netimpl->declaredShapes.end()) return 0;
        const MatShape& sh = it->second;
        if (ax < 0 || ax >= sh.dims) return 0;
        return sh[ax];
    }

    bool foldShapeOperand(const vector<Ptr<LayerInfo>>& prog, const std::map<int,int>& producer,
                          vector<bool>& dropped, const std::set<int>& externalArgs,
                          const Ptr<LayerInfo>& layer)
    {
        if (layer->inputs.size() != 2) return false;
        const Arg root = layer->inputs[0], shapeArg = layer->inputs[1];
        if (netimpl->isConstArg(shapeArg)) return false;
        if (usecounts[shapeArg.idx] != 1 || externalArgs.count(shapeArg.idx)) return false;

        auto pit = producer.find(shapeArg.idx);
        if (pit == producer.end() || dropped[pit->second]) return false;
        const Ptr<LayerInfo>& cl = prog[pit->second];
        Concat2Layer* cat = dynamic_cast<Concat2Layer*>(cl.get());
        if (!cat || cat->axis != 0 || cl->outputs.size() != 1) return false;

        std::vector<int> shape, chain;
        for (size_t k = 0; k < cl->inputs.size(); k++) {
            std::vector<int> vals;
            if (constInts(cl->inputs[k], vals)) {
                shape.insert(shape.end(), vals.begin(), vals.end());
                continue;
            }
            auto uit = producer.find(cl->inputs[k].idx);
            if (uit == producer.end()) return false;
            const Ptr<LayerInfo>& ul = prog[uit->second];
            if (!dynamic_cast<UnsqueezeLayer*>(ul.get()) || ul->inputs.empty()) return false;
            chain.push_back(uit->second);

            auto git = producer.find(ul->inputs[0].idx);
            if (git == producer.end()) return false;
            const Ptr<LayerInfo>& gl = prog[git->second];
            Gather2Layer* gath = dynamic_cast<Gather2Layer*>(gl.get());
            if (!gath || gath->axis != 0 || gl->inputs.size() != 2) return false;
            chain.push_back(git->second);

            std::vector<int> idx;
            if (!constInts(gl->inputs[1], idx) || idx.size() != 1) return false;

            auto sit = producer.find(gl->inputs[0].idx);
            if (sit == producer.end()) return false;
            const Ptr<LayerInfo>& sl = prog[sit->second];
            ShapeLayer* shp = dynamic_cast<ShapeLayer*>(sl.get());
            if (!shp || shp->start != 0 || shp->end != INT_MAX || sl->inputs.empty())
                return false;
            chain.push_back(sit->second);

            const int pos = (int)shape.size();
            const int gathered = declaredDim(sl->inputs[0], idx[0]);
            // Only a shared symbolic dim stays equal at runtime; matching statics need not.
            if (gathered >= 0 || gathered != declaredDim(root, pos)) return false;
            shape.push_back(0);
        }
        if (shape.empty() || std::count(shape.begin(), shape.end(), -1) > 1) return false;

        Mat sm(1, (int)shape.size(), CV_32S, shape.data());
        layer->inputs[1] = netimpl->newConstArg(netimpl->argData(shapeArg).name + "/const",
                                                sm.clone());
        dropped[pit->second] = true;
        usecounts[shapeArg.idx] = 0;

        // Each triple is Unsqueeze, Gather, Shape. A producer is released only once its own
        // reader here went, so a Shape several Gathers share outlives all but the last.
        for (size_t k = 0; k + 2 < chain.size(); k += 3) {
            for (size_t t = k; t < k + 3; t++) {
                const int li = chain[t];
                if (dropped[li] || prog[li]->outputs.size() != 1) break;
                const int oi = prog[li]->outputs[0].idx;
                if (usecounts[oi] == 0 || externalArgs.count(oi)) break;
                if (--usecounts[oi] != 0) break;
                dropped[li] = true;
            }
        }
        return true;
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
            const Ptr<LayerInfo>& layer = prog[i];
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
                        const Ptr<LayerInfo>& pl = prog[prod_idx];
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

            Reshape2Layer* rs = dynamic_cast<Reshape2Layer*>(layer.get());
            if (rs && layer->outputs.size() == 1) {
                // Fold first: the merge below reads the resolved spec to decide if it can run.
                if (foldShapeOperand(prog, producer, dropped, externalArgs, layer))
                    modified = true;

                //Reshape + Reshape -> Reshape (drop the inner reshape).
                auto it = producer.find(layer->inputs[0].idx);
                if (it != producer.end()) {
                    int prod_idx = it->second;
                    if (prod_idx >= 0 && !dropped[prod_idx]) {
                        const Ptr<LayerInfo>& pl = prog[prod_idx];
                        Reshape2Layer* prevRs = dynamic_cast<Reshape2Layer*>(pl.get());
                        Arg prevOut = layer->inputs[0];
                        bool single_consumer = usecounts[prevOut.idx] == 1
                                            && externalArgs.count(prevOut.idx) == 0;
                        // 0 in the spec means inputs[0]'s own dim, which the rewrite
                        // below changes. An unknown spec leaves the merge as it was.
                        std::vector<int> spec;
                        if (rs->newShapeDesc.dims >= 0)
                            for (int d = 0; d < rs->newShapeDesc.dims; d++)
                                spec.push_back(rs->newShapeDesc[d]);
                        else if (layer->inputs.size() == 2)
                            constInts(layer->inputs[1], spec);
                        bool keeps_dims = std::find(spec.begin(), spec.end(), 0) == spec.end();
                        if (prevRs && pl->outputs.size() == 1 && single_consumer && keeps_dims)
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
            vector<Ptr<LayerInfo>> newprog;
            newprog.reserve(nops);
            for (size_t i = 0; i < nops; i++) {
                if (!dropped[i] && prog[i])
                    newprog.push_back(prog[i]);
            }
            graph->setProg(newprog);
        }

        return modified;
    }

    void redirectConsumers(const vector<Ptr<LayerInfo>>& prog,
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
