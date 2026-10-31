// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Folds the ops around a dynamically quantized int8 MatMul into it: the dequantizing
// Cast/Mul/Add, the DynamicQuantizeLinear that feeds it and the attention head split after it.

#include "precomp.hpp"
#include "net_impl.hpp"
#include "layers/layers_common.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;

struct ModelFusionMatMulIntegerDequant
{
    explicit ModelFusionMatMulIntegerDequant(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse() { fuseGraph(netimpl->mainGraph); }

    // a float constant of at most one dimension, with one value or one value per column
    bool isColumnConst(Arg a, int N, bool allowScalar) const
    {
        if (a.idx <= 0 || !netimpl->isConstArg(a))
            return false;
        const Mat& t = netimpl->argTensor(a);
        return t.type() == CV_32F && t.dims <= 1 &&
               (t.total() == (size_t)N || (allowScalar && t.total() == 1));
    }

    bool isScale(Arg a, int N, const std::map<int, int>& producer,
                 const vector<Ptr<LayerInfo>>& prog) const
    {
        if (isColumnConst(a, N, true))
            return true;
        auto it = producer.find(a.idx);
        if (it == producer.end())
            return false;
        const Ptr<LayerInfo>& p = prog[it->second];
        // y_scale of DynamicQuantizeLinear is a float scalar by the spec
        if (dynamic_cast<DynamicQuantizeLinearLayer*>(p.get()))
            return p->outputs.size() > 1 && p->outputs[1].idx == a.idx;
        NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(p.get());
        if (!elt || elt->op != NaryEltwiseLayer::OPERATION::PROD || p->inputs.size() != 2)
            return false;
        // a product of two column vectors would be a matrix, so one side must be a scalar
        bool scalar0 = isScalarScale(p->inputs[0], producer, prog);
        bool scalar1 = isScalarScale(p->inputs[1], producer, prog);
        return (scalar0 && (scalar1 || isColumnConst(p->inputs[1], N, true))) ||
               (scalar1 && isColumnConst(p->inputs[0], N, true));
    }

    bool isScalarScale(Arg a, const std::map<int, int>& producer,
                       const vector<Ptr<LayerInfo>>& prog) const
    {
        if (isColumnConst(a, 1, true))
            return true;
        auto it = producer.find(a.idx);
        if (it == producer.end())
            return false;
        const Ptr<LayerInfo>& p = prog[it->second];
        return dynamic_cast<DynamicQuantizeLinearLayer*>(p.get()) &&
               p->outputs.size() > 1 && p->outputs[1].idx == a.idx;
    }

    // absent, a one-value constant or a DynamicQuantizeLinear zero point, which is a scalar too
    bool isScalarZeroPoint(Arg a, const std::map<int, int>& producer,
                           const vector<Ptr<LayerInfo>>& prog) const
    {
        if (a.idx <= 0)
            return true;
        if (netimpl->isConstArg(a))
            return netimpl->argTensor(a).total() == 1;
        auto it = producer.find(a.idx);
        if (it == producer.end())
            return false;
        const Ptr<LayerInfo>& p = prog[it->second];
        return dynamic_cast<DynamicQuantizeLinearLayer*>(p.get()) &&
               p->outputs.size() > 2 && p->outputs[2].idx == a.idx;
    }

    // the only consumer of `a`, or -1
    static int soleConsumer(Arg a, const vector<int>& usecounts,
                            const std::map<int, vector<int>>& consumers)
    {
        auto it = consumers.find(a.idx);
        if (usecounts[a.idx] != 1 || it == consumers.end() || it->second.size() != 1)
            return -1;
        return it->second[0];
    }

    // the op that produces each arg and the ops that read it, as indices into prog
    static void collectUses(const vector<Ptr<LayerInfo>>& prog, std::map<int, int>& producer,
                            std::map<int, vector<int>>& consumers)
    {
        for (size_t i = 0; i < prog.size(); i++) {
            if (!prog[i]) continue;
            for (Arg out : prog[i]->outputs)
                producer[out.idx] = (int)i;
            for (Arg inp : prog[i]->inputs)
                if (inp.idx > 0)
                    consumers[inp.idx].push_back((int)i);
        }
    }

    // drops the absorbed ops; where emitAt[i] >= 0, the fused op prog[emitAt[i]] goes at position i
    static void rebuildProg(Ptr<Graph>& graph, const vector<bool>& dropped,
                            const vector<int>& emitAt = vector<int>())
    {
        const vector<Ptr<LayerInfo>>& prog = graph->prog();
        vector<Ptr<LayerInfo>> newprog;
        newprog.reserve(prog.size());
        for (size_t i = 0; i < prog.size(); i++) {
            if (!emitAt.empty() && emitAt[i] >= 0)
                newprog.push_back(prog[emitAt[i]]);
            else if (!dropped[i] && prog[i])
                newprog.push_back(prog[i]);
        }
        graph->setProg(newprog);
    }

    // MatMulInteger -> Cast(float) -> Mul(scale) [-> Add(bias)]  =>  MatMulInt8(dequant_output)
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

        // Graph outputs count as uses, so one of them never has a use count of 1 here.
        vector<int> usecounts;
        netimpl->useCounts(usecounts);

        std::map<int, int> producer;
        std::map<int, vector<int>> consumers;
        collectUses(prog, producer, consumers);

        vector<bool> dropped(nops, false);
        vector<int> emitAt(nops, -1);

        for (size_t i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& layer = prog[i];
            MatMulInt8Layer* mm = dynamic_cast<MatMulInt8Layer*>(layer.get());
            if (!mm || !mm->int32_output || mm->dequant_output || dropped[i])
                continue;
            if (layer->inputs.size() < 2 || layer->inputs.size() > 4 || layer->outputs.size() != 1)
                continue;
            Arg B = layer->inputs[1];
            if (!netimpl->isConstArg(B))
                continue;
            // the kernel applies a per-row zero point after its int32 store, which the scale skips
            if (layer->inputs.size() > 2 && !isScalarZeroPoint(layer->inputs[2], producer, prog))
                continue;
            const Mat& Bt = netimpl->argTensor(B);
            if (Bt.dims != 2 || Bt.size[0] == 0)
                continue;
            const int N = Bt.size[1];

            int ci = soleConsumer(layer->outputs[0], usecounts, consumers);
            if (ci < 0 || dropped[ci])
                continue;
            const Ptr<LayerInfo>& cast = prog[ci];
            if (!dynamic_cast<Cast2Layer*>(cast.get()) || cast->inputs.size() != 1 ||
                cast->outputs.size() != 1)
                continue;
            vector<MatType> castOut, castInternals;
            cast->getTypes({CV_32S}, 1, 0, castOut, castInternals);
            if (castOut.size() != 1 || castOut[0] != CV_32F)
                continue;

            int mi = soleConsumer(cast->outputs[0], usecounts, consumers);
            if (mi < 0 || dropped[mi])
                continue;
            const Ptr<LayerInfo>& mul = prog[mi];
            NaryEltwiseLayer* mulElt = dynamic_cast<NaryEltwiseLayer*>(mul.get());
            if (!mulElt || mulElt->op != NaryEltwiseLayer::OPERATION::PROD ||
                mul->inputs.size() != 2 || mul->outputs.size() != 1)
                continue;
            Arg scale = mul->inputs[mul->inputs[0].idx == cast->outputs[0].idx ? 1 : 0];
            if (!isScale(scale, N, producer, prog))
                continue;

            // the last absorbed op, whose position comes after everything the scale needs
            int last = mi;
            Arg out = mul->outputs[0], bias;
            int ai = soleConsumer(out, usecounts, consumers);
            if (ai >= 0 && !dropped[ai]) {
                const Ptr<LayerInfo>& add = prog[ai];
                NaryEltwiseLayer* addElt = dynamic_cast<NaryEltwiseLayer*>(add.get());
                if (addElt && (addElt->op == NaryEltwiseLayer::OPERATION::ADD ||
                               addElt->op == NaryEltwiseLayer::OPERATION::SUM) &&
                    add->inputs.size() == 2 && add->outputs.size() == 1) {
                    Arg other = add->inputs[add->inputs[0].idx == out.idx ? 1 : 0];
                    if (isColumnConst(other, N, false)) {
                        bias = other;
                        out = add->outputs[0];
                        last = ai;
                    }
                }
            }

            vector<Arg> inputs = layer->inputs;
            inputs.resize(4);   // absent zero points stay the empty Arg()
            inputs.push_back(scale);
            if (bias.idx > 0)
                inputs.push_back(bias);
            layer->inputs = inputs;
            layer->outputs[0] = out;
            mm->dequant_output = true;

            dropped[i] = dropped[ci] = dropped[mi] = true;
            dropped[last] = true;
            emitAt[last] = (int)i;
            modified = true;
        }

        if (modified)
            rebuildProg(graph, dropped, emitAt);
        return modified;
    }

    // DynamicQuantizeLinear -> MatMulInt8(dequant_output)  =>  MatMulInt8(quantize_input), when
    // the matmul is the only user of the DQL outputs; a product with a weight scale folds in too
    bool fuseQuantizeInput(Ptr<Graph>& graph)
    {
        const vector<Ptr<LayerInfo>>& prog = graph->prog();
        size_t nops = prog.size();
        bool modified = false;
        for (size_t i = 0; i < nops; i++) {
            vector<Ptr<Graph>>* subgraphs = prog[i] ? prog[i]->subgraphs() : nullptr;
            if (subgraphs)
                for (Ptr<Graph>& g : *subgraphs)
                    if (fuseQuantizeInput(g)) modified = true;
        }

        vector<int> usecounts;
        netimpl->useCounts(usecounts);
        std::map<int, int> producer;
        std::map<int, vector<int>> consumers;
        collectUses(prog, producer, consumers);

        vector<bool> dropped(nops, false);
        for (size_t i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& layer = prog[i];
            MatMulInt8Layer* mm = dynamic_cast<MatMulInt8Layer*>(layer.get());
            if (!mm || !mm->dequant_output || mm->quantize_input || layer->inputs.size() < 5)
                continue;
            auto pit = producer.find(layer->inputs[0].idx);
            if (pit == producer.end() || dropped[pit->second])
                continue;
            const int di = pit->second;
            const Ptr<LayerInfo>& dql = prog[di];
            if (!dynamic_cast<DynamicQuantizeLinearLayer*>(dql.get()) || dql->outputs.size() != 3 ||
                dql->outputs[0].idx != layer->inputs[0].idx ||
                soleConsumer(dql->outputs[0], usecounts, consumers) != (int)i)
                continue;
            Arg yScale = dql->outputs[1], yZp = dql->outputs[2];
            // folded, the layer subtracts the zero point it computes: the matmul must take that one
            if (yZp.idx <= 0 || layer->inputs[2].idx != yZp.idx ||
                soleConsumer(yZp, usecounts, consumers) != (int)i)
                continue;

            const int N = netimpl->argTensor(layer->inputs[1]).size[1];
            Arg scale = layer->inputs[4], wScale;
            int pi = -1;
            if (scale.idx == yScale.idx) {
                if (soleConsumer(yScale, usecounts, consumers) != (int)i)
                    continue;
            } else {
                auto sit = producer.find(scale.idx);
                if (sit == producer.end() || dropped[sit->second] ||
                    soleConsumer(scale, usecounts, consumers) != (int)i)
                    continue;
                pi = sit->second;
                const Ptr<LayerInfo>& prod = prog[pi];
                NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(prod.get());
                if (!elt || elt->op != NaryEltwiseLayer::OPERATION::PROD ||
                    prod->inputs.size() != 2 || prod->outputs.size() != 1 ||
                    soleConsumer(yScale, usecounts, consumers) != pi)
                    continue;
                wScale = prod->inputs[prod->inputs[0].idx == yScale.idx ? 1 : 0];
                if (!isColumnConst(wScale, N, true))
                    continue;
            }

            layer->inputs[0] = dql->inputs[0];
            layer->inputs[2] = Arg();
            layer->inputs[4] = wScale;
            mm->quantize_input = true;
            dropped[di] = true;
            if (pi >= 0)
                dropped[pi] = true;
            modified = true;
        }

        if (modified)
            rebuildProg(graph, dropped);
        return modified;
    }

    // MatMulInt8(dequant_output) -> Reshape([B,S,H,D]) -> Transpose(0,2,1,3)  =>  head_split
    bool fuseHeadSplit(Ptr<Graph>& graph)
    {
        const vector<Ptr<LayerInfo>>& prog = graph->prog();
        size_t nops = prog.size();
        bool modified = false;
        for (size_t i = 0; i < nops; i++) {
            vector<Ptr<Graph>>* subgraphs = prog[i] ? prog[i]->subgraphs() : nullptr;
            if (subgraphs)
                for (Ptr<Graph>& g : *subgraphs)
                    if (fuseHeadSplit(g)) modified = true;
        }

        vector<int> usecounts;
        netimpl->useCounts(usecounts);
        std::map<int, int> producer;
        std::map<int, vector<int>> consumers;
        collectUses(prog, producer, consumers);

        vector<bool> dropped(nops, false);
        vector<int> emitAt(nops, -1);
        for (size_t i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& layer = prog[i];
            MatMulInt8Layer* mm = dynamic_cast<MatMulInt8Layer*>(layer.get());
            if (!mm || !mm->dequant_output || mm->head_split || dropped[i])
                continue;
            int ri = soleConsumer(layer->outputs[0], usecounts, consumers);
            if (ri < 0 || dropped[ri])
                continue;
            const Ptr<LayerInfo>& rs = prog[ri];
            Reshape2Layer* reshape = dynamic_cast<Reshape2Layer*>(rs.get());
            if (!reshape || rs->outputs.size() != 1 || rs->inputs.empty() ||
                rs->inputs.size() > 2 || rs->inputs[0].idx != layer->outputs[0].idx ||
                (rs->inputs.size() == 1 && reshape->newShapeDesc.dims < 0))
                continue;
            int ti = soleConsumer(rs->outputs[0], usecounts, consumers);
            if (ti < 0 || dropped[ti])
                continue;
            const Ptr<LayerInfo>& tr = prog[ti];
            TransposeLayer* transpose = dynamic_cast<TransposeLayer*>(tr.get());
            if (!transpose || tr->inputs.size() != 1 || tr->outputs.size() != 1 ||
                transpose->perm != vector<int>({0, 2, 1, 3}))
                continue;

            if (rs->inputs.size() == 2 && netimpl->isConstArg(rs->inputs[1])) {
                // a constant shape keeps the output shape known before forward()
                const Mat& t = netimpl->argTensor(rs->inputs[1]);
                if (!t.isContinuous() || (t.type() != CV_32S && t.type() != CV_64S))
                    continue;
                mm->head_split_shape = tensorToShape(t);
            } else if (rs->inputs.size() == 2) {
                vector<Arg> inputs = layer->inputs;
                inputs.resize(6);   // an absent bias stays the empty Arg()
                inputs.push_back(rs->inputs[1]);
                layer->inputs = inputs;
            } else {
                mm->head_split_shape = reshape->newShapeDesc;
            }
            mm->head_split = true;
            layer->outputs[0] = tr->outputs[0];
            // after the Reshape, so a runtime shape is computed before the fused op reads it
            dropped[i] = dropped[ri] = dropped[ti] = true;
            emitAt[ti] = (int)i;
            modified = true;
        }

        if (modified)
            rebuildProg(graph, dropped, emitAt);
        return modified;
    }

    Net::Impl* netimpl;
};

void Net::Impl::fuseMatMulIntegerDequant()
{
    if (preferableBackend != DNN_BACKEND_OPENCV || preferableTarget != DNN_TARGET_CPU)
        return;
    ModelFusionMatMulIntegerDequant pass(this);
    pass.fuse();
    pass.fuseQuantizeInput(mainGraph);
    pass.fuseHeadSplit(mainGraph);
}

CV__DNN_INLINE_NS_END
}}
