// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Rewrites a MatMul whose B (and optionally bias) are constant blobs into a
// Gemm with `flatten_a=false`. The Gemm path pre-packs B in MLAS layout at
// finalize() and dispatches each forward through MlasGemmPacked, which is
// notably faster than the OpenCV-internal packed AVX2 kernel that the MatMul
// constant-B path otherwise hits.
//
// Pattern (post-importer; bias has already been absorbed into MatMul.blobs[1]
// by BiasedMatmulSubgraph when it was a const Add):
//   A(any rank) -> MatMul[const B (2D), optional const bias]
//
// Replacement:
//   A -> Gemm[transA=false, transB=mm.trans_b, alpha=mm.alpha, beta=mm.beta,
//             constB=true, const_C=have_bias, flatten_a=false,
//             blobs={B, bias?}]
//
// Restrictions (skip otherwise):
//   - MatMul.trans_a must be false (ND-A flattening assumes row-major K is the
//     last axis)
//   - blobs.size() in {1, 2} and blobs[0] (B) is 2D float32
//   - if a bias is present, it must have 1 element, or N elements with last
//     dimension N, and no more dimensions than A (requiring A's rank to be
//     known when bias.dims > 1); higher-rank biases are reshaped to 1D before
//     building Gemm. The Gemm flatten_a=false bias path tiles a per-row
//     pattern, so per-row biases like [M, 1] or [M, N] aren't supported.
//
// A per-channel constant Mul/Add right after the MatMul (e.g. ViT LayerScale) is folded
// into B and the bias, so it costs nothing at run time. Only done when alpha == beta == 1.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

struct ModelFusionMatMulToGemm
{
    explicit ModelFusionMatMulToGemm(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse() { fuseGraph(netimpl->mainGraph); }

    // Returns a constant scalar or 1D length-N tensor as a 1xN float row.
    // Higher-rank constants are skipped, since they could change the output rank.
    bool extractChannelConst(Arg a, int N, Mat& vec) const
    {
        if (!netimpl->isConstArg(a)) return false;
        const Mat& t = netimpl->argTensor(a);
        if (t.dims > 1 || !t.isContinuous()) return false;
        if (t.type() != CV_32F && t.type() != CV_64F) return false;
        const int total = (int)t.total();
        if (total != 1 && total != N) return false;
        Mat v;
        Mat(1, total, t.type(), t.data).convertTo(v, CV_32F);
        if (total == 1)
            v = Mat(1, N, CV_32F, Scalar(v.at<float>(0)));
        if (!checkRange(v)) return false;
        vec = v;
        return true;
    }

    // Folds the per-channel Mul/Add chain after the MatMul into B and bias.
    // Returns the outputs the Gemm takes over.
    vector<Arg> foldChannelAffine(const vector<Ptr<LayerInfo>>& prog, const Ptr<LayerInfo>& layer,
                                  const std::map<int, int>& consumer, vector<int>& usecounts,
                                  vector<bool>& dropped, bool trans_b, int N,
                                  Mat& B, Mat& bias) const
    {
        vector<Arg> outputs = layer->outputs;
        bool cloned = false;
        while (outputs.size() == 1) {
            const int out = outputs[0].idx;
            // A graph output counts as a use, so this also skips MatMul results that are outputs.
            if (usecounts[out] != 1) break;
            auto cit = consumer.find(out);
            if (cit == consumer.end() || dropped[cit->second]) break;
            const Ptr<LayerInfo>& c = prog[cit->second];
            NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(c.get());
            if (!elt || c->inputs.size() != 2 || c->outputs.size() != 1) break;
            const bool is_mul = elt->op == NaryEltwiseLayer::OPERATION::PROD;
            const bool is_add = elt->op == NaryEltwiseLayer::OPERATION::ADD ||
                                elt->op == NaryEltwiseLayer::OPERATION::SUM;
            if (!is_mul && !is_add) break;
            const int cslot = c->inputs[0].idx == out ? 1 : 0;
            if (c->inputs[1 - cslot].idx != out) break;
            Mat vec;
            if (!extractChannelConst(c->inputs[cslot], N, vec)) break;

            if (!cloned) {
                // Work on copies: these weights may be shared with other layers.
                B = B.clone();
                if (!bias.empty()) {
                    Mat b;
                    if (bias.total() == 1)
                        b = Mat(1, N, CV_32F, Scalar(bias.at<float>(0)));
                    else
                        b = Mat(1, N, CV_32F, bias.data).clone();
                    bias = b;
                }
                cloned = true;
            }

            const float* g = vec.ptr<float>();
            if (is_mul) {
                if (trans_b) {
                    for (int n = 0; n < N; n++) {
                        Mat Brow = B.row(n);
                        Brow *= g[n];
                    }
                } else {
                    for (int k = 0; k < B.rows; k++) {
                        float* row = B.ptr<float>(k);
                        for (int n = 0; n < N; n++)
                            row[n] *= g[n];
                    }
                }
                if (!bias.empty())
                    bias = bias.mul(vec);
            } else {
                bias = bias.empty() ? vec.clone() : Mat(bias + vec);
            }

            outputs = c->outputs;
            dropped[cit->second] = true;
            usecounts[out] = 0;
        }
        if (cloned && !bias.empty())
            bias = bias.reshape(1, std::vector<int>{N});
        return outputs;
    }

    static bool keepsFirstInputRank(const LayerInfo* l)
    {
        return dynamic_cast<const ActivationLayer*>(l) ||
               dynamic_cast<const ClipLayer*>(l) ||
               dynamic_cast<const LayerNormLayer*>(l) ||
               dynamic_cast<const LayerNorm2Layer*>(l) ||
               dynamic_cast<const RMSNormLayer*>(l) ||
               dynamic_cast<const InstanceNormLayer*>(l) ||
               dynamic_cast<const GroupNormLayer*>(l) ||
               dynamic_cast<const BatchNorm2Layer*>(l) ||
               dynamic_cast<const MVNLayer*>(l) ||
               dynamic_cast<const LRNLayer*>(l) ||
               dynamic_cast<const SoftmaxLayer*>(l) ||
               dynamic_cast<const HardmaxLayer*>(l) ||
               dynamic_cast<const Cast2Layer*>(l) ||
               dynamic_cast<const TransposeLayer*>(l) ||
               dynamic_cast<const Concat2Layer*>(l) ||
               dynamic_cast<const Pad2Layer*>(l) ||
               dynamic_cast<const Slice2Layer*>(l) ||
               dynamic_cast<const Tile2Layer*>(l) ||
               dynamic_cast<const Conv2Layer*>(l) ||
               dynamic_cast<const ConvTranspose2Layer*>(l) ||
               dynamic_cast<const MaxPoolLayer*>(l) ||
               dynamic_cast<const AveragePoolLayer*>(l) ||
               dynamic_cast<const LpPoolLayer*>(l) ||
               dynamic_cast<const TriluLayer*>(l) ||
               dynamic_cast<const CumSumLayer*>(l) ||
               dynamic_cast<const CumProdLayer*>(l);
    }

    static int rankOf(const vector<int>& min_rank, Arg a)
    {
        return (a.idx > 0 && (size_t)a.idx < min_rank.size()) ? min_rank[a.idx] : -1;
    }

    static void setMinRank(vector<int>& min_rank, Arg a, int r)
    {
        if (a.idx > 0 && (size_t)a.idx < min_rank.size())
            min_rank[a.idx] = r;
    }

    // Sound lower bound on output 0's rank (-1 when unknown). Intermediates have no shape at
    // fusion time, so declared input/constant ranks are propagated across rank-predictable ops.
    int inferOutputMinRank(const LayerInfo* op, const vector<int>& min_rank) const
    {
        if (dynamic_cast<const NaryEltwiseLayer*>(op)) {
            int r = -1;
            for (Arg inp : op->inputs)
                r = std::max(r, rankOf(min_rank, inp));
            return r;
        }
        if (const GemmLayer* gemm = dynamic_cast<const GemmLayer*>(op))
            return gemm->flatten_a ? 2 : (op->inputs.empty() ? -1 : rankOf(min_rank, op->inputs[0]));
        if (const MatMulLayer* mm = dynamic_cast<const MatMulLayer*>(op)) {
            if (op->inputs.empty() || dynamic_cast<const MatMulInt8Layer*>(op))
                return -1;
            const int b_rank = !mm->blobs.empty() ? mm->blobs[0].dims
                                                  : (op->inputs.size() >= 2 ? rankOf(min_rank, op->inputs[1]) : -1);
            const int r = rankOf(min_rank, op->inputs[0]);
            if (b_rank < 2 || r < 1)
                return -1;
            return r == 1 ? b_rank - 1 : std::max(r, b_rank);
        }
        if (const Reshape2Layer* rs = dynamic_cast<const Reshape2Layer*>(op)) {
            if (op->inputs.size() == 1 && rs->newShapeDesc.dims >= 0)
                return rs->newShapeDesc.dims;
            if (op->inputs.size() >= 2 && netimpl->isConstArg(op->inputs[1])) {
                const Mat& shp = netimpl->argTensor(op->inputs[1]);
                if (shp.dims <= 1)
                    return (int)shp.total();
            }
            return -1;
        }
        if (!op->inputs.empty() && keepsFirstInputRank(op))
            return rankOf(min_rank, op->inputs[0]);
        return -1;
    }

    void propagateMinRank(const LayerInfo* op, vector<int>& min_rank) const
    {
        if (dynamic_cast<const Split2Layer*>(op)) {
            const int r = op->inputs.empty() ? -1 : rankOf(min_rank, op->inputs[0]);
            for (Arg out : op->outputs)
                setMinRank(min_rank, out, r);
            return;
        }
        if (!op->outputs.empty())
            setMinRank(min_rank, op->outputs[0], inferOutputMinRank(op, min_rank));
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

        vector<Ptr<LayerInfo>> newprog = prog;
        bool changed = false;

        vector<int> usecounts;
        netimpl->useCounts(usecounts);
        std::map<int, int> consumer;  // arg -> the op that uses it (read only when it has one user)
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            for (Arg inp : prog[i]->inputs)
                consumer[inp.idx] = (int)i;
        }
        vector<bool> dropped(nops, false);

        vector<int> min_rank(netimpl->args.size(), -1);
        for (size_t idx = 1; idx < netimpl->args.size(); idx++) {
            Arg a((int)idx);
            if (netimpl->isConstArg(a))
                min_rank[idx] = netimpl->argTensor(a).dims;
            else if (netimpl->argData(a).kind == DNN_ARG_INPUT)
                min_rank[idx] = netimpl->argData(a).shape.dims;
        }
        for (Arg inp : graph->inputs())
            setMinRank(min_rank, inp, netimpl->argData(inp).shape.dims);

        for (size_t i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& layer = newprog[i];
            if (!layer || dropped[i]) continue;
            propagateMinRank(layer.get(), min_rank);

            MatMulLayer* mm = dynamic_cast<MatMulLayer*>(layer.get());
            if (!mm) continue;
            // MatMulInt8Layer carries quantization state that this rewriter
            // doesn't know how to translate to Gemm.
            if (dynamic_cast<MatMulInt8Layer*>(layer.get())) continue;

            // Constant B lives in blobs[0] post-parse; if blobs is empty the
            // MatMul has a runtime B and isn't a candidate.
            if (layer->blobs.empty() || layer->blobs.size() > 2) continue;

            if (mm->trans_a) continue;            // ND-A flatten assumes K = last axis

            const Mat& B = layer->blobs[0];
            if (B.dims != 2 || B.type() != CV_32F) continue;

            // Single runtime input expected (the const B was absorbed into blobs).
            if (layer->inputs.size() != 1) continue;

            int N = mm->trans_b ? B.size[0] : B.size[1];
            int K = mm->trans_b ? B.size[1] : B.size[0];
            (void)K;
            if (N <= 0) continue;

            const int a_min_rank = rankOf(min_rank, layer->inputs[0]);
            bool have_bias = layer->blobs.size() == 2;
            if (have_bias) {
                const Mat& bias = layer->blobs[1];
                if (bias.type() != CV_32F) continue;
                int total = (int)bias.total();
                if (total != 1 && (bias.dims < 1 || total != N || bias.size[bias.dims - 1] != N))
                    continue;
                // A 0-D or 1-D bias never raises the rank above A's (rank >= 1); higher-rank biases
                // require A to be known to have at least as many dimensions, or Gemm drops leading axes.
                if (bias.dims > 1 && a_min_rank < bias.dims)
                    continue;
            }

            Mat Bw = B;
            Mat bias;
            if (have_bias) {
                const Mat& raw_bias = layer->blobs[1];
                Mat cont = raw_bias.isContinuous() ? raw_bias : raw_bias.clone();
                bias = cont.reshape(1, std::vector<int>{(int)cont.total()});
            }
            vector<Arg> outputs = layer->outputs;
            if (mm->alpha == 1.f && mm->beta == 1.f)
                outputs = foldChannelAffine(newprog, layer, consumer, usecounts,
                                            dropped, mm->trans_b, N, Bw, bias);
            have_bias = !bias.empty();

            // Build the Gemm replacement.
            LayerParams gp;
            gp.name = layer->name;        // keep the name for profiling continuity
            gp.type = "Gemm";
            gp.set("transA", false);
            gp.set("transB", mm->trans_b);
            gp.set("alpha", mm->alpha);
            gp.set("beta",  mm->beta);
            gp.set("constB", true);
            gp.set("have_bias", have_bias);
            gp.set("const_C", have_bias);
            // flatten_a=false: keep A's leading dims so downstream consumers
            // see the same shape they did when the producer was a MatMul.
            gp.set("flatten_a", false);
            // For the new GemmLayerImpl flatten_a=false bias path the bias is
            // restricted to scalar or [N]; both of those map to real_ndims_C
            // <= 1, but we don't actually consult it in that path — set it
            // anyway so the Ngraph/CANN backends still get a sensible value.
            if (have_bias) {
                int total = (int)bias.total();
                gp.set("real_ndims_C", total == 1 ? 0 : 1);
            }

            gp.blobs.push_back(Bw);
            if (have_bias) gp.blobs.push_back(bias);

            Ptr<LayerInfo> gemm = LayerFactory::createLayerInstance("Gemm", gp);
            CV_Assert(gemm);  // the Mul/Add layers folded into it are already marked as dropped
            gemm->inputs  = layer->inputs;
            gemm->outputs = outputs;
            gemm->netimpl = netimpl;

            if (!outputs.empty())
                setMinRank(min_rank, outputs[0], a_min_rank);
            newprog[i] = gemm;
            changed = true;
            modified = true;
        }

        if (changed) {
            vector<Ptr<LayerInfo>> kept;
            kept.reserve(nops);
            for (size_t i = 0; i < nops; i++)
                if (!dropped[i] && newprog[i])
                    kept.push_back(newprog[i]);
            graph->setProg(kept);
        }
        return modified;
    }

    Net::Impl* netimpl;
};

void Net::Impl::fuseMatMulConstBToGemm()
{
    ModelFusionMatMulToGemm pass(this);
    pass.fuse();
}

CV__DNN_INLINE_NS_END
}}
