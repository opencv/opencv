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
//   A(any rank) -> MatMul[const B (2D), optional const bias [N] / scalar]
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
//   - if a bias is present, it must be scalar (total==1) or 1D length-N. The
//     Gemm flatten_a=false bias path tiles a per-row pattern, so 2D / per-row
//     biases like [M, N] aren't supported by this rewriter (they would change
//     value across the flattened rows).
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
                                  const std::map<int, int>& consumer,
                                  const std::set<int>& externalArgs, vector<int>& usecounts,
                                  vector<bool>& dropped, bool trans_b, int N,
                                  Mat& B, Mat& bias) const
    {
        vector<Arg> outputs = layer->outputs;
        bool cloned = false;
        while (outputs.size() == 1) {
            const int out = outputs[0].idx;
            if (usecounts[out] != 1 || externalArgs.count(out)) break;
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
        std::set<int> externalArgs;
        for (Arg out : graph->outputs())
            externalArgs.insert(out.idx);
        std::map<int, int> consumer;  // arg -> the op that uses it (read only when it has one user)
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            for (Arg inp : prog[i]->inputs)
                consumer[inp.idx] = (int)i;
        }
        vector<bool> dropped(nops, false);

        for (size_t i = 0; i < nops; i++) {
            const Ptr<LayerInfo>& layer = newprog[i];
            if (!layer || dropped[i]) continue;

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

            bool have_bias = layer->blobs.size() == 2;
            if (have_bias) {
                const Mat& bias = layer->blobs[1];
                if (bias.type() != CV_32F) continue;
                int total = (int)bias.total();
                if (total != 1 && total != N) continue;
            }

            Mat Bw = B, bias = have_bias ? layer->blobs[1] : Mat();
            vector<Arg> outputs = layer->outputs;
            if (mm->alpha == 1.f && mm->beta == 1.f)
                outputs = foldChannelAffine(newprog, layer, consumer, externalArgs, usecounts,
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
