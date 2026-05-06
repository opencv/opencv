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

        vector<Ptr<Layer>> newprog = prog;
        bool changed = false;

        for (size_t i = 0; i < nops; i++) {
            const Ptr<Layer>& layer = newprog[i];
            if (!layer) continue;

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

            const bool have_bias = layer->blobs.size() == 2;
            if (have_bias) {
                const Mat& bias = layer->blobs[1];
                if (bias.type() != CV_32F) continue;
                int total = (int)bias.total();
                if (total != 1 && total != N) continue;
            }

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
                int total = (int)layer->blobs[1].total();
                gp.set("real_ndims_C", total == 1 ? 0 : 1);
            }

            gp.blobs.push_back(B);
            if (have_bias) gp.blobs.push_back(layer->blobs[1]);

            Ptr<Layer> gemm = LayerFactory::createLayerInstance("Gemm", gp);
            if (!gemm) continue;
            gemm->inputs  = layer->inputs;
            gemm->outputs = layer->outputs;
            gemm->netimpl = netimpl;

            newprog[i] = gemm;
            changed = true;
            modified = true;
        }

        if (changed) graph->setProg(newprog);
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
