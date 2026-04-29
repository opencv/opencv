// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Fuses multiple Gemm layers that share the same input into one wider Gemm.
//
// Pattern:
//   X --+--> Gemm[W0, b0] -> Y0
//       +--> Gemm[W1, b1] -> Y1
//       +--> Gemm[W2, b2] -> Y2
//       ...
//
// Replacement:
//   X -> Gemm[concat(Wi), concat(bi)] -> Y_concat
//        Y_concat -> Slice2(0, N0)         -> Y0
//                 -> Slice2(N0, N0+N1)     -> Y1
//                 -> Slice2(N0+N1, ...)    -> Y2
//                 ...
//
// Triggers when:
//   - At least 2 Gemm layers share the same input arg.
//   - Each Gemm has a const weight blob and (optionally) const bias.
//   - All Gemms have identical (alpha, beta, trans_a, trans_b) and matching K.
//
// Win comes from running one wider matmul (better thread utilization, single
// packA / fewer parallel-for setups) instead of several narrow ones. The added
// slices copy data, but each panel is independent and the panels are written
// contiguously so the slices stream from L2.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

namespace {

// Read [rows, cols] float weight from blobs[0], honoring trans_b. The returned
// matrix is in [K, N] (K = input_hidden, N = output_hidden) layout regardless
// of how the underlying blob was stored.
static bool readGemmWeight(const Ptr<Layer>& l, bool trans_b, Mat& W_out)
{
    if (l->blobs.empty()) return false;
    const Mat& W = l->blobs[0];
    if (W.dims != 2 || W.type() != CV_32F) return false;
    if (trans_b) {
        // Stored as [N, K]; transpose to [K, N].
        cv::transpose(W, W_out);
    } else {
        W.copyTo(W_out);
    }
    return true;
}

static bool readGemmBias(const Ptr<Layer>& l, Mat& b_out)
{
    if (l->blobs.size() < 2) { b_out.release(); return true; }
    const Mat& b = l->blobs[1];
    if (b.type() != CV_32F) return false;
    b.copyTo(b_out);
    return true;
}

} // anonymous namespace

struct ModelFusionSharedGemm
{
    explicit ModelFusionSharedGemm(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    // Returns true if `l` is a Gemm with a const weight blob, no extra runtime
    // inputs (only the shared activation), and parameters compatible with the
    // wider-fusion path. Records the trans_b/M/N/K/alpha/beta into out fields.
    struct GemmInfo
    {
        int    layer_idx = -1;
        Arg    input;
        int    K  = 0;
        int    N  = 0;
        bool   trans_a = false;
        bool   trans_b = false;
        float  alpha = 1.f;
        float  beta  = 1.f;
        bool   has_bias = false;
    };

    bool inspectGemm(const vector<Ptr<Layer>>& prog, int idx, GemmInfo& info) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx]) return false;
        const Ptr<Layer>& l = prog[idx];
        GemmLayer* g = dynamic_cast<GemmLayer*>(l.get());
        if (!g) return false;

        // Only handle the simple "linear with const weights" case: one runtime
        // input, weights (and optional bias) in blobs.
        if (l->inputs.size() != 1) return false;
        if (l->blobs.empty() || l->blobs.size() > 2) return false;

        const Mat& W = l->blobs[0];
        if (W.dims != 2 || W.type() != CV_32F) return false;

        bool trans_a = g->trans_a;
        bool trans_b = g->trans_b;
        float alpha  = g->alpha;
        float beta   = g->beta;

        // We only fuse "vanilla" matmuls: A@B (no transpose on A), and we want
        // the bias term to act in the natural way (beta=1).
        if (trans_a) return false;
        if (alpha != 1.f) return false;
        if (l->blobs.size() == 2 && beta != 1.f) return false;

        // K is the inner dimension. For trans_b: W is [N, K] -> K = W.size[1].
        // Otherwise W is [K, N] -> K = W.size[0].
        int K = trans_b ? W.size[1] : W.size[0];
        int N = trans_b ? W.size[0] : W.size[1];
        if (K <= 0 || N <= 0) return false;

        if (l->blobs.size() == 2) {
            const Mat& b = l->blobs[1];
            if (b.type() != CV_32F) return false;
            // Bias must be 1D length N (or trivially broadcastable to N).
            if ((int)b.total() != N) return false;
        }

        info.layer_idx = idx;
        info.input     = l->inputs[0];
        info.K         = K;
        info.N         = N;
        info.trans_a   = trans_a;
        info.trans_b   = trans_b;
        info.alpha     = alpha;
        info.beta      = beta;
        info.has_bias  = (l->blobs.size() == 2);
        return true;
    }

    void fuse() { fuseGraph(netimpl->mainGraph); }

    bool fuseGraph(Ptr<Graph>& graph)
    {
        const vector<Ptr<Layer>>& prog = graph->prog();
        size_t nops = prog.size();

        // Group eligible Gemm layers by their (input, trans_b, K, alpha, beta)
        // signature. We require trans_b/K/alpha/beta to match across the group
        // so the fused Gemm has consistent shape and semantics.
        struct Key { int input_idx; int trans_b; int K; };
        auto keyLess = [](const Key& a, const Key& b) {
            if (a.input_idx != b.input_idx) return a.input_idx < b.input_idx;
            if (a.trans_b   != b.trans_b)   return a.trans_b   < b.trans_b;
            return a.K < b.K;
        };
        std::map<Key, vector<GemmInfo>, decltype(keyLess)> groups(keyLess);

        for (size_t i = 0; i < nops; i++) {
            GemmInfo info;
            if (!inspectGemm(prog, (int)i, info)) continue;
            Key k{info.input.idx, info.trans_b ? 1 : 0, info.K};
            groups[k].push_back(info);
        }

        bool modified = false;
        std::set<int> removed_ops;
        vector<std::pair<int, vector<Ptr<Layer>>>> insertions;  // (insert_pos, fused-and-slice layers)

        for (auto& [key, infos] : groups) {
            if (infos.size() < 2) continue;

            // Make sure all members have the same has_bias flag. Mixing biased
            // and bias-free gemms would force us to synthesize zero bias slices
            // — keep it simple and only fuse uniform groups.
            bool all_have_bias = infos[0].has_bias;
            bool uniform = true;
            for (auto& info : infos)
                if (info.has_bias != all_have_bias) { uniform = false; break; }
            if (!uniform) continue;

            int K = infos[0].K;
            int total_N = 0;
            for (auto& info : infos) total_N += info.N;

            // Concatenate weights into a [K, total_N] matrix in MatMul-style
            // (i.e. trans_b == false in the fused Gemm), regardless of how the
            // originals were stored. We then drive the fused Gemm with
            // trans_b=false, so fast_gemm sees a contiguous [K, total_N] B.
            int wshape[] = { K, total_N };
            Mat W_concat(2, wshape, CV_32F);
            int col_offset = 0;
            for (auto& info : infos) {
                Mat W;
                if (!readGemmWeight(prog[info.layer_idx], info.trans_b, W)) {
                    uniform = false; break;
                }
                CV_Assert(W.size[0] == K && W.size[1] == info.N);
                for (int r = 0; r < K; r++) {
                    float* dst = W_concat.ptr<float>(r) + col_offset;
                    memcpy(dst, W.ptr<float>(r), info.N * sizeof(float));
                }
                col_offset += info.N;
            }
            if (!uniform) continue;

            Mat b_concat;
            if (all_have_bias) {
                int bshape[] = { total_N };
                b_concat.create(1, bshape, CV_32F);
                int b_offset = 0;
                for (auto& info : infos) {
                    Mat b;
                    if (!readGemmBias(prog[info.layer_idx], b)) {
                        uniform = false; break;
                    }
                    CV_Assert((int)b.total() == info.N);
                    memcpy(b_concat.ptr<float>() + b_offset, b.ptr<float>(),
                           info.N * sizeof(float));
                    b_offset += info.N;
                }
                if (!uniform) continue;
            }

            // Find an insertion position: just before the earliest member, so
            // its output (the fused result) is materialized before any slice
            // consumes it.
            int insert_pos = (int)nops;
            for (auto& info : infos) insert_pos = std::min(insert_pos, info.layer_idx);

            // Construct the fused Gemm.
            LayerParams fp;
            fp.name = prog[infos[0].layer_idx]->name + "_shared_input_fused";
            fp.type = "Gemm";
            fp.set("transA", false);
            fp.set("transB", false);
            fp.set("alpha", 1.f);
            fp.set("beta",  1.f);
            fp.blobs.push_back(W_concat);
            if (all_have_bias) fp.blobs.push_back(b_concat);

            Ptr<Layer> fused = LayerFactory::createLayerInstance("Gemm", fp);
            if (!fused) continue;

            // The fused output needs a fresh arg in the graph.
            string fused_out_name = fp.name + "_out";
            Arg fused_out_arg = netimpl->getArg(fused_out_name);
            fused->inputs  = { infos[0].input };
            fused->outputs = { fused_out_arg };
            fused->netimpl = netimpl;

            // Build one Slice2 per member taking [..., col_lo : col_hi] of the
            // fused output. Slice2 expects starts/ends/axes/steps as input args.
            vector<Ptr<Layer>> slices;
            int col_cursor = 0;
            for (size_t s = 0; s < infos.size(); s++) {
                int N_s = infos[s].N;
                int begin = col_cursor;
                int end   = col_cursor + N_s;
                col_cursor += N_s;

                LayerParams sp;
                sp.name = prog[infos[s].layer_idx]->name + "_shared_input_slice";
                sp.type = "Slice2";

                Mat starts(1, 1, CV_32S); starts.at<int>(0) = begin;
                Mat ends  (1, 1, CV_32S); ends.at<int>(0)   = end;
                Mat axes  (1, 1, CV_32S); axes.at<int>(0)   = -1;  // last axis
                Mat steps (1, 1, CV_32S); steps.at<int>(0)  = 1;

                Arg starts_arg = netimpl->newConstArg(sp.name + "_starts", starts);
                Arg ends_arg   = netimpl->newConstArg(sp.name + "_ends",   ends);
                Arg axes_arg   = netimpl->newConstArg(sp.name + "_axes",   axes);
                Arg steps_arg  = netimpl->newConstArg(sp.name + "_steps",  steps);

                Ptr<Layer> slice = LayerFactory::createLayerInstance("Slice2", sp);
                if (!slice) { uniform = false; break; }
                slice->inputs  = { fused_out_arg, starts_arg, ends_arg, axes_arg, steps_arg };
                slice->outputs = prog[infos[s].layer_idx]->outputs;
                slice->netimpl = netimpl;
                slices.push_back(slice);
            }
            if (!uniform) continue;

            for (auto& info : infos) removed_ops.insert(info.layer_idx);
            vector<Ptr<Layer>> bundle;
            bundle.push_back(fused);
            for (auto& s : slices) bundle.push_back(s);
            insertions.emplace_back(insert_pos, std::move(bundle));
            modified = true;
        }

        if (!modified) return false;

        std::sort(insertions.begin(), insertions.end(),
                  [](auto& a, auto& b) { return a.first < b.first; });

        vector<Ptr<Layer>> newprog;
        size_t ins_idx = 0;
        for (size_t i = 0; i < nops; i++) {
            while (ins_idx < insertions.size() &&
                   insertions[ins_idx].first == (int)i) {
                for (auto& l : insertions[ins_idx].second) newprog.push_back(l);
                ins_idx++;
            }
            if (removed_ops.count((int)i)) continue;
            newprog.push_back(prog[i]);
        }
        graph->setProg(newprog);
        return true;
    }

    Net::Impl* netimpl;
};

void Net::Impl::fuseSharedInputGemm()
{
    ModelFusionSharedGemm pass(this);
    pass.fuse();
}

CV__DNN_INLINE_NS_END
}}
