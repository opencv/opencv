// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

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

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

namespace {

static bool readGemmWeight(const Ptr<Layer>& l, bool trans_b, Mat& W_out)
{
    if (l->blobs.empty()) return false;
    const Mat& W = l->blobs[0];
    if (W.dims != 2 || W.type() != CV_32F) return false;
    if (trans_b) {
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
}

struct ModelFusionSharedGemm
{
    explicit ModelFusionSharedGemm(Net::Impl* netimpl_) : netimpl(netimpl_) {}

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
        bool   flatten_a = true;
    };

    bool inspectGemm(const vector<Ptr<Layer>>& prog, int idx, GemmInfo& info) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx]) return false;
        const Ptr<Layer>& l = prog[idx];
        GemmLayer* g = dynamic_cast<GemmLayer*>(l.get());
        if (!g) return false;

        if (l->inputs.size() != 1) return false;
        if (l->blobs.empty() || l->blobs.size() > 2) return false;

        const Mat& W = l->blobs[0];
        if (W.dims != 2 || W.type() != CV_32F) return false;

        bool trans_a = g->trans_a;
        bool trans_b = g->trans_b;
        float alpha  = g->alpha;
        float beta   = g->beta;

        if (trans_a) return false;
        if (alpha != 1.f) return false;
        if (l->blobs.size() == 2 && beta != 1.f) return false;

        int K = trans_b ? W.size[1] : W.size[0];
        int N = trans_b ? W.size[0] : W.size[1];
        if (K <= 0 || N <= 0) return false;

        if (l->blobs.size() == 2) {
            const Mat& b = l->blobs[1];
            if (b.type() != CV_32F) return false;
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
        info.flatten_a = g->flatten_a;
        return true;
    }

    void fuse() { fuseGraph(netimpl->mainGraph); }

    bool fuseGraph(Ptr<Graph>& graph)
    {
        const vector<Ptr<Layer>>& prog = graph->prog();
        size_t nops = prog.size();

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

            bool all_have_bias = infos[0].has_bias;
            bool uniform = true;
            for (auto& info : infos)
                if (info.has_bias != all_have_bias) { uniform = false; break; }
            if (!uniform) continue;

            // Require uniform flatten_a too — the fused Gemm has a single
            // value, and mixing 2D-output and ND-output downstream consumers
            // would need different post-fusion shape handling.
            bool all_flatten_a = infos[0].flatten_a;
            for (auto& info : infos)
                if (info.flatten_a != all_flatten_a) { uniform = false; break; }
            if (!uniform) continue;

            int K = infos[0].K;
            int total_N = 0;
            for (auto& info : infos) total_N += info.N;

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

            int insert_pos = (int)nops;
            for (auto& info : infos) insert_pos = std::min(insert_pos, info.layer_idx);

            LayerParams fp;
            fp.name = prog[infos[0].layer_idx]->name + "_shared_input_fused";
            fp.type = "Gemm";
            fp.set("transA", false);
            fp.set("transB", false);
            fp.set("alpha", 1.f);
            fp.set("beta",  1.f);
            fp.set("flatten_a", all_flatten_a);
            // Mirror the constB / const_C / have_bias signalling that the
            // GemmLayerImpl's getOpMode() reads from LayerParams. We always
            // ship the weights as a constant blob, and (when biased) the bias
            // too — so this fused Gemm has only one runtime input.
            fp.set("constB", true);
            fp.set("have_bias", all_have_bias);
            fp.set("const_C", all_have_bias);
            if (all_have_bias) fp.set("real_ndims_C", 1);
            fp.blobs.push_back(W_concat);
            if (all_have_bias) fp.blobs.push_back(b_concat);

            Ptr<Layer> fused = LayerFactory::createLayerInstance("Gemm", fp);
            if (!fused) continue;

            string fused_out_name = fp.name + "_out";
            Arg fused_out_arg = netimpl->getArg(fused_out_name);
            fused->inputs  = { infos[0].input };
            fused->outputs = { fused_out_arg };
            fused->netimpl = netimpl;

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
