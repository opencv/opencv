// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Fuses multi-head attention subgraphs into a single Attention layer.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

struct ModelFusionAttention
{
    ModelFusionAttention(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    void fuse()
    {
        fuseGraph(netimpl->mainGraph);
    }

    int singleConsumer(Arg a) const
    {
        auto it = consumers_.find(a.idx);
        if (it == consumers_.end() || it->second.size() != 1)
            return -1;
        return it->second[0];
    }

    bool isReshape(const vector<Ptr<Layer>>& prog, int idx) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        return dynamic_cast<Reshape2Layer*>(prog[idx].get()) != nullptr;
    }

    bool isTranspose(const vector<Ptr<Layer>>& prog, int idx) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        return dynamic_cast<TransposeLayer*>(prog[idx].get()) != nullptr;
    }

    bool isSoftmax(const vector<Ptr<Layer>>& prog, int idx) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        return prog[idx]->type == "Softmax";
    }

    bool isMatMul(const vector<Ptr<Layer>>& prog, int idx) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        return dynamic_cast<MatMulLayer*>(prog[idx].get()) != nullptr;
    }

    static bool isProjCandidate(const Ptr<Layer>& l)
    {
        if (l->blobs.empty() || l->inputs.size() != 1) return false;
        if (dynamic_cast<MatMulLayer*>(l.get()))
            return true;
        GemmLayer* g = dynamic_cast<GemmLayer*>(l.get());
        if (!g) return false;
        if (g->trans_a || g->alpha != 1.f) return false;
        if (l->blobs.size() == 2 && g->beta != 1.f) return false;
        return true;
    }

    // Returns the projection weight in [K, N] (input_hidden, output_hidden)
    // layout, transposing if the source is a Gemm with trans_b.
    static Mat getProjWeight(const Ptr<Layer>& l)
    {
        const Mat& W = l->blobs[0];
        GemmLayer* g = dynamic_cast<GemmLayer*>(l.get());
        if (g && g->trans_b) {
            Mat Wt;
            cv::transpose(W, Wt);
            return Wt;
        }
        return W;
    }

    bool isScalarBinOp(const vector<Ptr<Layer>>& prog, int idx,
                       NaryEltwiseLayer::OPERATION op, float* val) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(prog[idx].get());
        if (!elt || elt->op != op)
            return false;
        const auto& inputs = prog[idx]->inputs;
        if (inputs.size() != 2) return false;
        for (int k = 0; k < 2; k++) {
            Arg inp = inputs[k];
            if (netimpl->isConstArg(inp)) {
                Mat t = netimpl->argTensor(inp);
                if (t.total() == 1 && t.type() == CV_32F) {
                    *val = t.at<float>(0);
                    return true;
                }
            }
        }
        return false;
    }

    bool isScalarMul(const vector<Ptr<Layer>>& prog, int idx, float* val) const
    {
        return isScalarBinOp(prog, idx, NaryEltwiseLayer::OPERATION::PROD, val);
    }

    bool isScalarDiv(const vector<Ptr<Layer>>& prog, int idx, float* val) const
    {
        return isScalarBinOp(prog, idx, NaryEltwiseLayer::OPERATION::DIV, val);
    }

    // True if `arg` is produced by the dynamic scale chain Sqrt<-Cast<-Div(1,.)<-Sqrt<-Cast<-Slice<-Shape; visited ops are appended to `chain_ops`.
    bool isRuntimeQKScaleChain(const vector<Ptr<Layer>>& prog, Arg arg,
                                std::set<int>& chain_ops) const
    {
        const std::vector<std::string> expected = {
            "Sqrt", "Cast2", "NaryEltwise" /*Div*/, "Sqrt",
            "Cast2", "Slice2", "Shape"
        };
        Arg cur = arg;
        for (const std::string& want : expected) {
            auto it = producer_.find(cur.idx);
            if (it == producer_.end()) return false;
            int idx = it->second;
            if (idx < 0 || idx >= (int)prog.size() || !prog[idx]) return false;
            const Ptr<Layer>& l = prog[idx];
            if (want == "NaryEltwise") {
                NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(l.get());
                if (!elt || elt->op != NaryEltwiseLayer::OPERATION::DIV) return false;

                bool one_seen = false;
                Arg runtime_in;
                bool runtime_seen = false;
                for (Arg in : l->inputs) {
                    if (netimpl->isConstArg(in)) {
                        Mat t = netimpl->argTensor(in);
                        if (t.total() != 1) return false;
                        float v = 0.f;
                        if      (t.type() == CV_32F) v = t.at<float>(0);
                        else if (t.type() == CV_64F) v = (float)t.at<double>(0);
                        else return false;
                        if (std::abs(v - 1.f) > 1e-5f) return false;
                        one_seen = true;
                    } else {
                        runtime_in = in;
                        runtime_seen = true;
                    }
                }
                if (!one_seen || !runtime_seen) return false;
                chain_ops.insert(idx);
                cur = runtime_in;
            } else {
                if (l->type != want) return false;
                chain_ops.insert(idx);
                if (l->inputs.empty()) return false;
                cur = l->inputs[0];
            }
        }
        return true;
    }

    // Accept Add op with exactly two inputs; identify the non-constant runtime
    // input (the mask tensor). Returns false if the Add doesn't match.
    bool isMaskAdd(const vector<Ptr<Layer>>& prog, int idx, Arg* out_mask) const
    {
        if (idx < 0 || idx >= (int)prog.size() || !prog[idx])
            return false;
        NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(prog[idx].get());
        if (!elt || elt->op != NaryEltwiseLayer::OPERATION::ADD)
            return false;
        const auto& inputs = prog[idx]->inputs;
        if (inputs.size() != 2) return false;
        *out_mask = inputs[1];
        return true;
    }

    // Extract a scalar integer from a const-valued arg, possibly wrapped in an
    // Unsqueeze of a scalar const. Returns -1 if extraction fails.
    int extractConstInt(const vector<Ptr<Layer>>& prog, Arg a) const
    {
        auto readScalar = [&](Arg x) -> int {
            if (!netimpl->isConstArg(x)) return -1;
            Mat t = netimpl->argTensor(x);
            if (t.total() != 1) return -1;
            if (t.type() == CV_64S) return (int)t.at<int64_t>(0);
            if (t.type() == CV_32S) return (int)t.at<int32_t>(0);
            return -1;
        };
        int v = readScalar(a);
        if (v > 0) return v;
        auto it = producer_.find(a.idx);
        if (it == producer_.end()) return -1;
        int pidx = it->second;
        if (pidx < 0 || pidx >= (int)prog.size() || !prog[pidx]) return -1;
        if (prog[pidx]->type == "Unsqueeze" && !prog[pidx]->inputs.empty())
            return readScalar(prog[pidx]->inputs[0]);
        return -1;
    }

    void collectShapeChain(const vector<Ptr<Layer>>& prog, int concat_idx,
                           std::set<int>& chain) const
    {
        if (concat_idx < 0 || concat_idx >= (int)prog.size() || !prog[concat_idx])
            return;
        chain.insert(concat_idx);
        for (Arg ci : prog[concat_idx]->inputs) {
            if (netimpl->isConstArg(ci)) continue;
            int cur = -1;
            auto it = producer_.find(ci.idx);
            if (it != producer_.end()) cur = it->second;
            while (cur >= 0 && (int)prog.size() > cur && prog[cur]) {
                const std::string& t = prog[cur]->type;
                if (t != "Unsqueeze" && t != "Gather2" && t != "Shape") break;
                Arg cur_out = prog[cur]->outputs[0];
                auto cit = consumers_.find(cur_out.idx);
                if (cit == consumers_.end()) break;
                bool all_in_chain = true;
                for (int c : cit->second) if (!chain.count(c)) { all_in_chain = false; break; }
                if (!all_in_chain) break;
                chain.insert(cur);
                if (prog[cur]->inputs.empty()) break;
                Arg prev = prog[cur]->inputs[0];
                auto pit = producer_.find(prev.idx);
                if (pit == producer_.end()) break;
                cur = pit->second;
            }
        }
    }

    template <class Pred>
    int findMatchingConsumer(const vector<Ptr<Layer>>& prog, Arg out,
                             Pred pred, std::set<int>* extra_shape_ops) const
    {
        auto it = consumers_.find(out.idx);
        if (it == consumers_.end()) return -1;
        int matched = -1;
        for (int c : it->second) {
            if (c < 0 || c >= (int)prog.size() || !prog[c]) return -1;
            if (pred(prog[c].get())) {
                if (matched >= 0) return -1;
                matched = c;
            } else if (prog[c]->type == "Shape") {
                if (extra_shape_ops) extra_shape_ops->insert(c);
            } else {
                return -1;
            }
        }
        return matched;
    }

    int followProjChain(const vector<Ptr<Layer>>& prog,
                        int proj_matmul_idx,
                        int* out_reshape_idx,
                        int* out_num_heads,
                        vector<int>* out_perm,
                        std::set<int>* extra_ops_to_remove) const
    {
        if (proj_matmul_idx < 0) return -1;
        Arg proj_out = prog[proj_matmul_idx]->outputs[0];
        int reshape_idx = findMatchingConsumer(prog, proj_out,
            [](Layer* L){ return dynamic_cast<Reshape2Layer*>(L) != nullptr; },
            extra_ops_to_remove);
        if (!isReshape(prog, reshape_idx)) return -1;

        const auto& reshape_inputs = prog[reshape_idx]->inputs;
        if (reshape_inputs.size() < 2) return -1;
        Arg shape_arg = reshape_inputs[1];

        int num_heads = -1;
        if (netimpl->isConstArg(shape_arg)) {
            Mat shape_mat = netimpl->argTensor(shape_arg);
            if (shape_mat.total() != 4) return -1;
            const int64_t* shape_data = shape_mat.ptr<int64_t>();
            num_heads = static_cast<int>(shape_data[2]);
        } else {
            auto it = producer_.find(shape_arg.idx);
            if (it == producer_.end()) return -1;
            int concat_idx = it->second;
            if (concat_idx < 0 || concat_idx >= (int)prog.size() || !prog[concat_idx])
                return -1;
            if (!dynamic_cast<Concat2Layer*>(prog[concat_idx].get()))
                return -1;
            const auto& cinputs = prog[concat_idx]->inputs;
            if (cinputs.size() != 4) return -1;
            num_heads = extractConstInt(prog, cinputs[2]);
            if (num_heads <= 0) return -1;
            if (extra_ops_to_remove)
                collectShapeChain(prog, concat_idx, *extra_ops_to_remove);
        }
        *out_num_heads = num_heads;
        *out_reshape_idx = reshape_idx;

        Arg reshape_out = prog[reshape_idx]->outputs[0];
        int transpose_idx = singleConsumer(reshape_out);
        if (!isTranspose(prog, transpose_idx)) return -1;
        TransposeLayer* tr = dynamic_cast<TransposeLayer*>(prog[transpose_idx].get());
        if (!tr) return -1;
        *out_perm = tr->perm;
        return transpose_idx;
    }

    bool fuseGraph(Ptr<Graph>& graph)
    {
        const vector<Ptr<Layer>>& prog = graph->prog();
        size_t nops = prog.size();

        producer_.clear();
        consumers_.clear();
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            for (Arg out : prog[i]->outputs)
                producer_[out.idx] = (int)i;
            for (Arg inp : prog[i]->inputs)
                consumers_[inp.idx].push_back((int)i);
        }

        std::map<int, vector<int>> qkv_candidates;
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i]) continue;
            if (!isProjCandidate(prog[i])) continue;
            Arg inp = prog[i]->inputs[0];
            qkv_candidates[inp.idx].push_back((int)i);
        }

        bool modified = false;
        std::set<int> removed_ops;

        for (auto& [inp_idx, matmul_indices] : qkv_candidates) {
            if (matmul_indices.size() < 3) continue;

            for (size_t qi = 0; qi + 2 < matmul_indices.size(); qi++)
            for (size_t ki = qi + 1; ki + 1 < matmul_indices.size(); ki++)
            for (size_t vi = ki + 1; vi < matmul_indices.size(); vi++)
            {
                int indices[3] = { matmul_indices[qi], matmul_indices[ki], matmul_indices[vi] };

                bool any_removed = false;
                for (int idx : indices)
                    if (removed_ops.count(idx)) any_removed = true;
                if (any_removed) continue;

                bool shapes_ok = true;
                for (int k = 0; k < 3; k++) {
                    const Mat& w = prog[indices[k]]->blobs[0];
                    if (w.dims != 2) { shapes_ok = false; break; }
                }
                if (!shapes_ok) continue;

                int reshape_idx[3], num_heads[3], transpose_idx[3];
                vector<int> perms[3];
                std::set<int> extra_ops;
                bool pattern_ok = true;
                for (int k = 0; k < 3; k++) {
                    transpose_idx[k] = followProjChain(prog, indices[k],
                                                       &reshape_idx[k],
                                                       &num_heads[k], &perms[k],
                                                       &extra_ops);
                    if (transpose_idx[k] < 0) { pattern_ok = false; break; }
                }
                if (!pattern_ok) continue;
                if (num_heads[0] != num_heads[1] || num_heads[1] != num_heads[2])
                    continue;

                // K is identified by perm [0,2,3,1] (transposes head_dim to second-last);
                // Q and V use perm [0,2,1,3].
                int k_slot = -1;
                vector<int> perm_k = {0, 2, 3, 1};
                for (int k = 0; k < 3; k++)
                    if (perms[k] == perm_k) k_slot = k;
                if (k_slot < 0) continue;

                int remaining[2];
                { int j = 0; for (int k = 0; k < 3; k++) if (k != k_slot) remaining[j++] = k; }

                Arg k_tr_out = prog[transpose_idx[k_slot]]->outputs[0];
                // Tolerate a Shape consumer alongside the Mul/MatMul: the runtime-scale chain (Shape->Slice->Cast->Sqrt...) branches off the Q/K transpose.
                int k_next = findMatchingConsumer(prog, k_tr_out,
                    [](Layer* L) {
                        return dynamic_cast<NaryEltwiseLayer*>(L) != nullptr ||
                               dynamic_cast<MatMulLayer*>(L) != nullptr;
                    },
                    &extra_ops);
                int k_mul_idx = -1;
                float k_scale_val = 1.f;
                int qk_matmul_idx = -1;
                bool vit_style;
                bool runtime_qk_scale = false;
                std::set<int> qk_scale_chain_ops;
                if (isScalarMul(prog, k_next, &k_scale_val)) {
                    vit_style = true;
                    k_mul_idx = k_next;
                    qk_matmul_idx = singleConsumer(prog[k_mul_idx]->outputs[0]);
                } else if (isMatMul(prog, k_next)) {
                    vit_style = false;
                    qk_matmul_idx = k_next;
                } else if (k_next >= 0 && k_next < (int)prog.size() && prog[k_next]) {
                    // K_Transpose -> Mul(K, runtime_scale) where scale = Sqrt(Cast(Div(1,Sqrt(Cast(Slice(Shape(...))))))) and the same scale feeds the Q-side Mul (verified later).
                    NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(prog[k_next].get());
                    if (!elt || elt->op != NaryEltwiseLayer::OPERATION::PROD ||
                        prog[k_next]->inputs.size() != 2) {
                        continue;
                    }
                    Arg k_scale_in;
                    bool tensor_input_seen = false;
                    for (Arg in : prog[k_next]->inputs) {
                        if (in.idx == k_tr_out.idx) tensor_input_seen = true;
                        else k_scale_in = in;
                    }
                    if (!tensor_input_seen) continue;
                    if (!isRuntimeQKScaleChain(prog, k_scale_in, qk_scale_chain_ops)) continue;

                    runtime_qk_scale = true;
                    vit_style = true;
                    k_mul_idx = k_next;
                    qk_matmul_idx = singleConsumer(prog[k_mul_idx]->outputs[0]);
                } else {
                    continue;
                }
                if (!isMatMul(prog, qk_matmul_idx)) continue;

                int q_slot = -1, v_slot = -1;
                for (int attempt = 0; attempt < 2; attempt++) {
                    q_slot = remaining[attempt];
                    v_slot = remaining[1 - attempt];

                    int q_mul_idx = -1;
                    float q_scale_val = 1.f;
                    Arg q_tr_out = prog[transpose_idx[q_slot]]->outputs[0];

                    if (vit_style) {
                        int q_next = findMatchingConsumer(prog, q_tr_out,
                            [](Layer* L) {
                                return dynamic_cast<NaryEltwiseLayer*>(L) != nullptr ||
                                       dynamic_cast<MatMulLayer*>(L) != nullptr;
                            },
                            &extra_ops);
                        if (isScalarMul(prog, q_next, &q_scale_val)) {
                            // existing constant-scalar path
                        } else if (runtime_qk_scale && q_next >= 0 &&
                                   q_next < (int)prog.size() && prog[q_next]) {
                            NaryEltwiseLayer* elt = dynamic_cast<NaryEltwiseLayer*>(prog[q_next].get());
                            if (!elt || elt->op != NaryEltwiseLayer::OPERATION::PROD ||
                                prog[q_next]->inputs.size() != 2) {
                                continue;
                            }
                            Arg q_scale_in;
                            bool tensor_input_seen = false;
                            for (Arg in : prog[q_next]->inputs) {
                                if (in.idx == q_tr_out.idx) tensor_input_seen = true;
                                else q_scale_in = in;
                            }
                            if (!tensor_input_seen) continue;
                            if (!isRuntimeQKScaleChain(prog, q_scale_in, qk_scale_chain_ops)) continue;
                        } else {
                            continue;
                        }
                        q_mul_idx = q_next;
                        Arg q_mul_out = prog[q_mul_idx]->outputs[0];
                        int q_dest = singleConsumer(q_mul_out);
                        if (q_dest != qk_matmul_idx) continue;
                    } else {
                        const auto& qk_inputs = prog[qk_matmul_idx]->inputs;
                        bool q_connected = false;
                        for (auto& ai : qk_inputs)
                            if (ai.idx == q_tr_out.idx) q_connected = true;
                        if (!q_connected) continue;
                    }

                    // qk_matmul → [Div(√d_k)] → [Add(mask)] → Softmax   (BERT)
                    //          OR                                 → Softmax   (ViT)
                    Arg cur_out = prog[qk_matmul_idx]->outputs[0];
                    int qk_div_idx = -1;
                    int qk_add_idx = -1;
                    Arg mask_arg;
                    bool has_mask = false;
                    float post_scale_val = 1.f;

                    int cur_consumer = singleConsumer(cur_out);
                    if (!vit_style) {
                        if (!isScalarDiv(prog, cur_consumer, &post_scale_val)) continue;
                        qk_div_idx = cur_consumer;
                        cur_out = prog[qk_div_idx]->outputs[0];
                        cur_consumer = singleConsumer(cur_out);
                        Arg maybe_mask;
                        if (isMaskAdd(prog, cur_consumer, &maybe_mask)) {
                            qk_add_idx = cur_consumer;
                            mask_arg = maybe_mask;
                            has_mask = true;
                            cur_out = prog[qk_add_idx]->outputs[0];
                            cur_consumer = singleConsumer(cur_out);
                        }
                    }
                    int softmax_idx = cur_consumer;
                    if (!isSoftmax(prog, softmax_idx)) continue;

                    Arg softmax_out = prog[softmax_idx]->outputs[0];
                    int attnv_matmul_idx = singleConsumer(softmax_out);
                    if (!isMatMul(prog, attnv_matmul_idx)) continue;

                    Arg v_tr_out = prog[transpose_idx[v_slot]]->outputs[0];
                    const auto& av_inputs = prog[attnv_matmul_idx]->inputs;
                    bool v_connected = false;
                    for (auto& ai : av_inputs)
                        if (ai.idx == v_tr_out.idx) v_connected = true;
                    if (!v_connected) continue;

                    Arg attnv_out = prog[attnv_matmul_idx]->outputs[0];
                    int out_transpose_idx = singleConsumer(attnv_out);
                    if (!isTranspose(prog, out_transpose_idx)) continue;

                    Arg out_tr_out = prog[out_transpose_idx]->outputs[0];
                    int out_reshape_idx = findMatchingConsumer(prog, out_tr_out,
                        [](Layer* L){ return dynamic_cast<Reshape2Layer*>(L) != nullptr; },
                        &extra_ops);
                    if (!isReshape(prog, out_reshape_idx)) continue;

                    const auto& or_inputs = prog[out_reshape_idx]->inputs;
                    if (or_inputs.size() >= 2 && !netimpl->isConstArg(or_inputs[1])) {
                        auto it = producer_.find(or_inputs[1].idx);
                        if (it != producer_.end())
                            collectShapeChain(prog, it->second, extra_ops);
                    }

                    // Normalize each projection weight to [K, N] regardless of
                    // whether the source op is MatMul or Gemm-with-trans_b.
                    Mat Wq = getProjWeight(prog[indices[q_slot]]);
                    Mat Wk = getProjWeight(prog[indices[k_slot]]);
                    Mat Wv = getProjWeight(prog[indices[v_slot]]);
                    int input_hidden = Wq.size[0];
                    int q_hidden = Wq.size[1];
                    int k_hidden = Wk.size[1];
                    int v_hidden = Wv.size[1];
                    int total_hidden = q_hidden + k_hidden + v_hidden;

                    int wshape[] = {input_hidden, total_hidden};
                    Mat W_qkv(2, wshape, CV_32F);
                    for (int r = 0; r < input_hidden; r++) {
                        float* dst = W_qkv.ptr<float>(r);
                        memcpy(dst, Wq.ptr<float>(r), q_hidden * sizeof(float));
                        memcpy(dst + q_hidden, Wk.ptr<float>(r), k_hidden * sizeof(float));
                        memcpy(dst + q_hidden + k_hidden, Wv.ptr<float>(r), v_hidden * sizeof(float));
                    }

                    Mat bias_qkv;
                    bool has_bias = prog[indices[q_slot]]->blobs.size() >= 2;
                    if (has_bias) {
                        const Mat& bq = prog[indices[q_slot]]->blobs[1];
                        const Mat& bk = prog[indices[k_slot]]->blobs[1];
                        const Mat& bv = prog[indices[v_slot]]->blobs[1];
                        int bias_total = (int)(bq.total() + bk.total() + bv.total());
                        bias_qkv.create(1, &bias_total, CV_32F);
                        float* dst = bias_qkv.ptr<float>();
                        memcpy(dst, bq.ptr<float>(), bq.total() * sizeof(float));
                        memcpy(dst + bq.total(), bk.ptr<float>(), bk.total() * sizeof(float));
                        memcpy(dst + bq.total() + bk.total(), bv.ptr<float>(), bv.total() * sizeof(float));
                    }

                    float param_scale;
                    if (runtime_qk_scale) {
                        // Both Q and K were scaled by head_dim^-0.25, so
                        // (Q'*K'^T) = (1/sqrt(d)) * (Q*K^T). The Attention
                        // layer's `scale` param is the divisor applied
                        // before softmax — i.e., sqrt(d).
                        if (num_heads[0] <= 0 || q_hidden % num_heads[0] != 0) continue;
                        int head_dim = q_hidden / num_heads[0];
                        if (head_dim <= 0) continue;
                        param_scale = std::sqrt((float)head_dim);
                    } else {
                        param_scale = vit_style ? (1.f / (q_scale_val * k_scale_val))
                                                : post_scale_val;
                    }

                    LayerParams attn_params;
                    attn_params.name = prog[indices[q_slot]]->name + "_fused_attention";
                    attn_params.type = "Attention";
                    attn_params.set("num_heads", num_heads[0]);
                    DictValue qkv_sizes_param = DictValue::arrayInt(
                        std::vector<int>{q_hidden, k_hidden, v_hidden}.data(), 3);
                    attn_params.set("qkv_hidden_sizes", qkv_sizes_param);
                    attn_params.set("scale", param_scale);
                    attn_params.set("output_ndims", 3);

                    attn_params.blobs.push_back(W_qkv);
                    if (has_bias)
                        attn_params.blobs.push_back(bias_qkv);

                    Ptr<Layer> attn_layer = LayerFactory::createLayerInstance(
                        attn_params.type, attn_params);
                    CV_Assert(attn_layer);

                    Arg shared_input = prog[indices[0]]->inputs[0];
                    if (has_mask)
                        attn_layer->inputs = { shared_input, mask_arg };
                    else
                        attn_layer->inputs = { shared_input };
                    attn_layer->outputs = prog[out_reshape_idx]->outputs;
                    attn_layer->netimpl = netimpl;

                    std::set<int> to_remove = {
                        indices[0], indices[1], indices[2],
                        reshape_idx[0], reshape_idx[1], reshape_idx[2],
                        transpose_idx[0], transpose_idx[1], transpose_idx[2],
                        qk_matmul_idx, softmax_idx, attnv_matmul_idx,
                        out_transpose_idx, out_reshape_idx
                    };
                    if (k_mul_idx >= 0) to_remove.insert(k_mul_idx);
                    if (q_mul_idx >= 0) to_remove.insert(q_mul_idx);
                    if (qk_div_idx >= 0) to_remove.insert(qk_div_idx);
                    if (qk_add_idx >= 0) to_remove.insert(qk_add_idx);
                    for (int op : extra_ops) to_remove.insert(op);

                    // Drop chain ops (Sqrt/Cast/Div/Slice/Shape) only if every consumer is also being removed — by this fusion or another chain op already in the set; the upstream is shared between Q and K branches so the chain is treated as a unit.
                    for (int op_idx : qk_scale_chain_ops) {
                        if (op_idx < 0 || op_idx >= (int)prog.size() || !prog[op_idx]) continue;
                        bool all_consumers_removed = true;
                        for (Arg out : prog[op_idx]->outputs) {
                            auto cit = consumers_.find(out.idx);
                            if (cit == consumers_.end()) continue;
                            for (int c : cit->second) {
                                if (to_remove.count(c) || qk_scale_chain_ops.count(c)) continue;
                                all_consumers_removed = false;
                                break;
                            }
                            if (!all_consumers_removed) break;
                        }
                        if (all_consumers_removed) to_remove.insert(op_idx);
                    }

                    for (int op : to_remove)
                        removed_ops.insert(op);

                    int insert_pos = *std::min_element(to_remove.begin(), to_remove.end());
                    attention_replacements_.push_back({insert_pos, attn_layer});
                    modified = true;
                    break;
                }
            }
        }

        if (modified) {
            vector<Ptr<Layer>> newprog;
            std::sort(attention_replacements_.begin(), attention_replacements_.end(),
                      [](auto& a, auto& b) { return a.first < b.first; });

            size_t repl_idx = 0;
            for (size_t i = 0; i < nops; i++) {
                while (repl_idx < attention_replacements_.size() &&
                       attention_replacements_[repl_idx].first == (int)i) {
                    newprog.push_back(attention_replacements_[repl_idx].second);
                    repl_idx++;
                }
                if (removed_ops.count((int)i))
                    continue;
                newprog.push_back(prog[i]);
            }
            graph->setProg(newprog);
        }

        return modified;
    }

    Net::Impl* netimpl;

private:
    std::map<int, int> producer_;
    std::map<int, vector<int>> consumers_;
    vector<std::pair<int, Ptr<Layer>>> attention_replacements_;
};

void Net::Impl::fuseAttention()
{
    ModelFusionAttention attnFusion(this);
    attnFusion.fuse();
}

CV__DNN_INLINE_NS_END
}}
