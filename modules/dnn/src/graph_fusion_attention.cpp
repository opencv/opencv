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

    // Combined-QKV attention: QKV proj -> Reshape ->
    // Transpose -> 3 Gathers -> QK^T -> Softmax(no mask) -> *V.
    bool tryFuseCombinedQKV(const vector<Ptr<Layer>>& prog, int qkv_matmul_idx,
                            std::set<int>& removed_ops,
                            vector<std::pair<int, Ptr<Layer>>>& replacements)
    {
        if (qkv_matmul_idx < 0 || qkv_matmul_idx >= (int)prog.size() || !prog[qkv_matmul_idx])
            return false;
        if (removed_ops.count(qkv_matmul_idx)) return false;
        if (!isProjCandidate(prog[qkv_matmul_idx])) return false;

        Mat W = getProjWeight(prog[qkv_matmul_idx]);
        if (W.dims != 2 || W.size[1] % 3 != 0) return false;
        const int input_hidden = W.size[0];
        const int proj_hidden  = W.size[1] / 3;
        const int total_hidden = W.size[1];

        Arg cur = prog[qkv_matmul_idx]->outputs[0];

        // Optional bias: separate Add op or Gemm's blobs[1].
        int add_idx = -1;
        Mat bias_mat;
        bool has_bias = false;
        if (prog[qkv_matmul_idx]->blobs.size() >= 2) {
            bias_mat = prog[qkv_matmul_idx]->blobs[1];
            has_bias = bias_mat.total() == (size_t)total_hidden && bias_mat.type() == CV_32F;
            if (!has_bias) return false;
        }
        int next = singleConsumer(cur);
        if (!has_bias && next >= 0 && next < (int)prog.size() && prog[next]) {
            NaryEltwiseLayer* e = dynamic_cast<NaryEltwiseLayer*>(prog[next].get());
            if (e && e->op == NaryEltwiseLayer::OPERATION::ADD &&
                prog[next]->inputs.size() == 2)
            {
                Mat b_candidate;
                for (Arg in : prog[next]->inputs) {
                    if (netimpl->isConstArg(in)) {
                        Mat t = netimpl->argTensor(in);
                        if (t.type() == CV_32F && (int)t.total() == total_hidden)
                            b_candidate = t;
                    }
                }
                if (!b_candidate.empty()) {
                    add_idx = next;
                    bias_mat = b_candidate;
                    has_bias = true;
                    cur = prog[add_idx]->outputs[0];
                    next = singleConsumer(cur);
                }
            }
        }

        if (!isReshape(prog, next)) return false;
        const int reshape_idx = next;
        const auto& rinputs = prog[reshape_idx]->inputs;
        if (rinputs.size() < 2) return false;

        std::set<int> extra_ops;
        int num_heads = -1, head_dim = -1;
        Arg shape_arg = rinputs[1];
        if (netimpl->isConstArg(shape_arg)) {
            Mat sh = netimpl->argTensor(shape_arg);
            if (sh.total() != 5) return false;
            const int64_t* sd = sh.ptr<int64_t>();
            if ((int)sd[2] != 3) return false;
            num_heads = (int)sd[3];
            head_dim  = (int)sd[4];
        } else {
            auto it = producer_.find(shape_arg.idx);
            if (it == producer_.end()) return false;
            int concat_idx = it->second;
            if (concat_idx < 0 || concat_idx >= (int)prog.size() || !prog[concat_idx])
                return false;
            if (!dynamic_cast<Concat2Layer*>(prog[concat_idx].get())) return false;
            const auto& cinputs = prog[concat_idx]->inputs;
            if (cinputs.size() != 5) return false;
            if (extractConstInt(prog, cinputs[2]) != 3) return false;
            num_heads = extractConstInt(prog, cinputs[3]);
            head_dim  = extractConstInt(prog, cinputs[4]);
            if (num_heads <= 0 || head_dim <= 0) return false;
            collectShapeChain(prog, concat_idx, extra_ops);
        }
        if (num_heads * head_dim != proj_hidden) return false;

        // Transpose must bring "3" to front; only perm[0]==2 is required.
        Arg reshape_out = prog[reshape_idx]->outputs[0];
        int transpose_idx = singleConsumer(reshape_out);
        if (!isTranspose(prog, transpose_idx)) return false;
        TransposeLayer* tr = dynamic_cast<TransposeLayer*>(prog[transpose_idx].get());
        if (!tr || tr->perm.size() != 5 || tr->perm[0] != 2) return false;

        Arg tr_out = prog[transpose_idx]->outputs[0];
        auto cit = consumers_.find(tr_out.idx);
        if (cit == consumers_.end() || cit->second.size() != 3) return false;

        int gather_for_idx[3] = { -1, -1, -1 };  // [Q,K,V] = [0,1,2]
        for (int c : cit->second) {
            if (c < 0 || c >= (int)prog.size() || !prog[c]) return false;
            Gather2Layer* g = dynamic_cast<Gather2Layer*>(prog[c].get());
            if (!g || g->axis != 0 || prog[c]->inputs.size() < 2) return false;
            Arg idx_arg = prog[c]->inputs[1];
            if (!netimpl->isConstArg(idx_arg)) return false;
            Mat t = netimpl->argTensor(idx_arg);
            if (t.total() != 1) return false;
            int v;
            if      (t.type() == CV_64S) v = (int)t.at<int64_t>(0);
            else if (t.type() == CV_32S) v = (int)t.at<int32_t>(0);
            else return false;
            if (v < 0 || v > 2 || gather_for_idx[v] != -1) return false;
            gather_for_idx[v] = c;
        }
        const int q_gather = gather_for_idx[0];
        const int k_gather = gather_for_idx[1];
        const int v_gather = gather_for_idx[2];
        if (q_gather < 0 || k_gather < 0 || v_gather < 0) return false;

        Arg q_out = prog[q_gather]->outputs[0];
        Arg k_out = prog[k_gather]->outputs[0];
        Arg v_out = prog[v_gather]->outputs[0];

        int q_mul_idx = singleConsumer(q_out);
        float q_scale = 1.f;
        if (!isScalarMul(prog, q_mul_idx, &q_scale)) return false;
        if (q_scale == 0.f) return false;
        int qk_matmul_idx = singleConsumer(prog[q_mul_idx]->outputs[0]);
        if (!isMatMul(prog, qk_matmul_idx)) return false;

        int k_trans_idx = singleConsumer(k_out);
        if (!isTranspose(prog, k_trans_idx)) return false;
        Arg k_trans_out = prog[k_trans_idx]->outputs[0];
        bool k_connected = false;
        for (Arg in : prog[qk_matmul_idx]->inputs)
            if (in.idx == k_trans_out.idx) { k_connected = true; break; }
        if (!k_connected) return false;

        int softmax_idx = singleConsumer(prog[qk_matmul_idx]->outputs[0]);
        if (!isSoftmax(prog, softmax_idx)) return false;
        int av_matmul_idx = singleConsumer(prog[softmax_idx]->outputs[0]);
        if (!isMatMul(prog, av_matmul_idx)) return false;
        bool v_connected = false;
        for (Arg in : prog[av_matmul_idx]->inputs)
            if (in.idx == v_out.idx) { v_connected = true; break; }
        if (!v_connected) return false;

        int out_trans_idx = singleConsumer(prog[av_matmul_idx]->outputs[0]);
        if (!isTranspose(prog, out_trans_idx)) return false;
        int out_reshape_idx = findMatchingConsumer(prog, prog[out_trans_idx]->outputs[0],
            [](Layer* L){ return dynamic_cast<Reshape2Layer*>(L) != nullptr; },
            &extra_ops);
        if (!isReshape(prog, out_reshape_idx)) return false;

        const auto& or_inputs = prog[out_reshape_idx]->inputs;
        if (or_inputs.size() >= 2 && !netimpl->isConstArg(or_inputs[1])) {
            auto it = producer_.find(or_inputs[1].idx);
            if (it != producer_.end())
                collectShapeChain(prog, it->second, extra_ops);
        }

        // W is already laid out as [Q|K|V] along the output dim.
        Mat W_qkv = W.clone();
        Mat bias_qkv;
        if (has_bias) bias_qkv = bias_mat.clone();

        // Attention `scale` is the pre-softmax divisor; Q was multiplied by
        // q_scale, so scale = 1/q_scale.
        const float param_scale = 1.0f / q_scale;

        LayerParams attn_params;
        attn_params.name = prog[qkv_matmul_idx]->name + "_fused_attention";
        attn_params.type = "Attention";
        attn_params.set("num_heads", num_heads);
        int qkv_sizes[3] = { proj_hidden, proj_hidden, proj_hidden };
        attn_params.set("qkv_hidden_sizes", DictValue::arrayInt(qkv_sizes, 3));
        attn_params.set("scale", param_scale);
        attn_params.set("output_ndims", 3);
        attn_params.blobs.push_back(W_qkv);
        if (has_bias) attn_params.blobs.push_back(bias_qkv);

        Ptr<Layer> attn_layer = LayerFactory::createLayerInstance(attn_params.type, attn_params);
        CV_Assert(attn_layer);
        Arg shared_input = prog[qkv_matmul_idx]->inputs[0];
        attn_layer->inputs  = { shared_input };
        attn_layer->outputs = prog[out_reshape_idx]->outputs;
        attn_layer->netimpl = netimpl;

        std::set<int> to_remove = {
            qkv_matmul_idx, reshape_idx, transpose_idx,
            q_gather, k_gather, v_gather,
            q_mul_idx, k_trans_idx,
            qk_matmul_idx, softmax_idx,
            av_matmul_idx, out_trans_idx, out_reshape_idx
        };
        if (add_idx >= 0) to_remove.insert(add_idx);
        for (int op : extra_ops) to_remove.insert(op);

        (void)input_hidden;  // input_hidden currently unused; kept for symmetry with existing path.

        for (int op : to_remove) removed_ops.insert(op);
        const int insert_pos = *std::min_element(to_remove.begin(), to_remove.end());
        replacements.push_back({insert_pos, attn_layer});
        return true;
    }

    // CLIP-branch trace: arg -> [Transpose3D(K^T)] -> Reshape3D -> Transpose ->
    // Reshape4D -> [Mul(Q scale)] -> [Add(bias)] -> proj_MatMul.
    int traceClipBranch(const vector<Ptr<Layer>>& prog, Arg arg,
                        bool is_q_branch, bool is_k_branch,
                        Mat& out_W, Mat& out_bias, int& out_num_heads,
                        float& out_q_scale,
                        std::set<int>& ops_consumed) const
    {
        Arg cur = arg;

        auto stepProducer = [&](Arg a) -> int {
            auto it = producer_.find(a.idx);
            return it == producer_.end() ? -1 : it->second;
        };

        // K side: peel the (B*H,S,D) -> (B*H,D,S) transpose-3D first.
        if (is_k_branch) {
            int idx = stepProducer(cur);
            if (idx < 0 || !prog[idx]) return -1;
            TransposeLayer* tr = dynamic_cast<TransposeLayer*>(prog[idx].get());
            if (!tr || tr->perm.size() != 3) return -1;
            if (tr->perm[0] != 0 || tr->perm[1] != 2 || tr->perm[2] != 1) return -1;
            ops_consumed.insert(idx);
            cur = prog[idx]->inputs[0];
        }

        // (B,H,S,D) -> (B*H,S,D); shape may be dynamic, validated downstream.
        int r3d_idx = stepProducer(cur);
        if (!isReshape(prog, r3d_idx)) return -1;
        ops_consumed.insert(r3d_idx);
        if (prog[r3d_idx]->inputs.size() < 2) return -1;
        Arg shape_arg_r3d = prog[r3d_idx]->inputs[1];
        if (!netimpl->isConstArg(shape_arg_r3d)) {
            int sh_idx = stepProducer(shape_arg_r3d);
            if (sh_idx >= 0)
                collectShapeChain(prog, sh_idx, ops_consumed);
        }
        cur = prog[r3d_idx]->inputs[0];

        int t4d_idx = stepProducer(cur);
        if (!isTranspose(prog, t4d_idx)) return -1;
        TransposeLayer* tr4 = dynamic_cast<TransposeLayer*>(prog[t4d_idx].get());
        if (!tr4 || tr4->perm.size() != 4) return -1;
        if (tr4->perm[0] != 0 || tr4->perm[1] != 2 ||
            tr4->perm[2] != 1 || tr4->perm[3] != 3) return -1;
        ops_consumed.insert(t4d_idx);
        cur = prog[t4d_idx]->inputs[0];

        // (B,S,H*D) -> (B,S,H,D); num_heads is the third shape entry.
        int r4d_idx = stepProducer(cur);
        if (!isReshape(prog, r4d_idx)) return -1;
        if (prog[r4d_idx]->inputs.size() < 2) return -1;
        Arg shape_arg_r4d = prog[r4d_idx]->inputs[1];
        int num_heads = -1;
        if (netimpl->isConstArg(shape_arg_r4d)) {
            Mat shape_mat = netimpl->argTensor(shape_arg_r4d);
            if (shape_mat.total() != 4) return -1;
            if (shape_mat.type() == CV_64S)
                num_heads = static_cast<int>(shape_mat.ptr<int64_t>()[2]);
            else if (shape_mat.type() == CV_32S)
                num_heads = static_cast<int>(shape_mat.ptr<int32_t>()[2]);
            else return -1;
        } else {
            int concat_idx = stepProducer(shape_arg_r4d);
            if (concat_idx < 0 || !prog[concat_idx]) return -1;
            if (!dynamic_cast<Concat2Layer*>(prog[concat_idx].get())) return -1;
            const auto& cinputs = prog[concat_idx]->inputs;
            if (cinputs.size() != 4) return -1;
            num_heads = extractConstInt(prog, cinputs[2]);
            if (num_heads <= 0) return -1;
            collectShapeChain(prog, concat_idx, ops_consumed);
        }
        if (num_heads <= 0) return -1;
        out_num_heads = num_heads;
        ops_consumed.insert(r4d_idx);
        cur = prog[r4d_idx]->inputs[0];

        out_q_scale = 1.0f;
        if (is_q_branch) {
            int mul_idx = stepProducer(cur);
            if (mul_idx < 0 || !prog[mul_idx]) return -1;
            NaryEltwiseLayer* mul =
                dynamic_cast<NaryEltwiseLayer*>(prog[mul_idx].get());
            if (!mul || mul->op != NaryEltwiseLayer::OPERATION::PROD) return -1;
            if (prog[mul_idx]->inputs.size() != 2) return -1;
            Arg runtime_arg;
            bool got_scale = false, got_runtime = false;
            for (Arg in : prog[mul_idx]->inputs) {
                if (netimpl->isConstArg(in)) {
                    Mat t = netimpl->argTensor(in);
                    if (t.total() != 1) return -1;
                    if      (t.type() == CV_32F) out_q_scale = t.at<float>(0);
                    else if (t.type() == CV_64F) out_q_scale = (float)t.at<double>(0);
                    else return -1;
                    got_scale = true;
                } else {
                    runtime_arg = in;
                    got_runtime = true;
                }
            }
            if (!got_scale || !got_runtime) return -1;
            ops_consumed.insert(mul_idx);
            cur = runtime_arg;
        }

        int next_idx = stepProducer(cur);
        if (next_idx < 0 || !prog[next_idx]) return -1;

        out_bias = Mat();
        int mm_idx = -1;
        if (dynamic_cast<NaryEltwiseLayer*>(prog[next_idx].get())) {
            NaryEltwiseLayer* add =
                dynamic_cast<NaryEltwiseLayer*>(prog[next_idx].get());
            if (!add || add->op != NaryEltwiseLayer::OPERATION::ADD) return -1;
            if (prog[next_idx]->inputs.size() != 2) return -1;
            Arg bias_arg;
            Arg matmul_out_arg;
            bool got_bias = false, got_runtime2 = false;
            for (Arg in : prog[next_idx]->inputs) {
                if (netimpl->isConstArg(in)) {
                    bias_arg = in;
                    got_bias = true;
                } else {
                    matmul_out_arg = in;
                    got_runtime2 = true;
                }
            }
            if (!got_bias || !got_runtime2) return -1;
            out_bias = netimpl->argTensor(bias_arg).clone();
            ops_consumed.insert(next_idx);
            mm_idx = stepProducer(matmul_out_arg);
        } else {
            mm_idx = next_idx;
        }

        if (mm_idx < 0 || !prog[mm_idx]) return -1;
        if (!dynamic_cast<MatMulLayer*>(prog[mm_idx].get())) return -1;
        if (prog[mm_idx]->blobs.empty()) return -1;
        if (prog[mm_idx]->inputs.size() != 1) return -1;
        out_W = prog[mm_idx]->blobs[0].clone();
        // Folded MatMul carries bias as a second blob (real_ndims_C >= 1).
        if (out_bias.empty() && prog[mm_idx]->blobs.size() >= 2)
            out_bias = prog[mm_idx]->blobs.back().clone();
        return mm_idx;
    }

    // 3 separate q/k/v projections, Q scaled, K^T at
    // the QK^T matmul, output reshaped+transposed back to (B,S,H*D).
    bool tryFuseClipAttention(const vector<Ptr<Layer>>& prog, int softmax_idx,
                              std::set<int>& removed_ops,
                              vector<std::pair<int, Ptr<Layer>>>& replacements)
    {
        if (softmax_idx < 0 || softmax_idx >= (int)prog.size() || !prog[softmax_idx])
            return false;
        if (removed_ops.count(softmax_idx)) return false;
        SoftmaxLayer* sm = dynamic_cast<SoftmaxLayer*>(prog[softmax_idx].get());
        if (!sm || sm->logSoftMax) return false;
        if (prog[softmax_idx]->inputs.size() != 1) return false;

        Arg sm_in = prog[softmax_idx]->inputs[0];
        auto it = producer_.find(sm_in.idx);
        if (it == producer_.end()) return false;
        int qk_matmul_idx = it->second;
        if (qk_matmul_idx < 0 || !prog[qk_matmul_idx]) return false;
        if (!dynamic_cast<MatMulLayer*>(prog[qk_matmul_idx].get())) return false;
        if (!prog[qk_matmul_idx]->blobs.empty()) return false;
        if (prog[qk_matmul_idx]->inputs.size() != 2) return false;
        Arg q_arg = prog[qk_matmul_idx]->inputs[0];
        Arg k_arg = prog[qk_matmul_idx]->inputs[1];

        Arg sm_out = prog[softmax_idx]->outputs[0];
        int av_matmul_idx = singleConsumer(sm_out);
        if (av_matmul_idx < 0 || !prog[av_matmul_idx]) return false;
        if (!dynamic_cast<MatMulLayer*>(prog[av_matmul_idx].get())) return false;
        if (!prog[av_matmul_idx]->blobs.empty()) return false;
        if (prog[av_matmul_idx]->inputs.size() != 2) return false;
        Arg v_arg;
        bool got_v = false;
        for (Arg in : prog[av_matmul_idx]->inputs) {
            if (in.idx == sm_out.idx) continue;
            v_arg = in; got_v = true;
        }
        if (!got_v) return false;

        std::set<int> consumed;
        Mat Wq, Wk, Wv, bq, bk, bv;
        int nh_q = 0, nh_k = 0, nh_v = 0;
        float q_scale = 1.f;
        float dummy = 1.f;
        int q_mm_idx = traceClipBranch(prog, q_arg, /*is_q=*/true, /*is_k=*/false,
                                       Wq, bq, nh_q, q_scale, consumed);
        if (q_mm_idx < 0) return false;
        int k_mm_idx = traceClipBranch(prog, k_arg, /*is_q=*/false, /*is_k=*/true,
                                       Wk, bk, nh_k, dummy, consumed);
        if (k_mm_idx < 0) return false;
        int v_mm_idx = traceClipBranch(prog, v_arg, /*is_q=*/false, /*is_k=*/false,
                                       Wv, bv, nh_v, dummy, consumed);
        if (v_mm_idx < 0) return false;

        if (q_mm_idx == k_mm_idx || k_mm_idx == v_mm_idx || q_mm_idx == v_mm_idx)
            return false;
        if (nh_q != nh_k || nh_k != nh_v) return false;
        const int num_heads = nh_q;
        if (q_scale == 0.f) return false;

        // All three projections share the same input.
        Arg shared_input = prog[q_mm_idx]->inputs[0];
        if (prog[k_mm_idx]->inputs[0].idx != shared_input.idx) return false;
        if (prog[v_mm_idx]->inputs[0].idx != shared_input.idx) return false;

        if (Wq.dims != 2 || Wk.dims != 2 || Wv.dims != 2) return false;
        if (Wq.size[0] != Wk.size[0] || Wk.size[0] != Wv.size[0]) return false;
        const int hidden_in = Wq.size[0];
        const int q_hidden = Wq.size[1];
        const int k_hidden = Wk.size[1];
        const int v_hidden = Wv.size[1];
        if (q_hidden != k_hidden || k_hidden != v_hidden) return false;
        if (q_hidden % num_heads != 0) return false;

        // Output chain: Reshape4D -> Transpose([0,2,1,3]) -> Reshape3D, the
        // boundary of the fused block.
        Arg av_out = prog[av_matmul_idx]->outputs[0];
        int out_r4d = singleConsumer(av_out);
        if (!isReshape(prog, out_r4d)) return false;
        Arg out_r4d_out = prog[out_r4d]->outputs[0];
        int out_t4d = singleConsumer(out_r4d_out);
        if (!isTranspose(prog, out_t4d)) return false;
        TransposeLayer* out_tr =
            dynamic_cast<TransposeLayer*>(prog[out_t4d].get());
        if (!out_tr || out_tr->perm.size() != 4) return false;
        if (out_tr->perm[0] != 0 || out_tr->perm[1] != 2 ||
            out_tr->perm[2] != 1 || out_tr->perm[3] != 3) return false;
        Arg out_t4d_out = prog[out_t4d]->outputs[0];
        int out_r3d = singleConsumer(out_t4d_out);
        if (!isReshape(prog, out_r3d)) return false;

        // Combined [Q|K|V] weight along the output dim.
        int total_hidden = q_hidden + k_hidden + v_hidden;
        int wshape[] = {hidden_in, total_hidden};
        Mat W_qkv(2, wshape, CV_32F);
        for (int r = 0; r < hidden_in; r++) {
            float* dst = W_qkv.ptr<float>(r);
            std::memcpy(dst,                       Wq.ptr<float>(r), q_hidden * sizeof(float));
            std::memcpy(dst + q_hidden,            Wk.ptr<float>(r), k_hidden * sizeof(float));
            std::memcpy(dst + q_hidden + k_hidden, Wv.ptr<float>(r), v_hidden * sizeof(float));
        }
        Mat bias_qkv;
        if (!bq.empty() && !bk.empty() && !bv.empty()) {
            const int bias_total = q_hidden + k_hidden + v_hidden;
            bias_qkv.create(1, &bias_total, CV_32F);
            float* dst = bias_qkv.ptr<float>();
            std::memcpy(dst,                       bq.ptr<float>(), q_hidden * sizeof(float));
            std::memcpy(dst + q_hidden,            bk.ptr<float>(), k_hidden * sizeof(float));
            std::memcpy(dst + q_hidden + k_hidden, bv.ptr<float>(), v_hidden * sizeof(float));
        }

        // scale (pre-softmax divisor) = 1/q_scale, since Q was pre-multiplied.
        const float param_scale = 1.f / q_scale;

        LayerParams attn_params;
        attn_params.name = prog[q_mm_idx]->name + "_fused_attention";
        attn_params.type = "Attention";
        attn_params.set("num_heads", num_heads);
        int qkv_sizes[3] = { q_hidden, k_hidden, v_hidden };
        attn_params.set("qkv_hidden_sizes", DictValue::arrayInt(qkv_sizes, 3));
        attn_params.set("scale", param_scale);
        attn_params.set("output_ndims", 3);
        attn_params.blobs.push_back(W_qkv);
        if (!bias_qkv.empty()) attn_params.blobs.push_back(bias_qkv);

        Ptr<Layer> attn_layer =
            LayerFactory::createLayerInstance(attn_params.type, attn_params);
        if (!attn_layer) return false;
        attn_layer->inputs  = { shared_input };
        attn_layer->outputs = prog[out_r3d]->outputs;
        attn_layer->netimpl = netimpl;

        std::set<int> to_remove = {
            q_mm_idx, k_mm_idx, v_mm_idx,
            qk_matmul_idx, softmax_idx, av_matmul_idx,
            out_r4d, out_t4d, out_r3d
        };
        for (int op : consumed) to_remove.insert(op);

        for (int op : to_remove) removed_ops.insert(op);
        int insert_pos = *std::min_element(to_remove.begin(), to_remove.end());
        replacements.push_back({insert_pos, attn_layer});
        return true;
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

        bool modified = false;
        std::set<int> removed_ops;

        // Pass 1: combined-QKV blocks.
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i] || removed_ops.count((int)i)) continue;
            if (tryFuseCombinedQKV(prog, (int)i, removed_ops, attention_replacements_))
                modified = true;
        }

        // Pass 2: the original 3-separate-projection path.
        std::map<int, vector<int>> qkv_candidates;
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i] || removed_ops.count((int)i)) continue;
            if (!isProjCandidate(prog[i])) continue;
            Arg inp = prog[i]->inputs[0];
            qkv_candidates[inp.idx].push_back((int)i);
        }

        for (auto& candidate : qkv_candidates) {
            auto matmul_indices = candidate.second;
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

        // Pass 3: anchored on each Softmax.
        for (size_t i = 0; i < nops; i++) {
            if (!prog[i] || removed_ops.count((int)i)) continue;
            if (prog[i]->type != "Softmax") continue;
            if (tryFuseClipAttention(prog, (int)i, removed_ops,
                                     attention_replacements_))
                modified = true;
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
