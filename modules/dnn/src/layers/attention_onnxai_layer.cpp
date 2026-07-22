// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "cpu_kernels/fast_gemm.hpp"
#include "cpu_kernels/fast_attn.hpp"

#include "layers_common.hpp"
#include "../net_impl.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

// Operator spec: https://onnx.ai/onnx/operators/onnx__Attention.html#attention-23
class AttentionOnnxAiLayerImpl CV_FINAL : public AttentionOnnxAiLayer {
 public:
     AttentionOnnxAiLayerImpl(const LayerParams &params) {
        setParamsFrom(params);
        is_causal = params.get<bool>("is_causal", false);
        kv_num_heads = params.get<int>("kv_num_heads", 0);
        q_num_heads = params.get<int>("q_num_heads", 0);
        qk_matmul_output_mode = params.get<int>("qk_matmul_output_mode", 0);
        scale = params.get<float>("scale", 1.0f );
        is_scale_set = params.has("scale");
        softcap = params.get<float>("softcap", 0.f);
        softmax_precision = params.get<int>("softmax_precision", 0);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    // ONNX Attention optional-input layout (Q, K, V are always inputs 0..2):
    //   inputs.size() == 4 : Q, K, V, attn_mask
    //   inputs.size() == 5 : Q, K, V, past_key, past_value          (no mask)
    //   inputs.size() == 6 : Q, K, V, attn_mask, past_key, past_value
    static bool hasMaskInput(size_t n_inputs) { return n_inputs == 4 || n_inputs == 6; }
    static int  pastKeyIdx(size_t n_inputs)   { return n_inputs == 5 ? 3 : n_inputs == 6 ? 4 : -1; }
    static int  pastValueIdx(size_t n_inputs) { return n_inputs == 5 ? 4 : n_inputs == 6 ? 5 : -1; }

    virtual void getTypes(const std::vector<MatType>&inputs,
                     const int requiredOutputs,
                     const int requiredInternals,
                     std::vector<MatType>&outputs,
                     std::vector<MatType>&internals) const CV_OVERRIDE {
        // type checks
        CV_CheckTrue(inputs.size() >= 3, "At least three inputs (query, key, value) are required");

        for (int i = 0; i < 3; i++) {
            CV_CheckType(inputs[i], inputs[i] == CV_16F || inputs[i] == CV_32F, "");
        }

        CV_CheckType(inputs[0], inputs[0] == inputs[1] && inputs[0] == inputs[2], "");

        if (hasMaskInput(inputs.size())) {
            CV_CheckType(inputs[3], inputs[3] == CV_8U || inputs[3] == CV_8S ||
                         inputs[3] == CV_16U || inputs[3] == CV_16S ||
                         inputs[3] == CV_32S || inputs[3] == CV_64S ||
                         inputs[3] == CV_64U || inputs[3] == CV_Bool ||
                         inputs[3] == inputs[2], ""); // attention_mask
        }

        const int past_k_idx = pastKeyIdx(inputs.size());
        if (past_k_idx >= 0) {
            CV_CheckType(inputs[past_k_idx],   inputs[past_k_idx]   == inputs[0], ""); // past_key
            CV_CheckType(inputs[past_k_idx+1], inputs[past_k_idx+1] == inputs[0], ""); // past_value
        }

        outputs.assign(requiredOutputs, inputs[0]);

        // internals:
        internals.clear();

        // 1. attention_prob
        internals.push_back(inputs[0]);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        CV_CheckTrue(inputs.size() >= 3, "At least three inputs (query, key, value) are required");
        CV_CheckTrue(inputs[0].dims == inputs[1].dims &&
                     inputs[0].dims == inputs[2].dims,
                     "Query, key and value must have the same number of dimensions");

        const int input_dims = inputs[0].dims;
        CV_CheckTrue(
            input_dims == 4 ||  (q_num_heads > 0 && kv_num_heads > 0 && input_dims == 3),
            "Input dimensions must be 4D or 3D (in the latter case, q_num_heads and kv_num_heads must be set)"
        );

        const int seq_dim = input_dims - 2;
        const int batch_size = inputs[0][0];
        const int seq_len_q = inputs[0][seq_dim];
        int seq_len_k = inputs[1][seq_dim];
        int seq_len_v = inputs[2][seq_dim];

        int past_seq_kv = 0;
        Net::Impl* netimpl = getNetImpl(const_cast<AttentionOnnxAiLayerImpl*>(this));
        if (netimpl && netimpl->useKVCache) {
            KVCacheManager& kvCacheManager = netimpl->kvCacheManager;
            if (kvCacheManager.isInitialized) {
                auto it_k = kvCacheManager.kData.find(name);
                CV_Assert(it_k != kvCacheManager.kData.end());
                if (it_k != kvCacheManager.kData.end())
                    seq_len_k += it_k->second.getNumTokens();

                auto it_v = kvCacheManager.vData.find(name);
                CV_Assert(it_v != kvCacheManager.vData.end());
                if (it_v != kvCacheManager.vData.end())
                    seq_len_v += it_v->second.getNumTokens();
            }
        } else {
            const int past_k_idx = pastKeyIdx(inputs.size());
            if (past_k_idx >= 0) {
                const int past_v_idx = pastValueIdx(inputs.size());
                CV_CheckTrue(past_v_idx >= 0 && past_v_idx < (int)inputs.size(),
                             "past_key and past_value must be provided as a pair");

                // past_key/past_value are always 4D [batch, nhkv, past_seq, head] (even for 3D Q/K/V).
                const MatShape& pk = inputs[past_k_idx];
                const MatShape& pv = inputs[past_v_idx];
                CV_CheckEQ(pk.dims, 4, "past_key must be 4D [batch, nhkv, past_seq, head]");
                CV_CheckEQ(pv.dims, 4, "past_value must be 4D [batch, nhkv, past_seq, head]");
                CV_CheckEQ(pk[0], batch_size, "past_key batch dimension must match query batch");
                CV_CheckEQ(pv[0], batch_size, "past_value batch dimension must match query batch");

                past_seq_kv = pk[pk.dims - 2];
                seq_len_k += past_seq_kv;
                seq_len_v += past_seq_kv;
            }
        }

        const int q_hn = input_dims == 4 ?
            inputs[0][1] : q_num_heads;
        const int kv_hn =input_dims == 4 ?
            inputs[1][1] : kv_num_heads;

        CV_CheckTrue(seq_len_v == seq_len_k,
                     "Key and value sequence lengths must be equal");
        const int nhq = input_dims == 4 ? inputs[0][1] : q_num_heads;
        const int total_seq_kv = seq_len_k;

        CV_CheckTrue(q_hn % kv_hn == 0,
                     "q_num_heads must be divisible by kv_num_heads");

        if (input_dims == 3)
        {
            CV_CheckTrue(kv_hn > 0,
                         "For 3D input, kv_num_heads must be greater than 0 (this normally means that kv_num_heads is not set)");
            CV_CheckTrue(q_hn > 0,
                         "For 3D input, q_num_heads must be greater than 0 (this normally means that q_num_heads is not set)");

            int v_head_size = inputs[2][2] / kv_hn;
            MatShape output_shape{batch_size, seq_len_q, v_head_size * q_num_heads};
            outputs.push_back(output_shape);
        }
        else
        {
            int v_head_size = inputs[2][3];
            MatShape output_shape{batch_size, nhq, seq_len_q, v_head_size};
            outputs.push_back(output_shape);
        }

        const int qk_head_size = input_dims == 4 ? inputs[1][3] : inputs[1][2] / kv_hn;
        const int v_head_size  = input_dims == 4 ? inputs[2][3] : inputs[2][2] / kv_hn;
        if (requiredOutputs > 1)
            outputs.push_back(MatShape{batch_size, kv_hn, total_seq_kv, qk_head_size});  // present_key
        if (requiredOutputs > 2)
            outputs.push_back(MatShape{batch_size, kv_hn, total_seq_kv, v_head_size});   // present_value
        if (requiredOutputs > 3)
            outputs.push_back(MatShape{batch_size, nhq, seq_len_q, total_seq_kv});       // qk_matmul_output

        MatShape attention_prob_shape{batch_size , nhq, seq_len_q, total_seq_kv};
        internals.push_back(attention_prob_shape);

        return false;
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        const int input_dims = inputs[0].dims;
        int64 B = inputs[0][0];
        int64 Sq = inputs[0][input_dims - 2];
        int64 Skv = inputs[1][input_dims - 2];
        int64 nhq = input_dims == 4 ? inputs[0][1] : q_num_heads;
        int64 qk_head = input_dims == 4 ? inputs[0][3] : inputs[0][2] / nhq;
        int64 nhkv = input_dims == 4 ? inputs[1][1] : kv_num_heads;
        int64 v_head = input_dims == 4 ? inputs[2][3] : inputs[2][2] / nhkv;

        // QK^T: batch * nhq * (2 * Sq * Skv * qk_head)
        int64 flops = B * nhq * CV_BIG_INT(2) * Sq * Skv * qk_head;
        // Softmax: ~4 ops per element
        flops += B * nhq * 4 * Sq * Skv;
        // Attention * V: batch * nhq * (2 * Sq * v_head * Skv)
        flops += B * nhq * CV_BIG_INT(2) * Sq * v_head * Skv;
        return flops;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        opt.init();

        Net::Impl* netimpl = getNetImpl(this);
        bool with_kv_cache = false;

        if (netimpl && netimpl->useKVCache) {
            with_kv_cache = netimpl->kvCacheManager.isInitialized;
        }

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);
        Mat &attention_prob = internals[internals.size() - 1];

        const int input_dims = inputs[0].dims;
        const int seq_dim = input_dims - 2;

        const int batch_size = inputs[0].size[0];

        const int seq_len_q = inputs[0].size[seq_dim];
        int seq_len_kv = inputs[1].size[seq_dim];

        const int nhq = input_dims == 3 ?
                        q_num_heads :
                        inputs[0].size[1];

        const int qk_head_size = input_dims == 3 ?
                                 inputs[0].size[2] / nhq :
                                 inputs[0].size[3];

        const int nhkv = input_dims == 3 ?
                        kv_num_heads :
                        inputs[1].size[1];

        const int v_head_size  = input_dims == 3 ?
                                inputs[2].size[2] / nhkv :
                                inputs[2].size[3];

        const bool has_mask_input = hasMaskInput(inputs.size());
        const Mat mask_mat = has_mask_input ? inputs[3] : Mat();

        scale = is_scale_set ? scale : 1.0f / std::sqrt(static_cast<float>(qk_head_size));

        std::vector<size_t> _q_offsets, _k_offsets, _v_offsets,
                            _a_offsets, _o_offsets;

        int past_seq_kv = 0;

        if (with_kv_cache){
            KVCacheManager& kvCacheManager = netimpl->kvCacheManager;

            auto it_k = kvCacheManager.kData.find(name);
            CV_Assert(it_k != kvCacheManager.kData.end());
            KCache&kData = it_k->second;

            kData.grow(inputs[1]);

            const std::vector<Mat>& kCachePages = kData.getPages();
            seq_len_kv = kData.getNumTokens();

            pagedAttnQKGemm(
                inputs[0], kCachePages, attention_prob,
                seq_len_q, nhq, nhkv, kData.getPageSize(),
                qk_head_size, seq_len_kv,
                scale, opt
            );

            past_seq_kv = seq_len_kv - seq_len_q;

            fused_softmax_softcap_mask(
                attention_prob, mask_mat,
                softcap, softcap > 0.f, 9.f, -FLT_MAX,
                has_mask_input, is_causal, past_seq_kv
            );

            auto it_v = kvCacheManager.vData.find(name);
            CV_Assert(it_v != kvCacheManager.vData.end());
            VCache& vData = it_v->second;

            vData.grow(inputs[2]);
            seq_len_kv = vData.getNumTokens();

            pagedAttnAVGemm(
                attention_prob, vData.getPages(), outputs[0],
                seq_len_q, nhq, nhkv, vData.getPageSize() , v_head_size, seq_len_kv,
                opt
            );
            return;
        }

        // Standard (non-paged) path, with optional past_key/past_value graph inputs
        const int past_k_idx = pastKeyIdx(inputs.size());
        const int past_v_idx = pastValueIdx(inputs.size());
        const bool use_past  = past_k_idx >= 0;
        // past_key/past_value are always 4D [batch, nhkv, past_seq, head]; use their own rank.
        past_seq_kv = use_past ? inputs[past_k_idx].size[inputs[past_k_idx].dims - 2] : 0;
        const int total_seq_kv = past_seq_kv + seq_len_kv;

        Mat K_eff, V_eff;
        if (use_past && past_seq_kv > 0) {
            if (input_dims == 4) {
                // 4D [batch, nhkv, seq, head]: concat on axis 2.
                int dims_k[4] = {batch_size, nhkv, total_seq_kv, qk_head_size};
                int dims_v[4] = {batch_size, nhkv, total_seq_kv, v_head_size};
                K_eff.create(4, dims_k, CV_32F);
                V_eff.create(4, dims_v, CV_32F);
                const std::vector<Range> past_r{Range::all(), Range::all(), Range(0, past_seq_kv), Range::all()};
                const std::vector<Range> cur_r {Range::all(), Range::all(), Range(past_seq_kv, total_seq_kv), Range::all()};
                inputs[past_k_idx].copyTo(K_eff(past_r)); inputs[1].copyTo(K_eff(cur_r));
                inputs[past_v_idx].copyTo(V_eff(past_r)); inputs[2].copyTo(V_eff(cur_r));
            } else {
                const int k_elem = nhkv * qk_head_size;
                const int v_elem = nhkv * v_head_size;
                int dims_k[3] = {batch_size, total_seq_kv, k_elem};
                int dims_v[3] = {batch_size, total_seq_kv, v_elem};
                int pk_sz[3] = {batch_size, past_seq_kv, k_elem};
                int pv_sz[3] = {batch_size, past_seq_kv, v_elem};
                K_eff.create(3, dims_k, CV_32F);
                V_eff.create(3, dims_v, CV_32F);
                const std::vector<Range> past_r{Range::all(), Range(0, past_seq_kv), Range::all()};
                const std::vector<Range> cur_r {Range::all(), Range(past_seq_kv, total_seq_kv), Range::all()};
                Mat pk, pv;
                cv::transposeND(inputs[past_k_idx], {0, 2, 1, 3}, pk);
                cv::transposeND(inputs[past_v_idx], {0, 2, 1, 3}, pv);
                pk.reshape(1, 3, pk_sz).copyTo(K_eff(past_r)); inputs[1].copyTo(K_eff(cur_r));
                pv.reshape(1, 3, pv_sz).copyTo(V_eff(past_r)); inputs[2].copyTo(V_eff(cur_r));
            }
        } else {
            K_eff = inputs[1];
            V_eff = inputs[2];
        }

        const auto* Q = inputs[0].ptr<const float>();
        const auto* K = K_eff.ptr<const float>();
        const auto* V = V_eff.ptr<const float>();

        const int num_gq_groups = nhq / nhkv;
        const auto seq_len_square = seq_len_q * total_seq_kv;

        _q_offsets.resize(nhq * batch_size);
        _k_offsets.resize(nhq * batch_size);
        _a_offsets.resize(nhq * batch_size);
        _v_offsets.resize(nhq * batch_size);
        _o_offsets.resize(nhq * batch_size);

        for (int b = 0; b < batch_size; b++)
            for (int n = 0; n < nhq; n++){
                _q_offsets[b * nhq + n] =
                    b * seq_len_q * qk_head_size * nhq +
                    (input_dims == 3 ? n * qk_head_size : n * qk_head_size * seq_len_q);
                _k_offsets[b * nhq + n] =
                    b * total_seq_kv * qk_head_size * nhkv +
                    (n / num_gq_groups * qk_head_size) * (input_dims == 3 ? 1 : total_seq_kv);
                _v_offsets[b * nhq + n] =
                    b * total_seq_kv * v_head_size * nhkv +
                    (n / num_gq_groups * v_head_size) * (input_dims == 3 ? 1 : total_seq_kv);
                _a_offsets[b * nhq + n] =
                    b * seq_len_square * nhq +
                    n * seq_len_square;
                _o_offsets[b * nhq + n] =
                    b * seq_len_q * v_head_size * nhq +
                    (input_dims == 3 ? n * v_head_size : n * v_head_size * seq_len_q);
            }

        const int ldq0 = input_dims == 3 ? qk_head_size * nhq : qk_head_size;
        const int ldk0 = input_dims == 3 ? qk_head_size * nhkv : qk_head_size;

        fastGemmBatch(
            batch_size * nhq,
            _q_offsets.data(), _k_offsets.data(), _a_offsets.data(),
            seq_len_q, total_seq_kv, qk_head_size , scale,
            Q, ldq0, 1,
            K, 1, ldk0,
            0.f,
            attention_prob.ptr<float>(), total_seq_kv,
            opt
        );

        // qk_matmul_output (optional 4th output), per qk_matmul_output_mode:
        //   0 = raw scaled QK^T,  1 = + attention bias,  2 = + softcap,  3 = post-softmax.
        const bool want_qk = outputs.size() > 3 && !outputs[3].empty();
        if (want_qk && qk_matmul_output_mode == 0) {
            attention_prob.copyTo(outputs[3]);
        } else if (want_qk && (qk_matmul_output_mode == 1 || qk_matmul_output_mode == 2)) {
            attention_prob.copyTo(outputs[3]);
            fused_softmax_softcap_mask(
                outputs[3], mask_mat,
                softcap, (qk_matmul_output_mode == 2) && (softcap > 0.f), 9.f,
                -std::numeric_limits<float>::infinity(),
                has_mask_input, is_causal, past_seq_kv, /*do_softmax=*/false
            );
        }

        fused_softmax_softcap_mask(
            attention_prob, mask_mat,
            softcap, softcap > 0.f, 9.f, -FLT_MAX,
            has_mask_input, is_causal, past_seq_kv
        );

        if (want_qk && qk_matmul_output_mode == 3)
            attention_prob.copyTo(outputs[3]);

        const int ldv0 = input_dims == 3 ? v_head_size * nhkv : v_head_size;
        const int ldout = input_dims == 3 ? v_head_size * nhq : v_head_size;

        fastGemmBatch(
            batch_size * nhq,
            _a_offsets.data(), _v_offsets.data(), _o_offsets.data(),
            seq_len_q, v_head_size, total_seq_kv, 1.f,
            attention_prob.ptr<float>(), total_seq_kv, 1,
            V, ldv0, 1,
            0.f,
            outputs[0].ptr<float>(), ldout,
            opt
        );

        auto writePresent = [&](Mat& out, const Mat& eff, int head_size) {
            if (out.empty()) return;
            if (input_dims == 4) {
                eff.reshape(1, out.dims, out.size.p).copyTo(out);
            } else {
                int sz[4] = {batch_size, total_seq_kv, nhkv, head_size};
                cv::transposeND(eff.reshape(1, 4, sz), {0, 2, 1, 3}, out);
            }
        };
        if (outputs.size() > 1) writePresent(outputs[1], K_eff, qk_head_size);
        if (outputs.size() > 2) writePresent(outputs[2], V_eff, v_head_size);
    }

 private:
    bool is_causal;
    int q_num_heads;
    int qk_matmul_output_mode;
    float scale;
    bool is_scale_set = false;
    float softcap;
    int softmax_precision;
    FastGemmOpt opt;
};

Ptr<AttentionOnnxAiLayer> AttentionOnnxAiLayer::create(const LayerParams &params) {
    return makePtr<AttentionOnnxAiLayerImpl>(params);
}

}} // cv::dnn
