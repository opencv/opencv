// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "cpu_kernels/fast_gemm.hpp"
#include "cpu_kernels/fast_attn.hpp"

#include "layers_common.hpp"
#include "../net_impl.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include <cmath>

namespace cv { namespace dnn {

// Operator spec: https://onnx.ai/onnx/operators/onnx__Attention.html#attention-23
// Also backs com.microsoft MultiHeadAttention and GroupQueryAttention.
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

        local_window_size = params.get<int>("local_window_size", -1);
        do_rotary = params.get<int>("do_rotary", 0) != 0;
        rotary_interleaved = params.get<int>("rotary_interleaved", 0) != 0;
        shared_kv_buffer = params.get<int>("shared_kv_buffer", 0) != 0;

        has_attn_mask_attr = params.get<int>("has_attn_mask", -1);
        has_past_attr      = params.get<int>("has_past", -1);
        has_seqlens_attr   = params.get<int>("has_seqlens_k", 0) != 0;
        has_rotary_attr    = params.get<int>("has_rotary_cache", 0) != 0;

        // The paged cache cannot express any of these; such a node reads history from the graph.
        paged_cache_supported = !(has_seqlens_attr || has_rotary_attr ||
                                  do_rotary || shared_kv_buffer || local_window_size > 0);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    // After Q,K,V: attn_mask, past pair, seqlens_k, cos+sin. None declared = legacy 4/5/6 count.
    struct InputMap { int mask = -1, past_k = -1, past_v = -1, seqlens = -1, cos = -1, sin = -1; };

    InputMap mapInputs(size_t n_inputs) const {
        InputMap m;
        bool with_mask, with_past;
        if (has_attn_mask_attr < 0 && has_past_attr < 0) {
            with_mask = (n_inputs == 4 || n_inputs == 6);
            with_past = (n_inputs == 5 || n_inputs == 6);
        } else {
            with_mask = has_attn_mask_attr > 0;
            with_past = has_past_attr > 0;
        }
        int i = 3;
        if (with_mask) m.mask = i++;
        if (with_past) { m.past_k = i++; m.past_v = i++; }
        if (has_seqlens_attr) m.seqlens = i++;
        if (has_rotary_attr) { m.cos = i++; m.sin = i++; }
        if (i != (int)n_inputs)
            CV_Error(Error::StsBadArg,
                     cv::format("AttentionOnnxAi: got %d inputs but the declared optional inputs "
                                "add up to %d (query, key, value%s%s%s%s)",
                                (int)n_inputs, i,
                                with_mask ? ", attn_mask" : "",
                                with_past ? ", past_key, past_value" : "",
                                has_seqlens_attr ? ", seqlens_k" : "",
                                has_rotary_attr ? ", cos_cache, sin_cache" : ""));
        return m;
    }

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

        const InputMap map = mapInputs(inputs.size());

        if (map.mask >= 0) {
            CV_CheckType(inputs[map.mask], inputs[map.mask] == CV_8U || inputs[map.mask] == CV_8S ||
                         inputs[map.mask] == CV_16U || inputs[map.mask] == CV_16S ||
                         inputs[map.mask] == CV_32S || inputs[map.mask] == CV_64S ||
                         inputs[map.mask] == CV_64U || inputs[map.mask] == CV_Bool ||
                         inputs[map.mask] == inputs[2], ""); // attention_mask
        }

        if (map.past_k >= 0) {
            CV_CheckType(inputs[map.past_k], inputs[map.past_k] == inputs[0], ""); // past_key
            CV_CheckType(inputs[map.past_v], inputs[map.past_v] == inputs[0], ""); // past_value
        }

        if (map.seqlens >= 0) {
            CV_CheckType(inputs[map.seqlens],
                         inputs[map.seqlens] == CV_32S || inputs[map.seqlens] == CV_64S,
                         "seqlens_k must be int32 or int64");
            // forward_fallback has no half path here; it would leave outputs untouched, silently.
            CV_CheckType(inputs[0], inputs[0] == CV_32F,
                         "GroupQueryAttention inputs must be CV_32F");
        }

        if (map.cos >= 0) {
            CV_CheckType(inputs[map.cos], inputs[map.cos] == CV_32F, "cos_cache must be float32");
            CV_CheckType(inputs[map.sin], inputs[map.sin] == CV_32F, "sin_cache must be float32");
        }

        outputs.assign(requiredOutputs, inputs[0]);

        // internals:
        internals.clear();

        // 1. rotated query and key, when the node rotates
        if (do_rotary) {
            internals.push_back(inputs[0]);
            internals.push_back(inputs[1]);
        }

        // 2. attention_prob
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

        const InputMap map = mapInputs(inputs.size());

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

        if (do_rotary)
            CV_CheckTrue(map.cos >= 0, "do_rotary needs cos_cache and sin_cache");

        // No cross-attention: the rotary and shared-buffer passes read key/value seq_len_q rows.
        if (map.seqlens >= 0) {
            CV_CheckEQ(seq_len_k, seq_len_q,
                       "key/value sequence length must match the query when seqlens_k is given");
            // These size every buffer below, so a bad value here becomes an overrun later.
            const int head_size = input_dims == 4 ? inputs[0][3] : inputs[0][2] / q_num_heads;
            CV_CheckGT(batch_size, 0, "batch size must be positive");
            CV_CheckGT(seq_len_q, 0, "query sequence length must be positive");
            CV_CheckGT(head_size, 0, "head size must be positive");
        }

        int past_seq_kv = 0;
        Net::Impl* netimpl = getNetImpl(const_cast<AttentionOnnxAiLayerImpl*>(this));
        if (netimpl && netimpl->useKVCache && paged_cache_supported) {
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
            if (map.past_k >= 0) {
                // past_key/past_value are always 4D [batch, nhkv, past_seq, head] (even for 3D Q/K/V).
                const MatShape& pk = inputs[map.past_k];
                const MatShape& pv = inputs[map.past_v];
                CV_CheckEQ(pk.dims, 4, "past_key must be 4D [batch, nhkv, past_seq, head]");
                CV_CheckEQ(pv.dims, 4, "past_value must be 4D [batch, nhkv, past_seq, head]");
                CV_CheckEQ(pk[0], batch_size, "past_key batch dimension must match query batch");
                CV_CheckEQ(pv[0], batch_size, "past_value batch dimension must match query batch");

                if (map.seqlens >= 0) {
                    // A disagreeing head count or size mis-strides every offset built from it.
                    const int kvh = input_dims == 4 ? inputs[1][1] : kv_num_heads;
                    CV_CheckEQ(pk[1], kvh, "past_key head count must be kv_num_heads");
                    CV_CheckEQ(pv[1], kvh, "past_value head count must be kv_num_heads");
                    CV_CheckEQ(pk[3], input_dims == 4 ? inputs[1][3] : inputs[1][2] / kvh,
                               "past_key head size must match the key");
                }

                past_seq_kv = pk[pk.dims - 2];
                CV_CheckGE(past_seq_kv, 0, "past_key sequence length must not be negative");
                if (shared_kv_buffer) {
                    // Rewritten in place, so the span is the buffer, not buffer + new tokens.
                    CV_CheckGE(past_seq_kv, seq_len_q,
                               "shared KV buffer is shorter than the query");
                    seq_len_k = past_seq_kv;
                    seq_len_v = past_seq_kv;
                } else {
                    seq_len_k += past_seq_kv;
                    seq_len_v += past_seq_kv;
                }
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

        // Order matches getTypes, and forward() reads attention_prob as the last internal.
        if (do_rotary) {
            internals.push_back(inputs[0]);  // rotated query
            internals.push_back(inputs[1]);  // rotated key
        }
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

    // Pass rotary_embedding_dim explicitly; left at 0 it overruns a narrower cos_cache.
    void ensureRope(int rotaryDim, int nhq, int nhkv) {
        if (ropeQ && ropeRotaryDim == rotaryDim)
            return;

        LayerParams lpQ;
        lpQ.set("num_heads", nhq);
        lpQ.set("interleaved", rotary_interleaved ? 1 : 0);
        lpQ.set("rotary_embedding_dim", rotaryDim);
        ropeQ = RotaryEmbeddingLayer::create(lpQ);

        LayerParams lpK;
        lpK.set("num_heads", nhkv);
        lpK.set("interleaved", rotary_interleaved ? 1 : 0);
        lpK.set("rotary_embedding_dim", rotaryDim);
        ropeK = RotaryEmbeddingLayer::create(lpK);

        ropeRotaryDim = rotaryDim;
    }

    static void applyRotary(const Ptr<RotaryEmbeddingLayer>& rope, const Mat& src, Mat& dst,
                            const Mat& cosCache, const Mat& sinCache, const Mat& positionIds) {
        const int B = positionIds.size[0], S = positionIds.size[1];
        const int dhalf = cosCache.size[cosCache.dims - 1];
        std::vector<Mat> rope_in{src, cosCache, sinCache, positionIds};
        std::vector<Mat> rope_out{dst};
        std::vector<Mat> rope_internals{
            Mat(std::vector<int>{B, S, dhalf}, CV_32F),
            Mat(std::vector<int>{B, S, dhalf}, CV_32F),
        };
        rope->forward(rope_in, rope_out, rope_internals);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        opt.init();

        Net::Impl* netimpl = getNetImpl(this);
        bool with_kv_cache = false;

        if (netimpl && netimpl->useKVCache && paged_cache_supported) {
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

        const InputMap map = mapInputs(inputs.size());

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

        const bool has_mask_input = map.mask >= 0;
        const Mat mask_mat = has_mask_input ? inputs[map.mask] : Mat();

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

            const std::vector<Mat> kCachePages = kData.getActivePages();
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
                attention_prob, vData.getActivePages(), outputs[0],
                seq_len_q, nhq, nhkv, vData.getPageSize() , v_head_size, seq_len_kv,
                opt
            );
            return;
        }

        // Standard (non-paged) path, with optional past_key/past_value graph inputs
        const bool use_past  = map.past_k >= 0;
        // past_key/past_value are always 4D [batch, nhkv, past_seq, head]; use their own rank.
        past_seq_kv = use_past ? inputs[map.past_k].size[inputs[map.past_k].dims - 2] : 0;
        const int total_seq_kv = shared_kv_buffer ? past_seq_kv : past_seq_kv + seq_len_kv;

        // validLen[b] = live length after this step; pastLen[b] = where its first token lands.
        const bool has_seqlens = map.seqlens >= 0;
        std::vector<int> validLen(batch_size, past_seq_kv + seq_len_q),
                         pastLen(batch_size, past_seq_kv);
        if (has_seqlens) {
            const Mat& sk = inputs[map.seqlens];
            CV_CheckEQ((int)sk.total(), batch_size, "seqlens_k must hold one entry per batch");
            for (int b = 0; b < batch_size; ++b) {
                const int64_t v = sk.depth() == CV_32S ? (int64_t)sk.ptr<int32_t>()[b]
                                                       : sk.ptr<int64_t>()[b];
                validLen[b] = static_cast<int>(v) + 1;
                CV_CheckLE(validLen[b], total_seq_kv, "seqlens_k exceeds the KV buffer length");
                pastLen[b] = validLen[b] - seq_len_q;
                CV_CheckGE(pastLen[b], 0, "seqlens_k is smaller than the query length");
                if (!shared_kv_buffer)
                    CV_CheckEQ(pastLen[b], past_seq_kv,
                        "seqlens_k disagrees with past_key's length for a growing cache; a "
                        "preallocated buffer needs a static past_key dimension");
            }
        }

        // Rotate before the cache write, so stored keys carry their absolute position once.
        Mat Q_src = inputs[0], K_src = inputs[1];
        if (do_rotary) {
            const Mat& cosCache = inputs[map.cos];
            const Mat& sinCache = inputs[map.sin];
            // Each cos_cache row holds one value per rotated pair, so it covers 2*width channels.
            const int rotaryDim = 2 * static_cast<int>(cosCache.size[cosCache.dims - 1]);
            CV_CheckLE(rotaryDim, qk_head_size, "cos_cache is wider than half the head size");

            Mat positionIds(std::vector<int>{batch_size, seq_len_q}, CV_MAKETYPE(CV_64S, 1));
            int64_t* posPtr = positionIds.ptr<int64_t>();
            for (int b = 0; b < batch_size; ++b)
                for (int i = 0; i < seq_len_q; ++i)
                    posPtr[(size_t)b * seq_len_q + i] = std::max(pastLen[b] + i, 0);

            ensureRope(rotaryDim, nhq, nhkv);
            applyRotary(ropeQ, inputs[0], internals[0], cosCache, sinCache, positionIds);
            applyRotary(ropeK, inputs[1], internals[1], cosCache, sinCache, positionIds);
            Q_src = internals[0];
            K_src = internals[1];
        }

        Mat K_eff, V_eff;
        if (use_past && shared_kv_buffer) {
            // Carry the live prefix, write this step at pastLen[b], zero the rest (matches ORT).
            buildSharedKVBuffer(inputs[map.past_k], K_src, K_eff,
                                batch_size, nhkv, total_seq_kv, seq_len_q, qk_head_size,
                                input_dims, validLen, pastLen);
            buildSharedKVBuffer(inputs[map.past_v], inputs[2], V_eff,
                                batch_size, nhkv, total_seq_kv, seq_len_q, v_head_size,
                                input_dims, validLen, pastLen);
        } else if (use_past && past_seq_kv > 0) {
            if (input_dims == 4) {
                // 4D [batch, nhkv, seq, head]: concat on axis 2.
                int dims_k[4] = {batch_size, nhkv, total_seq_kv, qk_head_size};
                int dims_v[4] = {batch_size, nhkv, total_seq_kv, v_head_size};
                K_eff.create(4, dims_k, CV_32F);
                V_eff.create(4, dims_v, CV_32F);
                const std::vector<Range> past_r{Range::all(), Range::all(), Range(0, past_seq_kv), Range::all()};
                const std::vector<Range> cur_r {Range::all(), Range::all(), Range(past_seq_kv, total_seq_kv), Range::all()};
                inputs[map.past_k].copyTo(K_eff(past_r)); K_src.copyTo(K_eff(cur_r));
                inputs[map.past_v].copyTo(V_eff(past_r)); inputs[2].copyTo(V_eff(cur_r));
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
                cv::transposeND(inputs[map.past_k], {0, 2, 1, 3}, pk);
                cv::transposeND(inputs[map.past_v], {0, 2, 1, 3}, pv);
                pk.reshape(1, 3, pk_sz).copyTo(K_eff(past_r)); K_src.copyTo(K_eff(cur_r));
                pv.reshape(1, 3, pv_sz).copyTo(V_eff(past_r)); inputs[2].copyTo(V_eff(cur_r));
            }
        } else {
            K_eff = K_src;
            V_eff = inputs[2];
        }

        const auto* Q = Q_src.ptr<const float>();
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

        // is_causal takes one past length for the whole batch; a window or ragged batch cannot.
        bool ragged = false;
        for (int b = 1; b < batch_size && !ragged; ++b)
            ragged = (pastLen[b] != pastLen[0]);
        const bool window_mask = is_causal && (local_window_size > 0 || ragged);
        CV_CheckFalse(window_mask && has_mask_input,
                      "a sliding window cannot be combined with an attn_mask input");
        const int causal_past = pastLen[0];

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
                has_mask_input, is_causal, causal_past, /*do_softmax=*/false
            );
        }

        if (!window_mask) {
            fused_softmax_softcap_mask(
                attention_prob, mask_mat,
                softcap, softcap > 0.f, 9.f, -FLT_MAX,
                has_mask_input, is_causal, causal_past
            );
        } else {
            Mat mask(std::vector<int>{batch_size, 1, seq_len_q, total_seq_kv}, CV_8U, Scalar(0));
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < seq_len_q; ++i) {
                    const int hi = pastLen[b] + i;
                    int lo = 0;
                    // Spans local_window_size tokens ending at hi, not +1 (checked vs onnxruntime).
                    if (local_window_size > 0) lo = std::max(lo, hi - local_window_size + 1);
                    uchar* row = mask.ptr<uchar>(b, 0, i);
                    for (int j = lo; j <= hi && j < total_seq_kv; ++j)
                        row[j] = 1;
                }
            }

            // Cap before masking: the kernel masks first, and softcap would turn -FLT_MAX finite.
            if (softcap > 0.f)
                fused_softmax_softcap_mask(attention_prob, Mat(), softcap, true, 9.f, -FLT_MAX,
                                           /*has_mask*/ false, /*is_causal*/ false,
                                           /*past_seq_len*/ 0, /*do_softmax*/ false);

            fused_softmax_softcap_mask(attention_prob, mask, 0.f, false, 9.f, -FLT_MAX,
                                       /*has_mask*/ true, /*is_causal*/ false);
        }

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
    // past is 4D [B, nhkv, total, D]; fresh and out follow the query's rank.
    static void buildSharedKVBuffer(const Mat& past, const Mat& fresh, Mat& out,
                                    int batch_size, int nhkv, int total, int seq_len_q,
                                    int head_size, int input_dims,
                                    const std::vector<int>& validLen,
                                    const std::vector<int>& pastLen) {
        CV_Assert(fresh.isContinuous());
        if (input_dims == 4) {
            int dims[4] = {batch_size, nhkv, total, head_size};
            out.create(4, dims, CV_32F);
            CV_Assert(past.isContinuous());
            const float* src = past.ptr<float>();
            const float* add = fresh.ptr<float>();
            float* dst = out.ptr<float>();
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < nhkv; ++h) {
                    const size_t head = ((size_t)b * nhkv + h) * total * head_size;
                    if (pastLen[b] > 0)
                        std::memcpy(dst + head, src + head, sizeof(float) * pastLen[b] * head_size);
                    std::memcpy(dst + head + (size_t)pastLen[b] * head_size,
                                add + ((size_t)b * nhkv + h) * seq_len_q * head_size,
                                sizeof(float) * seq_len_q * head_size);
                    if (validLen[b] < total)
                        std::memset(dst + head + (size_t)validLen[b] * head_size, 0,
                                    sizeof(float) * (total - validLen[b]) * head_size);
                }
            }
        } else {
            // 3D: one token is one contiguous row, so heads need no separate loop.
            const int row = nhkv * head_size;
            int dims[3] = {batch_size, total, row};
            out.create(3, dims, CV_32F);
            Mat past3;
            cv::transposeND(past, {0, 2, 1, 3}, past3);
            const float* src = past3.ptr<float>();
            const float* add = fresh.ptr<float>();
            float* dst = out.ptr<float>();
            for (int b = 0; b < batch_size; ++b) {
                float* dstB = dst + (size_t)b * total * row;
                const float* srcB = src + (size_t)b * total * row;
                if (pastLen[b] > 0)
                    std::memcpy(dstB, srcB, sizeof(float) * pastLen[b] * row);
                std::memcpy(dstB + (size_t)pastLen[b] * row,
                            add + (size_t)b * seq_len_q * row,
                            sizeof(float) * seq_len_q * row);
                if (validLen[b] < total)
                    std::memset(dstB + (size_t)validLen[b] * row, 0,
                                sizeof(float) * (total - validLen[b]) * row);
            }
        }
    }

    bool is_causal;
    int q_num_heads;
    int qk_matmul_output_mode;
    float scale;
    bool is_scale_set = false;
    float softcap;
    int softmax_precision;

    int local_window_size;
    bool do_rotary;
    bool rotary_interleaved;
    bool shared_kv_buffer;
    int has_attn_mask_attr;
    int has_past_attr;
    bool has_seqlens_attr;
    bool has_rotary_attr;
    Ptr<RotaryEmbeddingLayer> ropeQ, ropeK;
    int ropeRotaryDim = -1;   // channel count the cached rotary layers were built for

    FastGemmOpt opt;
};

Ptr<AttentionOnnxAiLayer> AttentionOnnxAiLayer::create(const LayerParams &params) {
    return makePtr<AttentionOnnxAiLayerImpl>(params);
}

}} // cv::dnn
