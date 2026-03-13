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

    // Returns index of past_key in inputs (-1 if not present).
    // Convention:
    //   inputs.size() == 5: Q, K, V, past_key, past_value  (no mask)
    //   inputs.size() == 6: Q, K, V, mask, past_key, past_value
    static int pastKeyIdx(size_t n_inputs) {
        if (n_inputs == 5) return 3;
        if (n_inputs == 6) return 4;
        return -1;
    }

    // Returns index of past_value in inputs (-1 if not present).
    static int pastValueIdx(size_t n_inputs) {
        if (n_inputs == 5) return 4;
        if (n_inputs == 6) return 5;
        return -1;
    }

    // Returns true if mask is present in inputs.
    static bool hasMaskInput(size_t n_inputs) {
        return n_inputs == 4 || n_inputs == 6;
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

        // inputs[3] = attention_mask when size==4 or size==6
        // inputs[3] = past_key when size==5 (no mask)
        // inputs[4] = past_key when size==6, or past_value when size==5
        // inputs[5] = past_value when size==6
        const bool has_mask = hasMaskInput(inputs.size());
        const bool has_past_kv = (pastKeyIdx(inputs.size()) >= 0);

        if (has_mask) {
            CV_CheckType(inputs[3], inputs[3] == CV_8U || inputs[3] == CV_8S ||
                         inputs[3] == CV_16U || inputs[3] == CV_16S ||
                         inputs[3] == CV_32S || inputs[3] == CV_64S ||
                         inputs[3] == CV_64U || inputs[3] == CV_Bool ||
                         inputs[3] == inputs[2], ""); // attention_mask
        }

        if (has_past_kv) {
            const int past_k_idx = pastKeyIdx(inputs.size());
            const int past_v_idx = pastValueIdx(inputs.size());
            CV_CheckType(inputs[past_k_idx], inputs[past_k_idx] == inputs[0], ""); // past_key
            CV_CheckType(inputs[past_v_idx], inputs[past_v_idx] == inputs[0], ""); // past_value
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

        const int batch_size = inputs[0][0];
        const int seq_len_q = inputs[0][input_dims - 2];
        const int seq_len_kv = inputs[1][input_dims - 2];

        const int q_hn = input_dims == 4 ?
            inputs[0][1] : q_num_heads;
        const int kv_hn =input_dims == 4 ?
            inputs[1][1] : kv_num_heads;

        CV_CheckTrue(inputs[2][input_dims - 2] == seq_len_kv,
                     "Key and query sequence lengths must be equal");
        const int nhq = input_dims == 4 ? inputs[0][1] : q_num_heads;

        CV_CheckTrue(q_hn % kv_hn == 0,
                     "q_num_heads must be divisible by kv_num_heads");

        if (input_dims == 3)
        {
            CV_CheckTrue(kv_hn > 0,
                         "For 3D input, kv_num_heads must be greater than 0 (this normally means that kv_num_heads is not set)");
            CV_CheckTrue(q_hn > 0,
                         "For 3D input, q_num_heads must be greater than 0 (this normally means that q_num_heads is not set)");
        }

        // Determine KV cache inputs:
        // inputs[3] = attention_mask (optional) when size==4 or size==6
        // inputs[3] = past_key when size==5 (no mask); inputs[4] = past_value
        // inputs[4] = past_key, inputs[5] = past_value when size==6
        const int past_k_idx = pastKeyIdx(inputs.size());
        const bool use_kv_cache = (past_k_idx >= 0);
        const int past_seq_kv = use_kv_cache ? inputs[past_k_idx][input_dims - 2] : 0;
        const int total_seq_kv = past_seq_kv + seq_len_kv;

        if (input_dims == 3)
        {
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

        // Add present_key and present_value output shapes when KV cache is used
        if (use_kv_cache) {
            MatShape present_key_shape = inputs[1];
            present_key_shape[input_dims - 2] = total_seq_kv;
            outputs.push_back(present_key_shape);

            MatShape present_value_shape = inputs[2];
            present_value_shape[input_dims - 2] = total_seq_kv;
            outputs.push_back(present_value_shape);
        }

        MatShape attention_prob_shape{batch_size, nhq, seq_len_q, total_seq_kv};
        internals.push_back(attention_prob_shape);

        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        opt.init();

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        const int input_dims = inputs[0].dims;
        const int batch_size = inputs[0].size[0];

        const int nhq = input_dims == 3 ?
                         q_num_heads :
                         inputs[0].size[1];
        const int nhkv = input_dims == 3 ?
                         kv_num_heads :
                         inputs[1].size[1];

        const int qk_head_size = input_dims == 3 ?
                                 inputs[0].size[2] / nhq :
                                 inputs[0].size[3];
        const int v_head_size  = input_dims == 3 ?
                                 inputs[2].size[2] / nhkv :
                                 inputs[2].size[3];
        const int num_gq_groups = nhq / nhkv;
        const int seq_len_q = input_dims == 3 ?
                                inputs[0].size[1]:
                                inputs[0].size[2];
        const int seq_len_kv = input_dims == 3 ?
                                inputs[1].size[1]:
                                inputs[1].size[2];

        // KV cache: detect past_key / past_value inputs
        // Convention:
        //   inputs.size() == 3: Q, K, V
        //   inputs.size() == 4: Q, K, V, mask
        //   inputs.size() == 5: Q, K, V, past_key, past_value  (no mask)
        //   inputs.size() == 6: Q, K, V, mask, past_key, past_value
        const int past_k_idx = pastKeyIdx(inputs.size());
        const int past_v_idx = pastValueIdx(inputs.size());
        const bool use_kv_cache   = (past_k_idx >= 0);
        const bool has_mask_input = hasMaskInput(inputs.size());

        // Sequence dimension index within the K/V tensor (1 for 3D, 2 for 4D)
        const int seq_dim = input_dims - 2;
        const int past_seq_kv = use_kv_cache ? inputs[past_k_idx].size[seq_dim] : 0;
        const int total_seq_kv = past_seq_kv + seq_len_kv;

        // Build effective K and V by concatenating past_key/past_value with current K/V
        Mat K_eff, V_eff;
        if (use_kv_cache && past_seq_kv > 0) {
            const float* past_k_ptr = inputs[past_k_idx].ptr<const float>();
            const float* past_v_ptr = inputs[past_v_idx].ptr<const float>();
            const float* k_ptr = inputs[1].ptr<const float>();
            const float* v_ptr = inputs[2].ptr<const float>();

            if (input_dims == 3) {
                // Layout: [batch, seq, nhkv * head_size]
                const int k_elem = nhkv * qk_head_size;  // elements per seq position in K
                const int v_elem = nhkv * v_head_size;   // elements per seq position in V
                int dims_k[3] = {batch_size, total_seq_kv, k_elem};
                int dims_v[3] = {batch_size, total_seq_kv, v_elem};
                K_eff.create(3, dims_k, CV_32F);
                V_eff.create(3, dims_v, CV_32F);
                float* k_eff_ptr = K_eff.ptr<float>();
                float* v_eff_ptr = V_eff.ptr<float>();
                for (int b = 0; b < batch_size; b++) {
                    float* k_dst = k_eff_ptr + b * total_seq_kv * k_elem;
                    memcpy(k_dst, past_k_ptr + b * past_seq_kv * k_elem,
                           past_seq_kv * k_elem * sizeof(float));
                    memcpy(k_dst + past_seq_kv * k_elem, k_ptr + b * seq_len_kv * k_elem,
                           seq_len_kv * k_elem * sizeof(float));

                    float* v_dst = v_eff_ptr + b * total_seq_kv * v_elem;
                    memcpy(v_dst, past_v_ptr + b * past_seq_kv * v_elem,
                           past_seq_kv * v_elem * sizeof(float));
                    memcpy(v_dst + past_seq_kv * v_elem, v_ptr + b * seq_len_kv * v_elem,
                           seq_len_kv * v_elem * sizeof(float));
                }
            } else {
                // Layout: [batch, nhkv, seq, head_size]
                const int k_hs = qk_head_size;
                const int v_hs = v_head_size;
                int dims_k[4] = {batch_size, nhkv, total_seq_kv, k_hs};
                int dims_v[4] = {batch_size, nhkv, total_seq_kv, v_hs};
                K_eff.create(4, dims_k, CV_32F);
                V_eff.create(4, dims_v, CV_32F);
                float* k_eff_ptr = K_eff.ptr<float>();
                float* v_eff_ptr = V_eff.ptr<float>();
                for (int b = 0; b < batch_size; b++) {
                    for (int n = 0; n < nhkv; n++) {
                        float* k_dst = k_eff_ptr + (b * nhkv + n) * total_seq_kv * k_hs;
                        const float* k_src_past = past_k_ptr + (b * nhkv + n) * past_seq_kv * k_hs;
                        const float* k_src_cur  = k_ptr + (b * nhkv + n) * seq_len_kv * k_hs;
                        memcpy(k_dst, k_src_past, past_seq_kv * k_hs * sizeof(float));
                        memcpy(k_dst + past_seq_kv * k_hs, k_src_cur,
                               seq_len_kv * k_hs * sizeof(float));

                        float* v_dst = v_eff_ptr + (b * nhkv + n) * total_seq_kv * v_hs;
                        const float* v_src_past = past_v_ptr + (b * nhkv + n) * past_seq_kv * v_hs;
                        const float* v_src_cur  = v_ptr + (b * nhkv + n) * seq_len_kv * v_hs;
                        memcpy(v_dst, v_src_past, past_seq_kv * v_hs * sizeof(float));
                        memcpy(v_dst + past_seq_kv * v_hs, v_src_cur,
                               seq_len_kv * v_hs * sizeof(float));
                    }
                }
            }
        } else {
            // No concatenation needed: use current K/V directly
            K_eff = inputs[1];
            V_eff = inputs[2];
        }

        scale = is_scale_set ? scale : 1.0f / std::sqrt(static_cast<float>(qk_head_size));

        const auto* Q = inputs[0].ptr<const float>();
        const auto* K = K_eff.ptr<const float>();
        const auto* V = V_eff.ptr<const float>();

        const auto seq_len_square = seq_len_q * total_seq_kv;

        std::vector<size_t> _q_offsets(nhq * batch_size),
                _k_offsets(nhq * batch_size),
                _a_offsets(nhq * batch_size),
                _v_offsets(nhq * batch_size),
                _o_offsets(nhq * batch_size);

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
        auto &attention_prob = internals[internals.size() - 1];

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

        const Mat& mask_mat = has_mask_input ? inputs[3] : Mat();
        fused_softmax_softcap_mask(
            attention_prob,
            mask_mat,
            softcap, softcap > 0.f,
            9.f,
            -FLT_MAX,
            has_mask_input,
            is_causal,
            past_seq_kv
        );


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

        // Write present_key and present_value outputs
        if (use_kv_cache) {
            K_eff.copyTo(outputs[1]);
            V_eff.copyTo(outputs[2]);
        }
    }

 private:
    bool is_causal;
    int kv_num_heads;
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
