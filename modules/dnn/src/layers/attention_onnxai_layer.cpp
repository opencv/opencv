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

        if (inputs.size() >= 4) {
            CV_CheckType(inputs[3], inputs[3] == CV_8U || inputs[3] == CV_8S ||
                         inputs[3] == CV_16U || inputs[3] == CV_16S ||
                         inputs[3] == CV_32S || inputs[3] == CV_64S ||
                         inputs[3] == CV_64U || inputs[3] == CV_Bool ||
                         inputs[3] == inputs[2], ""); // attention_mask
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
        CV_CheckTrue(inputs.size() < 5, "past key and past value are not supported yet");
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

        MatShape attention_prob_shape{batch_size , nhq, seq_len_q, seq_len_kv};
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
        const auto seq_len_square = seq_len_q * seq_len_kv;

        const auto* Q =  inputs[0].ptr<const float>();
        const auto* K =  inputs[1].ptr<const float>();
        const auto* V =  inputs[2].ptr<const float>();

        scale = is_scale_set ? scale : 1.0f / std::sqrt(static_cast<float>(qk_head_size));

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
                    b * seq_len_kv * qk_head_size * nhkv +
                    (n / num_gq_groups * qk_head_size) * (input_dims == 3 ? 1 : seq_len_kv);
                _v_offsets[b * nhq + n] =
                    b * seq_len_kv * v_head_size * nhkv +
                    (n / num_gq_groups * v_head_size) * (input_dims == 3 ? 1 : seq_len_kv);
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
            seq_len_q, seq_len_kv, qk_head_size , scale,
            Q, ldq0, 1,
            K, 1, ldk0,
            0.f,
            attention_prob.ptr<float>(), seq_len_kv,
            opt
        );


        fused_softmax_softcap_mask(
            attention_prob,
            inputs.size() > 3 ? inputs[3] : Mat(),
            softcap, softcap > 0.f,
            9.f,
            -FLT_MAX,
            inputs.size() > 3,
            is_causal
        );


        const int ldv0 = input_dims == 3 ? v_head_size * nhkv : v_head_size;
        const int ldout = input_dims == 3 ? v_head_size * nhq : v_head_size;

        fastGemmBatch(
            batch_size * nhq,
            _a_offsets.data(), _v_offsets.data(), _o_offsets.data(),
            seq_len_q, v_head_size, seq_len_kv, 1.f,
            attention_prob.ptr<float>(), seq_len_kv, 1,
            V, ldv0, 1,
            0.f,
            outputs[0].ptr<float>(), ldout,
            opt
        );
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
