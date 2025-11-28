// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "cpu_kernels/fast_gemm.hpp"
#include "cpu_kernels/softmax.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

using std::tanh;


static void mask_bias(
    Mat &att_weights, const Mat &att_mask,
    const int seq_len_square, const int seq_len_kv,
    const float min_val, const bool is_causal
)
{
    const int total = att_mask.total();
    auto* mask_data = att_mask.ptr<const float>();
    auto* weights_data = att_weights.ptr<float>();
    for (int i = 0; i < total; i++)
    {
        const int seq_pos = i % seq_len_square;
        const int q_pos = seq_pos / seq_len_kv;
        const int k_pos = seq_pos % seq_len_kv;
        weights_data[i] = (is_causal && k_pos > q_pos) ?
                          min_val : weights_data[i] + mask_data[i];
    }
}

static void mask_bool(
    Mat &att_weights, const Mat &att_mask,
    const int seq_len_square, const int seq_len_kv,
    const float min_val, const bool is_causal
)
{
    const int elemSize = CV_ELEM_SIZE1(att_mask.depth());  // bytes per element
    const int depth = att_mask.depth();
    auto* mask_data = att_mask.ptr<const char>();
    auto* weights_data = att_weights.ptr<float>();
    const int total = att_mask.total();
    for (int i = 0; i < total; i++)
    {
        const char* src = mask_data + i * elemSize;
        bool mask_val = false;
        switch (depth)
        {
            case CV_Bool:
                mask_val = *(reinterpret_cast<const bool*>(src));
                break;
            case CV_8U:
                mask_val = (*(reinterpret_cast<const uint8_t*>(src)) != 0);
                break;
            case CV_8S:
                mask_val = (*(reinterpret_cast<const int8_t*>(src)) != 0);
                break;
            case CV_16U:
                mask_val = (*(reinterpret_cast<const uint16_t*>(src)) != 0);
                break;
            case CV_16S:
                mask_val = (*(reinterpret_cast<const int16_t*>(src)) != 0);
                break;
            case CV_32S:
                mask_val = (*(reinterpret_cast<const int32_t*>(src)) != 0);
                break;
            case CV_64S:
                mask_val = (*(reinterpret_cast<const int64_t*>(src)) != 0);
                break;
            case CV_64U:
                mask_val = (*(reinterpret_cast<const uint64_t*>(src)) != 0);
                break;
            default:
                CV_Error(Error::StsNotImplemented, "Unsupported depth for boolean mask in attention");
        }
        const int seq_pos = i % seq_len_square;
        const int q_pos = seq_pos / seq_len_kv;
        const int k_pos = seq_pos % seq_len_kv;
        if (!mask_val || (is_causal && k_pos > q_pos))
        {
            weights_data[i] = min_val;
        }
    }
}


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

        if(inputs.size() > 3)
            // 1 broadcasted attention_mask
            internals.push_back(inputs[3]);

        // 2. attention_prob
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

        // internals:
        // 1. attention_mask (broadcasted input attention mask - B x H_q x T_q x T_kv)
        // 2. attention_prob (B x H_q x T_q x T_kv)

        // 1. broadcasted attention_mask
        if (inputs.size() > 3) {
            MatShape mask_shape{batch_size, nhq, seq_len_q, seq_len_kv};
            internals.push_back(mask_shape);
        }

        // 2. attention_prob
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

        auto &attention_prob = internals[internals.size() - 1];
        {
            auto *output = attention_prob.ptr<float>();

            auto loops = batch_size * nhq;

            // parallel_for_(Range(0, loops), [&] (const Range r) {
            const Range r = Range(0, loops);
            for (int i = r.start; i < r.end; i++) {
                const int _batch_index = i / nhq;
                const int _q_head_index = i % nhq;
                const int _k_head_index = _q_head_index / num_gq_groups;

                const int _q_offset = input_dims == 3 ?
                            _q_head_index * qk_head_size : _q_head_index * qk_head_size * seq_len_q;
                const int _k_offset = input_dims == 3 ?
                            _k_head_index * qk_head_size : _k_head_index * qk_head_size * seq_len_kv;

                const auto *q = Q + _batch_index * seq_len_q * qk_head_size * nhq +
                                _q_offset;
                const auto *k = K + _batch_index * seq_len_kv * qk_head_size * nhkv +
                                _k_offset;
                const int output_offset = i * seq_len_square;

                const int ldq0 = input_dims == 3 ? qk_head_size * nhq : qk_head_size;
                const int ldk0 = input_dims == 3 ? qk_head_size * nhkv : qk_head_size;

                fastGemm(
                    false, true,
                    seq_len_q, qk_head_size,
                    seq_len_kv, qk_head_size,
                    scale,
                    q, ldq0, 1,
                    k, ldk0, 1,
                    0.f,
                    output + output_offset, seq_len_kv,
                    opt);
            }
            // }, loops * seq_len_q * q_num_heads * seq_len_kv * (1 / 1024.0));
        }

        // Attention masking
        if (inputs.size() > 3 || is_causal)
        {
            if (internals.size() > 1) {
                Mat& attention_mask = internals[internals.size() - 2];

                broadcast(
                    inputs[3],
                    shape(attention_mask),
                    attention_mask
                );
            }
            if (inputs.size() > 3)
                if (CV_IS_INT_TYPE(inputs[3].depth()))
                    mask_bool(
                        attention_prob,
                        internals[internals.size() - 2],
                        seq_len_square,
                        seq_len_kv,
                        -1e9f,
                        is_causal
                    );
                else
                    mask_bias(
                        attention_prob,
                        internals[internals.size() - 2],
                        seq_len_square,
                        seq_len_kv,
                        -1e9f,
                        is_causal
                    );
            else
            {
                Range r(0, attention_prob.total());
                for (int i = r.start; i < r.end; i++)
                {
                    const int seq_pos = i % seq_len_square;
                    const int q_pos = seq_pos / seq_len_kv;
                    const int k_pos = seq_pos % seq_len_kv;
                    if (k_pos > q_pos)
                    {
                        attention_prob.ptr<float>()[i] = -1e9f;
                    }
                }
            }
        }


        // softcap, if provided
        if (softcap > 0.f) {
            float* attn_data = attention_prob.ptr<float>();
            auto total_elems = attention_prob.total();
            for (size_t i = 0; i < total_elems; i++) {
                attn_data[i] = tanh(attn_data[i] / softcap) * softcap;
            }
        }

        // Compute softmax on the last dimension
        softmax(attention_prob, attention_prob, shape(attention_prob).size() - 1);
        const auto attn = attention_prob.ptr<const float>();
        auto *output = outputs[0].ptr<float>();
        auto loops = batch_size * nhq;
        // parallel_for_(Range(0, loops), [&] (const Range r) {
        const Range r = Range(0, loops);
        for (int i = r.start; i < r.end; i++) {
            const int _batch_index = i / nhq;
            const int _q_head_index = i % nhq;
            const int _v_head_index = _q_head_index / num_gq_groups;

            const int v_offset = _batch_index * seq_len_kv * v_head_size * nhkv + (
                input_dims == 3 ?
                    _v_head_index * v_head_size :
                    _v_head_index * v_head_size * seq_len_kv
            );

            const int out_offset = _batch_index * seq_len_q * v_head_size * nhq + (
                input_dims == 3 ?
                    _q_head_index * v_head_size :
                    _q_head_index * v_head_size * seq_len_q
            );

            const auto att = attn + i * seq_len_square;
            const auto v = V + v_offset;

            const int ldv0 = input_dims == 3 ? v_head_size * nhkv : v_head_size;
            const int ldout = input_dims == 3 ? v_head_size * nhq : v_head_size;

            fastGemm(
                false, false,
                seq_len_q, seq_len_kv,
                seq_len_kv, v_head_size,
                1.f,
                att, seq_len_kv, 1,
                v, ldv0, 1,
                0.f,
                output + out_offset, ldout,
                opt);
        }
        // }, loops * seq_len_square * q_num_heads * (1 / 1024.0));

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
