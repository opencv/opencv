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

        const int q_dims = inputs[0].dims;
        const int k_dims = inputs[1].dims;
        const int v_dims = inputs[2].dims;

        CV_CheckTrue(q_dims >= 2 || q_dims <= 4, "Query must be 2D, 3D or 4D");
        CV_CheckTrue(k_dims >= 2 || k_dims <= 4, "Key must be 2D, 3D or 4D");
        //CV_CheckTrue(v_dims >= 3 || v_dims == 4, "Value must be 3D or 4D");

        const int batch_size = inputs[0][0];
        const int seq_len_q = (q_dims >= 3) ? inputs[0][q_dims - 2] : 1;
        int seq_len_kv = (k_dims >= 3) ? inputs[1][k_dims - 2] : 1;

        if (inputs.size() >= 6 && !inputs[4].empty()){
            int past_dim = inputs[4].dims;
            int past_len = (past_dim >= 3) ? inputs[4][past_dim - 2] : inputs[4][0];
            seq_len_kv += past_len;
        }

        const int q_hn = q_dims == 4 ? inputs[0][1] : q_num_heads;
        int kv_hn = k_dims == 4 ? inputs[1][1] : kv_num_heads;
        if (kv_hn == 0 && q_hn > 0) kv_hn = q_hn;

        if (v_dims >= 3 && k_dims >= 3) {
            CV_CheckTrue(inputs[2][v_dims - 2] == inputs[1][k_dims - 2],  "Key and Value sequence lengths must be equal");
        }

        if (q_hn > 0 && kv_hn > 0) {
            CV_CheckTrue(q_hn % kv_hn == 0, "q_num_heads must be divisible by kv_num_heads");
        }

        const int nhq = q_hn;

        if (q_dims == 4)
        {
            int v_head_size = inputs[2][v_dims - 1];
            MatShape output_shape{batch_size, nhq, seq_len_q, v_head_size};
            outputs.push_back(output_shape);
        }
        else
        {
            CV_CheckTrue(kv_hn > 0, "kv_num_heads must be set for non-4D inputs");
            int total_v_dim = inputs[2][v_dims - 1];
            int v_head_size = total_v_dim / kv_hn;
            MatShape output_shape{batch_size, seq_len_q, v_head_size * q_num_heads};
            outputs.push_back(output_shape);
        }

        if (requiredOutputs > 1 || inputs.size() >= 6) {
            // Shape matches Input K but with updated sequence length
            MatShape k_shape = inputs[1];
            k_shape[k_dims - 2] = seq_len_kv; // Total length
            outputs.push_back(k_shape);

            // Shape matches Input V but with updated sequence length
            MatShape v_shape = inputs[2];
            v_shape[v_dims - 2] = seq_len_kv; // Total length
            outputs.push_back(v_shape);
        }

        MatShape attention_prob_shape{batch_size , nhq, seq_len_q, seq_len_kv};
        internals.push_back(attention_prob_shape);

        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        if (inputs.size() < 3) {
            CV_Error(Error::StsBadArg, format("AttentionOnnxAiLayer: Expected at least 3 inputs (Q, K, V), but got %zu.", inputs.size()));
        }

        opt.init();

        if (inputs[0].depth() == CV_16F) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        const int input_dims = inputs[0].dims;
        const int batch_size = inputs[0].size[0];

        const int nhq = (input_dims == 4) ? inputs[0].size[1] : q_num_heads;
        int nhkv = (inputs[1].dims == 4) ? inputs[1].size[1] : kv_num_heads;
        if (nhkv == 0 && nhq > 0) nhkv = nhq;

        const int seq_len_q = (input_dims >= 3) ? inputs[0].size[input_dims - 2] : 1;
        int seq_len_kv = (inputs[1].dims >= 3) ? inputs[1].size[inputs[1].dims - 2] : 1;

        int last_dim_q = inputs[0].size[input_dims - 1];
        const int qk_head_size = (input_dims == 4) ? last_dim_q : (last_dim_q / nhq);

        int last_dim_v = inputs[2].size[inputs[2].dims - 1];
        const int v_head_size = (inputs[2].dims == 4) ? last_dim_v : (last_dim_v / nhkv);

        const int num_gq_groups = nhq / nhkv;

        const float* Q = inputs[0].ptr<const float>();
        const float* K = inputs[1].ptr<const float>();
        const float* V = inputs[2].ptr<const float>();

        if (inputs.size() >= 6 && outputs.size() >= 3 && !inputs[4].empty())
        {
            int axis = (input_dims >= 3) ? (input_dims - 2) : 0;

            Mat& past_k = inputs[4];
            Mat& past_v = inputs[5];
            Mat& present_k = outputs[1];
            Mat& present_v = outputs[2];

            if (present_k.empty()) {
                std::vector<int> k_shape(inputs[1].dims);
                for (int i = 0; i < inputs[1].dims; i++) k_shape[i] = inputs[1].size[i];
                k_shape[axis] = past_k.size[axis] + inputs[1].size[axis];
                present_k.create(inputs[1].dims, k_shape.data(), inputs[1].type());
            }

            if (present_v.empty()) {
                std::vector<int> v_shape(inputs[2].dims);
                for (int i = 0; i < inputs[2].dims; i++) v_shape[i] = inputs[2].size[i];
                v_shape[axis] = past_v.size[axis] + inputs[2].size[axis]; // Sum: Past + New

                present_v.create(inputs[2].dims, v_shape.data(), inputs[2].type());
            }

            std::vector<Range> ranges_past(input_dims, Range::all());
            ranges_past[axis] = Range(0, past_k.size[axis]);
            past_k.copyTo(present_k(ranges_past));

            std::vector<Range> ranges_new(input_dims, Range::all());
            ranges_new[axis] = Range(past_k.size[axis], present_k.size[axis]);
            inputs[1].copyTo(present_k(ranges_new));

            past_v.copyTo(present_v(ranges_past));
            inputs[2].copyTo(present_v(ranges_new));

            K = present_k.ptr<const float>();
            V = present_v.ptr<const float>();

            seq_len_kv = present_k.size[axis];
        }

        const size_t seq_len_square = (size_t)seq_len_q * seq_len_kv;
        scale = is_scale_set ? scale : 1.0f / std::sqrt(static_cast<float>(qk_head_size));

        size_t required_size = (size_t)batch_size * nhq;
        if (_q_offsets.size() != required_size) {
            _q_offsets.resize(required_size);
            _k_offsets.resize(required_size);
            _a_offsets.resize(required_size);
            _v_offsets.resize(required_size);
            _o_offsets.resize(required_size);
        }

        for (int b = 0; b < batch_size; b++) {
            for (int n = 0; n < nhq; n++) {
                // Strides as size_t
                size_t q_stride_n = (input_dims == 4) ? ((size_t)qk_head_size * seq_len_q) : (size_t)qk_head_size;
                size_t kv_stride_n = (inputs[1].dims == 4) ? ((size_t)qk_head_size * seq_len_kv) : (size_t)qk_head_size;
                size_t v_stride_n = (inputs[2].dims == 4) ? ((size_t)v_head_size * seq_len_kv) : (size_t)v_head_size;
                size_t o_stride_n = (input_dims == 4) ? ((size_t)v_head_size * seq_len_q) : (size_t)v_head_size;

                _q_offsets[(size_t)b * nhq + n] = (size_t)b * seq_len_q * qk_head_size * nhq + (size_t)n * q_stride_n;
                _k_offsets[(size_t)b * nhq + n] = (size_t)b * seq_len_kv * qk_head_size * nhkv + ((size_t)n / num_gq_groups) * kv_stride_n;
                _v_offsets[(size_t)b * nhq + n] = (size_t)b * seq_len_kv * v_head_size * nhkv + ((size_t)n / num_gq_groups) * v_stride_n;
                _a_offsets[(size_t)b * nhq + n] = (size_t)b * seq_len_square * nhq + (size_t)n * seq_len_square;
                _o_offsets[(size_t)b * nhq + n] = (size_t)b * seq_len_q * v_head_size * nhq + (size_t)n * o_stride_n;
            }
        }

        const int ldq0 = (input_dims == 4) ? qk_head_size : (qk_head_size * nhq);
        const int ldk0 = (inputs[1].dims == 4) ? qk_head_size : (qk_head_size * nhkv);
        auto &attention_prob = internals[internals.size() - 1];

        fastGemmBatch(
            (size_t)batch_size * nhq,
            _q_offsets.data(), _k_offsets.data(), _a_offsets.data(),
            seq_len_q, seq_len_kv, qk_head_size, scale,
            Q, ldq0, 1,
            K, 1, ldk0,
            0.f,
            attention_prob.ptr<float>(), seq_len_kv,
            opt
        );

        bool use_mask = (inputs.size() > 3 && !inputs[3].empty());

        fused_softmax_softcap_mask(
            attention_prob,
            use_mask ? inputs[3] : Mat(),
            softcap, softcap > 0.f,
            static_cast<float>(nhq),
            -FLT_MAX,
            use_mask,
            is_causal
        );

        const int ldv0 = (inputs[2].dims == 4) ? v_head_size : (v_head_size * nhkv);
        const int ldout = (input_dims == 4) ? v_head_size : (v_head_size * nhq);

        if (outputs[0].empty() || outputs[0].total() != (size_t)batch_size * nhq * seq_len_q * v_head_size) {
            int out_sz[] = {batch_size, nhq, seq_len_q, v_head_size};
            outputs[0].create(4, out_sz, inputs[0].type());
        }

        fastGemmBatch(
            (size_t)batch_size * nhq,
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

    std::vector<size_t> _q_offsets, _k_offsets, _a_offsets, _v_offsets, _o_offsets;
};

Ptr<AttentionOnnxAiLayer> AttentionOnnxAiLayer::create(const LayerParams &params) {
    return makePtr<AttentionOnnxAiLayerImpl>(params);
}

}} // cv::dnn
