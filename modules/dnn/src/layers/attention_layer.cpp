// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "cpu_kernels/fast_gemm.hpp"
#include "cpu_kernels/softmax.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

// buffer_shape: [B, S, Dq+Dk+Dv] = [B, S, NH_q + NH_k + NH_v]
// bias_shape:   [Dq+Dk+Dv]
// q/k/v_shape:  [B, N, S, H] x 3
static void SplitQKVBufferWithBias(const float *buffer, const float *bias, float *q, float *k, float *v,
                                   size_t batch_size, size_t seq_len, size_t num_heads, size_t qk_head_size, size_t v_head_size) {
    size_t head_size = qk_head_size + qk_head_size + v_head_size,
           NS = num_heads * seq_len, NSH = num_heads * seq_len * head_size,
           NHqk = num_heads * qk_head_size, NH = num_heads * head_size,
           outer_size = batch_size * NS;

    auto worker = [&](const Range &r) {
        for (int i = r.start; i < r.end; i++) {
            const size_t dst_batch_index = i / NS;
            const size_t dst_head_index = (i - dst_batch_index * NS) / seq_len;
            const size_t dst_seq_index = i % seq_len;
            size_t buffer_offset = dst_batch_index * NSH + dst_seq_index * NH;

            // Split for Q and add bias
            std::memcpy(q, buffer + buffer_offset + dst_head_index * qk_head_size, qk_head_size * sizeof(float));
            for (size_t j = 0; j < qk_head_size; j++) {
                q[j] += bias[dst_head_index * qk_head_size + j];
            }
            q += qk_head_size;

            // Split for K and add bias
            std::memcpy(k, buffer + buffer_offset + NHqk + dst_head_index * qk_head_size, qk_head_size * sizeof(float));
            for (size_t j = 0; j < qk_head_size; j++) {
                k[j] += bias[NHqk + dst_head_index * qk_head_size + j];
            }
            k += qk_head_size;

            // Split for V and add bias
            std::memcpy(v, buffer + buffer_offset + 2 * NHqk + dst_head_index * v_head_size, v_head_size * sizeof(float));
            for (size_t j = 0; j < v_head_size; j++) {
                v[j] += bias[2 * NHqk + dst_head_index * v_head_size + j];
            }
            v += v_head_size;
        }
    };
    double nstripes = outer_size * (1 / 1024.0);
    parallel_for_(Range(0, outer_size), worker, nstripes);
}

// Operator spec: https://github.com/microsoft/onnxruntime/blob/v1.16.1/docs/ContribOperators.md#com.microsoft.Attention
class AttentionLayerImpl CV_FINAL : public AttentionLayer {
 public:
    AttentionLayerImpl(const LayerParams &params) {
        setParamsFrom(params);

        CV_CheckTrue(params.has("num_heads"), "DNN/Attention: num_heads is required but missing");
        num_heads = params.get<int>("num_heads"); // required, no default value

        CV_CheckTrue(params.has("qkv_hidden_sizes"), "DNN/Attention: qkv_hidden_sizes is required but missing");
        auto param_qkv_hidden_sizes = params.get("qkv_hidden_sizes");
        CV_CheckEQ(param_qkv_hidden_sizes.size(), 3, "DNN/Attention: qkv_hidden_sizes must and only have three elements");

        qkv_hidden_sizes.clear();
        qkv_hidden_sizes.resize(3);
        qkv_hidden_sizes[0] = static_cast<size_t>(param_qkv_hidden_sizes.get<int>(0));
        qkv_hidden_sizes[1] = static_cast<size_t>(param_qkv_hidden_sizes.get<int>(1));
        /* v_hidden_size needs to be initialized in finalize in case v_slice_end=INT_MAX */

        qkv_head_sizes.clear();
        qkv_head_sizes.resize(3);
        qkv_head_sizes[0] = static_cast<size_t>(qkv_hidden_sizes[0] / num_heads);
        qkv_head_sizes[1] = static_cast<size_t>(qkv_hidden_sizes[1] / num_heads);

        scale = 1.f / params.get<float>("scale", sqrt(qkv_head_sizes[0]));

        output_ndims = params.get<int>("output_ndims", 3);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        const int total_inputs = static_cast<int>(inputs.size() + blobs.size());
        CV_CheckGE(total_inputs, 1, "DNN/Attention: one input at least");
        CV_CheckLE(total_inputs, 3, "DNN/Attention: three inputs at most");
        const auto &input_shape = inputs[0];
        const auto &weight_shape = inputs.size() >= 2 ? inputs[1] : shape(blobs.front());
        const auto &bias_shape = inputs.size() >= 3 ? inputs[2] : shape(blobs.back());

        CV_CheckEQ(input_shape.size(), static_cast<size_t>(3), "DNN/Attention: invalid input dimension");
        CV_CheckEQ(weight_shape.size(), static_cast<size_t>(2), "DNN/Attention: invalid weight dimension");

        CV_CheckEQ(input_shape[2], weight_shape[0], "DNN/Attention: invalid input shape");
        CV_CheckEQ(weight_shape[1], bias_shape[0], "DNN/Attention: invalid weight or bias shape");

        if (output_ndims == 3) {
            outputs.assign(1, inputs[0]);
        } else if (output_ndims == 2) {
            int batch = input_shape[0], seq_len = input_shape[1], input_hidden_size = input_shape[2];
            MatShape output_shape{batch * seq_len, input_hidden_size};
            outputs.assign(1, output_shape);
        } else {
            CV_Error(Error::StsBadArg, format("DNN/Attention: invalid output dimension %zu, valid value is 2 or 3", output_ndims));
        }

        const int batch_size_ = input_shape[0], seq_len_ = input_shape[1],
                  hidden_size_ = weight_shape.back(),
                  num_heads_ = static_cast<int>(num_heads),
                  v_head_size_ = static_cast<int>((hidden_size_ - qkv_hidden_sizes[0] - qkv_hidden_sizes[1]) / num_heads);

        MatShape gemm_buffer_shape{batch_size_, seq_len_, hidden_size_},
                 attention_prob_shape{batch_size_ * num_heads_, seq_len_, seq_len_},
                 output_buffer_shape{batch_size_ * num_heads_, seq_len_, v_head_size_};
        internals.assign(1, gemm_buffer_shape);
        internals.push_back(gemm_buffer_shape);
        internals.push_back(attention_prob_shape);
        internals.push_back(output_buffer_shape);

        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        opt.init();

        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        const auto input_shape = shape(inputs[0]);
        batch_size = static_cast<size_t>(input_shape[0]);
        seq_len = static_cast<size_t>(input_shape[1]);
        input_hidden_size = static_cast<size_t>(input_shape[2]);

        const auto weight_shape = inputs.size() >= 2 ? shape(inputs[1]) : shape(blobs.front());
        hidden_size = weight_shape[1];
        qkv_hidden_sizes[2] = hidden_size - qkv_hidden_sizes[0] - qkv_hidden_sizes[1];
        qkv_head_sizes[2] = static_cast<size_t>(qkv_hidden_sizes[2] / num_heads);

        // Prepack weight
        matmulHelper.compute(false, false, input_shape, weight_shape, std::vector<int>{int(batch_size), int(seq_len), int(hidden_size)});
        if (inputs.size() < 2) {
            fastGemmPackB(blobs[0], packed_weight, false, opt);
            matmulHelper.updatePackedBOffsets(packed_weight.size());
        }
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        // Compute Add(MatMul(Input, Weight), Bias)
        auto &gemm_buffer = internals[0];
        if (blobs.empty()) {
            const auto *input = inputs[0].ptr<const float>();
            const auto *weight = inputs[1].ptr<const float>();
            fastGemmBatch(matmulHelper.batch, matmulHelper.A_offsets.data(), matmulHelper.B_offsets.data(), matmulHelper.C_offsets.data(),
                          matmulHelper.M, matmulHelper.N, matmulHelper.K, 1.0f, input, matmulHelper.lda0, matmulHelper.lda1,
                          weight, matmulHelper.ldb0, matmulHelper.ldb1, 0.f, gemm_buffer.ptr<float>(), matmulHelper.ldc, opt);
        } else {
            const auto *input = inputs[0].ptr<const float>();
            fastGemmBatch(matmulHelper.batch, matmulHelper.A_offsets.data(), matmulHelper.B_offsets.data(), matmulHelper.C_offsets.data(),
                          matmulHelper.M, matmulHelper.N, matmulHelper.K, 1.0f, input, matmulHelper.lda0, matmulHelper.lda1,
                          packed_weight.data(), 0.f, gemm_buffer.ptr<float>(), matmulHelper.ldc, opt);
        }
        // Split and get Q, K, V
        auto *Q = internals[1].ptr<float>(),
             *K = Q + batch_size * num_heads * seq_len * qkv_head_sizes[0],
             *V = K + batch_size * num_heads * seq_len * qkv_head_sizes[1];
        const auto &bias = blobs.empty() ? inputs[2] : blobs.back();
        SplitQKVBufferWithBias(gemm_buffer.ptr<const float>(), bias.ptr<const float>(), Q, K, V, batch_size, seq_len, num_heads, qkv_head_sizes[0], qkv_head_sizes[2]);

        // Compute softmax(scale * matmul(Q, K))
        auto &attention_prob = internals[2];
        {
            auto *output = attention_prob.ptr<float>();

            auto loops = batch_size * num_heads;
            auto seq_len_square = seq_len * seq_len;
            auto qk_head_size = qkv_head_sizes[0];
            auto qk_inner_size = seq_len * qk_head_size;

            // Compute: attention_prob = scale * MatMul(Q, K)
            // Q: [B, N, S, H]
            // K: [B, N, S, H] // transB=true
            // P: [B, N, S, S]
            std::vector<size_t> AB_offsets(loops), C_offsets(loops);
            for (size_t i = 0; i < loops; i++) {
                AB_offsets[i] = i * qk_inner_size;
                C_offsets[i] = i * seq_len_square;
            }
            fastGemmBatch(loops, AB_offsets.data(), AB_offsets.data(), C_offsets.data(),
                          seq_len, seq_len, qk_head_size, scale, Q, qk_head_size, 1,
                          K, 1, qk_head_size, 0.f, output, seq_len, opt);

            // Compute softmax on the last dimension
            softmax(attention_prob, attention_prob, shape(attention_prob).size() - 1);
        }

        // Compute np.matmul(attention_prob, V)
        auto &output_buffer = internals[3];
        {
            auto *output = outputs[0].ptr<float>();
            auto *output_buff = output_buffer.ptr<float>();
            const auto *prob = attention_prob.ptr<const float>();

            auto loops = batch_size * num_heads;
            auto prob_inner_size = seq_len * seq_len;
            auto v_head_size = qkv_head_sizes[2];
            auto v_inner_size = seq_len * v_head_size;

            opt.multi_thread = false;
            parallel_for_(Range(0, loops), [&] (const Range &r) {
                for (int i = r.start; i < r.end; i++) {
                    const int output_offset = i * v_inner_size;

                    const auto *p = prob + i * prob_inner_size, *v = V + i * v_inner_size;
                    fastGemm(false, false, seq_len, seq_len, seq_len, v_head_size,
                             1.f, p, seq_len, 1,
                             v, v_head_size, 1, 0.f,
                             output_buff + output_offset, v_head_size, opt);

                    // tranpose on the fly
                    const int batch_index = static_cast<int>(i / num_heads);
                    const int head_index = static_cast<int>(i % num_heads);
                    auto *src = output_buff + output_offset;
                    auto *dst = output + (batch_index * seq_len * num_heads + head_index) * v_head_size;
                    for (int j = 0; j < seq_len; j++) {
                        std::memcpy(dst, src, v_head_size * sizeof(float));
                        src += v_head_size;
                        dst += qkv_hidden_sizes[2];
                    }
                }
            }, loops * seq_len * seq_len * v_head_size * (1 / 1024.0));
        }
    }

 private:
    size_t num_heads;
    std::vector<size_t> qkv_hidden_sizes; // order: {qk_hidden_size, qk_hidden_size, v_hidden_size}
    float scale;
    size_t output_ndims;

    std::vector<size_t> qkv_head_sizes; // order: {qk_head_size, qk_head_size, v_head_size}

    size_t batch_size;
    size_t seq_len;
    size_t input_hidden_size;
    size_t hidden_size;

    std::vector<float> packed_weight;

    FastGemmOpt opt;
    MatMulHelper matmulHelper;
};

Ptr<AttentionLayer> AttentionLayer::create(const LayerParams &params) {
    return makePtr<AttentionLayerImpl>(params);
}

}} // cv::dnn
