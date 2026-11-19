// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "../op_cuda.hpp"
#include "cpu_kernels/fast_gemm.hpp"
#include "cpu_kernels/softmax.hpp"
#include "cpu_kernels/mlas_gemm.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/attention.hpp"
#endif

namespace cv { namespace dnn {

static void packWeight(size_t num_heads, size_t head_size, size_t input_hidden_size,
                       const float *weight_data, size_t hidden_size, std::vector<float> &packed_weight, const FastGemmOpt &opt) {
    // num_heads * pack(head_size, input_hidden_size)
    size_t pack_size = fastGemmPackBSize(head_size, input_hidden_size, opt);
    size_t packed_weight_size = num_heads * pack_size;
    packed_weight.resize(packed_weight_size, 0.f);
    auto *packed_weight_data = packed_weight.data();
    for (size_t i = 0; i < num_heads; i++) {
        fastGemmPackB(false, head_size, input_hidden_size, weight_data, hidden_size, packed_weight_data, opt);
        packed_weight_data += pack_size;
        weight_data += head_size;
    }
}


static void rotationKernel(
    float* data, const float* rotation_table,
    size_t seq_len, size_t d
)
{
    CV_Assert(d % 2 == 0);
    const size_t d_half = d / 2;

    double nstripes = double(seq_len) * d_half * (1.0/1024.0);

    auto fn = [&](const cv::Range& range)
    {
        for (int t = range.start; t < range.end; ++t)
        {
            float* out_ptr = data + size_t(t) * d;
            const float* table_ptr = rotation_table + size_t(t) * d;
            size_t i = 0;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            const size_t w = VTraits<v_float32>::vlanes();
            for (; i + w <= d_half; i += w)
            {
                v_float32 sin_v, cos_v, x_even, x_odd;
                v_load_deinterleave(table_ptr + 2*i, sin_v, cos_v);
                v_load_deinterleave(out_ptr    + 2*i, x_even, x_odd);

                v_float32 out_even = v_sub(v_mul(cos_v, x_even), v_mul(sin_v, x_odd));
                v_float32 out_odd  = v_add(v_mul(sin_v, x_even), v_mul(cos_v, x_odd));

                v_store_interleave(out_ptr + 2*i, out_even, out_odd);
            }
#endif
            // scalar tail
            for (; i < d_half; ++i)
            {
                float s  = table_ptr[2*i  ];
                float c  = table_ptr[2*i+1];
                float xe = out_ptr[2*i];
                float xo = out_ptr[2*i+1];
                out_ptr[2*i]   = xe * c - xo * s;
                out_ptr[2*i+1] = xo * c + xe * s;
            }
        }
    };
    // This will spin up threads and run fn over [0, seq_len)
    parallel_for_(cv::Range(0, int(seq_len)), fn, nstripes);
}

// Precomputes RoPE sin/cos table of shape [seq_len, d] (https://arxiv.org/pdf/2104.09864).
static void precompRotationTable(float *data,
                                  size_t seq_len,
                                  size_t d) {
    // RoPE precomputation
    // RoPE is a positional encoding method used in transformer models.
    // It uses sine and cosine functions to encode the position of tokens in a sequence
    // initially introduced for NLP in https://arxiv.org/pdf/2104.09864

    // assume data is of shape [seq_ken,d]
    const float  logBase = std::log(10000.0f);
    const float  inv_d   = 1.0f / float(d);
    const size_t d_half = d / 2;
    for (size_t pos = 0; pos < seq_len; ++pos) {

        size_t i = 0;
        float* data_ptr = data + pos * d;

#if (CV_SIMD || CV_SIMD_SCALABLE)
        const size_t w = VTraits<v_float32>::vlanes();
        const v_float32 v_logBase = vx_setall_f32(logBase);
        const v_float32 v_inv_d   = vx_setall_f32(inv_d);
        const v_float32 v_neg2    = vx_setall_f32(-2.0f);

        for (; i + w <= d_half; i+=w) {
            int idx_buf[VTraits<v_float32>::max_nlanes];
            for (int k = 0; k < int(w); ++k)
                idx_buf[k] = int(i + k);
            // [i, i+1, …, i+w-1]
            v_float32 v_idx = v_cvt_f32(vx_load(idx_buf));
            // [10_000^(-i/d), 10_000^(-(i+1)/d), …, 10_000^(-(i+w-1)/d)]
            v_float32 v_theta = v_exp(v_mul(v_mul(v_neg2, v_mul(v_idx, v_inv_d)), v_logBase));
            v_theta = v_mul(vx_setall_f32(float(pos)), v_theta);
            v_float32 sin_v, cos_v;
            v_sincos(v_theta, sin_v, cos_v);
            // store back with interleave
            v_store_interleave(data_ptr + 2*i, sin_v, cos_v);
        }
#endif
        // scalar tail
        for (; i < d_half; i+=1)
        {
            float theta = pos * std::exp(-2.f * i * inv_d * logBase);
            data_ptr[2*i    ] = std::sin(theta);
            data_ptr[2*i + 1] = std::cos(theta);
        }
    }
}


// Operator spec: https://github.com/microsoft/onnxruntime/blob/v1.16.1/docs/ContribOperators.md#com.microsoft.Attention
// Extension: a fused Attention may take [..., C] and group its tokens into batch and sequence the
// way the replaced QKV reshape did (spec: 3D only).
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

        do_rotary = params.get<bool>("do_rotary", false);

        // Set by the attention fusion: {factor, dims...} for the QKV reshape's batch and sequence
        // entries, factor -1 when the entry is inferred.
        auto readExpr = [&](const char* key, std::vector<int>& expr) {
            if (!params.has(key)) return;
            const DictValue& v = params.get(key);
            for (int i = 0; i < v.size(); i++)
                expr.push_back(v.get<int>(i));
        };
        readExpr("qkv_batch", qkv_batch);
        readExpr("qkv_seq", qkv_seq);
        CV_CheckEQ(qkv_batch.empty(), qkv_seq.empty(), "DNN/Attention: qkv_batch needs qkv_seq");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
#ifdef HAVE_CUDA
        if (backendId == DNN_BACKEND_CUDA) {
            // AttentionOp takes const float weight/bias blobs, the input and an optional additive
            // float mask; rotary stays on CPU
            Net::Impl* netimpl_ = getNetImpl(this);
            if (!netimpl_ || do_rotary || blobs.size() < 2 ||
                (this->inputs.size() != 1 && this->inputs.size() != 2))
                return false;
            // an intermediate arg has no type until the first forward; refuse only a known non-float one
            for (const Arg& a : this->inputs) {
                const int t = netimpl_->argData(a).type;
                if (t >= 0 && t != CV_32F)
                    return false;
            }
            return blobs.front().type() == CV_32F && blobs.back().type() == CV_32F &&
                   qkv_head_sizes[0] == qkv_head_sizes[1];
        }
#endif
        return backendId == DNN_BACKEND_OPENCV;
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void* context_,
                              InputArrayOfArrays,
                              InputArrayOfArrays) CV_OVERRIDE
    {
        auto context = reinterpret_cast<cuda4dnn::csl::CSLContext*>(context_);

        // finalize() has grouped the tokens and filled qkv_hidden_sizes[2]
        cuda4dnn::AttentionConfiguration cfg;
        cfg.num_heads = num_heads;
        cfg.qkv_hidden_sizes = qkv_hidden_sizes;
        cfg.input_hidden_size = input_hidden_size;
        cfg.batch_size = batch_size;
        cfg.seq_len = seq_len;
        cfg.scale = scale;

        return make_cuda_node<cuda4dnn::AttentionOp>(preferableTarget, std::move(context->stream),
                                                     std::move(context->cublas_handle),
                                                     blobs.front(), blobs.back(), cfg);
    }
#endif

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        int num_inputs = inputs.size() + blobs.size();
        CV_CheckGE(num_inputs, 3, "DNN/Attention: at least three inputs are required (data, weight, bias)");
        CV_CheckLE(num_inputs, 4, "DNN/Attention: at most four inputs are supported (data, weight, bias, mask)");
        const auto &input_shape = inputs[0];
        const auto &weight_shape = blobs.empty() ? inputs[1] : shape(blobs.front());
        const auto &bias_shape = blobs.empty() ? inputs[2] : shape(blobs.back());

        CV_CheckGE(input_shape.size(), static_cast<size_t>(3),
                   "DNN/Attention: invalid input dimension");
        CV_CheckEQ(weight_shape.size(), static_cast<size_t>(2), "DNN/Attention: invalid weight dimension");
        const int nd = (int)input_shape.size();
        int batch = 0, seq = 0;
        groupTokens(input_shape, batch, seq);

        CV_CheckEQ(input_shape[nd - 1], weight_shape[0], "DNN/Attention: invalid input shape");
        CV_CheckEQ(weight_shape[1], bias_shape[0], "DNN/Attention: invalid weight or bias shape");

        const int v_hidden = weight_shape[1] - (int)qkv_hidden_sizes[0] - (int)qkv_hidden_sizes[1];
        CV_CheckGT(v_hidden, 0, "DNN/Attention: qkv_hidden_sizes exceed the weight width");
        CV_CheckEQ(v_hidden % (int)num_heads, 0, "DNN/Attention: v_hidden_size must be divisible by num_heads");

        // Each token gets v_hidden output values, which may differ from the input width.
        if (output_ndims == 3) {
            MatShape output_shape = input_shape;
            output_shape[nd - 1] = v_hidden;
            outputs.assign(1, output_shape);
        } else if (output_ndims == 2) {
            MatShape output_shape{batch * seq, v_hidden};
            outputs.assign(1, output_shape);
        } else {
            CV_Error(Error::StsBadArg, format("DNN/Attention: invalid output dimension %zu, valid value is 2 or 3", output_ndims));
        }

        const int batch_size_ = batch, seq_len_ = seq,
                  hidden_size_ = weight_shape.back(),
                  num_heads_ = static_cast<int>(num_heads),
                  v_head_size_ = v_hidden / num_heads_;

        MatShape gemm_buffer_shape{batch_size_, seq_len_, hidden_size_},
                 attention_prob_shape{batch_size_ * num_heads_, seq_len_, seq_len_},
                 output_buffer_shape{batch_size_ * num_heads_, seq_len_, v_head_size_};
        if (useFlashAttention(inputs.size(), batch_size_, seq_len_, v_head_size_))
        {
            // FlashAttention writes directly to the output and never uses either
            // dense workspace. Keep their slots so the RoPE table index is stable.
            attention_prob_shape = MatShape{1};
            output_buffer_shape = MatShape{1};
        }
        internals.assign(1, gemm_buffer_shape);
        internals.push_back(attention_prob_shape);
        internals.push_back(output_buffer_shape);

        if (do_rotary)
        {
            CV_Assert(qkv_head_sizes[0] == qkv_head_sizes[1]);
            const int d = qkv_head_sizes[0];
            CV_Assert(d % 2 == 0);
            // pick maximum of q and k head dim

            MatShape rotation_table_shape{seq_len_, d};
            internals.push_back(rotation_table_shape);
        }

        return false;
    }

    // Batch and sequence length the attention runs over. The tokens are contiguous, so any split
    // of them recorded by the fusion is a view; without a record the input must be [B, S, C].
    void groupTokens(const MatShape& input_shape, int& batch, int& seq) const
    {
        const int nd = (int)input_shape.size();
        if (qkv_batch.empty()) {
            CV_CheckEQ(nd, 3, "DNN/Attention: input must be [batch, sequence, hidden]");
            batch = input_shape[0];
            seq = input_shape[1];
            return;
        }
        const int tokens = (int)total(input_shape, 0, nd - 1);
        auto eval = [&](const std::vector<int>& expr) {
            int v = expr[0];
            for (size_t i = 1; i < expr.size() && v >= 0; i++) {
                const int d = expr[i] < 0 ? expr[i] + nd : expr[i];
                CV_CheckGE(d, 0, "DNN/Attention: QKV reshape refers to a missing input dim");
                CV_CheckLT(d, nd - 1, "DNN/Attention: QKV reshape refers to a missing input dim");
                v *= input_shape[d];
            }
            return v;
        };
        batch = eval(qkv_batch);
        seq = eval(qkv_seq);
        if (batch < 0 && seq > 0) batch = tokens / seq;
        if (seq < 0 && batch > 0) seq = tokens / batch;
        CV_CheckGT(batch, 0, "DNN/Attention: invalid QKV reshape");
        CV_CheckEQ(batch * seq, tokens,
                   "DNN/Attention: QKV reshape does not match the token count");
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        const auto &input_shape = inputs[0];
        const int nd = (int)input_shape.size();
        int batch = 0, seq = 0;
        groupTokens(input_shape, batch, seq);
        int64 B = batch;
        int64 S = seq;
        int64 D = input_shape[nd - 1];

        const auto &weight_shape = blobs.empty() ? inputs[1] : shape(blobs.front());
        int64 hidden = weight_shape[1];

        int64 q_size = (int64)qkv_hidden_sizes[0];
        int64 k_size = (int64)qkv_hidden_sizes[1];
        int64 v_size = hidden - q_size - k_size;
        int64 q_head = (int64)qkv_head_sizes[0];
        int64 v_head = v_size / (int64)num_heads;

        // Input projection: Q, K, V = input * W + b
        int64 flops = B * S * (CV_BIG_INT(2) * D * hidden);
        // QK^T: (B*num_heads) * S * S * q_head_size
        flops += B * (int64)num_heads * CV_BIG_INT(2) * S * S * q_head;
        // Softmax: ~4 ops per element
        flops += B * (int64)num_heads * 4 * S * S;
        // Attention * V: (B*num_heads) * S * v_head_size * S
        flops += B * (int64)num_heads * CV_BIG_INT(2) * S * v_head * S;
        return flops;
    }

    // Geometry derives from the weight shape alone, so this is safe before finalize().
    void packQKV(const Mat& weight) {
        opt.init();
        const auto weight_shape = shape(weight);
        input_hidden_size = static_cast<size_t>(weight_shape[0]);
        hidden_size = weight_shape[1];
        qkv_hidden_sizes[2] = hidden_size - qkv_hidden_sizes[0] - qkv_hidden_sizes[1];
        qkv_head_sizes[2] = static_cast<size_t>(qkv_hidden_sizes[2] / num_heads);

        const auto *weight_data = weight.ptr<const float>();
#ifdef HAVE_MLAS
        // One MLAS-packed [input_hidden_size, head_size] slice per (Q/K/V, head).
        if (mlasAvailable() && !opt.use_rvv) {
            const size_t col_ofs[3] = {0, qkv_hidden_sizes[0], qkv_hidden_sizes[0] + qkv_hidden_sizes[1]};
            size_t total = 0;
            bool sizes_ok = true;
            for (int t = 0; t < 3; t++) {
                packed_qkv_ofs[t] = total;
                packed_qkv_step[t] = alignSize(mlasSgemmPackBSize(false, false, (int)qkv_head_sizes[t],
                                                                  (int)input_hidden_size), 64);
                sizes_ok = sizes_ok && packed_qkv_step[t] > 0;
                total += packed_qkv_step[t] * num_heads;
            }
            if (sizes_ok && total <= (size_t)INT_MAX) {
                packed_qkv_mlas.create(1, (int)total, CV_8U);
                bool ok = true;
                for (int t = 0; t < 3 && ok; t++)
                    for (size_t h = 0; h < num_heads && ok; h++)
                        ok = mlasSgemmPackB(false, false, (int)qkv_head_sizes[t], (int)input_hidden_size,
                                            weight_data + col_ofs[t] + h * qkv_head_sizes[t], (int)hidden_size,
                                            packed_qkv_mlas.ptr() + packed_qkv_ofs[t] + h * packed_qkv_step[t]);
                if (ok) {
                    packed_weight_q.clear();
                    packed_weight_k.clear();
                    packed_weight_v.clear();
                    return;
                }
            }
        }
        packed_qkv_mlas.release();
#endif
        packWeight(num_heads, qkv_head_sizes[0], input_hidden_size, weight_data,                                             hidden_size, packed_weight_q, opt);
        packWeight(num_heads, qkv_head_sizes[1], input_hidden_size, weight_data + qkv_hidden_sizes[0],                       hidden_size, packed_weight_k, opt);
        packWeight(num_heads, qkv_head_sizes[2], input_hidden_size, weight_data + qkv_hidden_sizes[0] + qkv_hidden_sizes[1], hidden_size, packed_weight_v, opt);
    }

    bool qkvPacked() const {
        return !packed_weight_q.empty() || !packed_qkv_mlas.empty();
    }

    // Dynamic weights (blobs empty) aren't available yet; forward() packs those.
    void prepackWeights() CV_OVERRIDE {
        if (!blobs.empty())
            packQKV(blobs.front());
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        opt.init();

        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        const auto input_shape = shape(inputs[0]);
        const int nd = (int)input_shape.size();
        int batch = 0, seq = 0;
        groupTokens(input_shape, batch, seq);
        batch_size = static_cast<size_t>(batch);
        seq_len = static_cast<size_t>(seq);
        input_hidden_size = static_cast<size_t>(input_shape[nd - 1]);

        const auto &weight = blobs.empty() ? inputs[1] : blobs.front();
        const auto weight_shape = shape(weight);
        hidden_size = weight_shape[1];
        qkv_hidden_sizes[2] = hidden_size - qkv_hidden_sizes[0] - qkv_hidden_sizes[1];
        qkv_head_sizes[2] = static_cast<size_t>(qkv_hidden_sizes[2] / num_heads);
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

        // Dynamic weight (blobs empty) may change each call, so repack it.
        if (blobs.empty())
            packQKV(inputs[1]);
        // Const weight: pack once as fallback when prepackWeights() wasn't called.
        else if (!qkvPacked())
            packQKV(blobs.front());

        float *packed_weights[3] = {packed_weight_q.data(), packed_weight_k.data(), packed_weight_v.data()};
        size_t packed_weights_size[3] = {packed_weight_q.size() / num_heads, packed_weight_k.size() / num_heads, packed_weight_v.size() / num_heads};
        CV_Assert(internals.size() == 3 + (do_rotary ? 1 : 0));

        if (do_rotary)
        {
            auto &rope_table = internals.back();
            auto *rope_table_data = rope_table.ptr<float>();
            // RoPE currently requires q and k head sizes to match.
            CV_Assert(qkv_head_sizes[0] == qkv_head_sizes[1]);
            precompRotationTable(rope_table_data, seq_len, qkv_head_sizes[0]);
        }

        auto &gemm_buffer = internals[0];
        auto *Q = gemm_buffer.ptr<float>();
        auto *K = Q + batch_size * seq_len * qkv_hidden_sizes[0];
        auto *V = K + batch_size * seq_len * qkv_hidden_sizes[1];
        float *QKV[3] = {Q, K, V}; // [B, N, S, H] per tensor
        {
            const auto &input = inputs[0];
            const auto &bias = blobs.empty() ? inputs[2] : blobs.back();
            const auto *input_data = input.ptr<const float>();
            const auto *bias_data = bias.ptr<const float>();

            // When do_rotary is false, internals.back() aliases output_buffer; harmless because rope_table is then unused.
            const auto &rope_table = internals.back();

            // Task i projects head (i / 3) % num_heads of batch (i / 3) / num_heads onto Q, K or V (i % 3).
            const size_t loops = 3 * batch_size * num_heads;
            auto taskInput = [&](size_t i) {
                return input_data + ((i / 3) / num_heads) * seq_len * input_hidden_size;
            };
            auto taskOutput = [&](size_t i) {
                return QKV[i % 3] + (i / 3) * seq_len * qkv_head_sizes[i % 3];
            };
            const size_t col_ofs[3] = {0, qkv_hidden_sizes[0], qkv_hidden_sizes[0] + qkv_hidden_sizes[1]};
            // Broadcast bias [NH] -> [BN, SH] into dst before gemm so beta=1 adds it implicitly.
            auto fillBias = [&](size_t i) {
                const size_t head_size = qkv_head_sizes[i % 3];
                const float *src = bias_data + col_ofs[i % 3] + ((i / 3) % num_heads) * head_size;
                float *dst = taskOutput(i);
                for (size_t s = 0; s < seq_len; s++, dst += head_size)
                    std::memcpy(dst, src, head_size * sizeof(float));
            };
            // Apply RoPE to Q/K in place (V is left untouched).
            auto applyRotary = [&](size_t i) {
                if (do_rotary && i % 3 < 2)
                    rotationKernel(taskOutput(i), rope_table.ptr<const float>(), seq_len, qkv_head_sizes[i % 3]);
            };

            opt.multi_thread = false;
            bool projected = false;
#ifdef HAVE_MLAS
            if (!packed_qkv_mlas.empty()) {
                parallel_for_(Range(0, (int)loops), [&](const Range &r) {
                    for (int i = r.start; i < r.end; i++)
                        fillBias(i);
                });
                // All per-head GEMMs in one MLAS batch, which also splits each GEMM across threads.
                // Q, K and V go in separate batches only if their head sizes differ.
                const bool same_size = qkv_head_sizes[0] == qkv_head_sizes[1] && qkv_head_sizes[1] == qkv_head_sizes[2];
                std::vector<const float*> A;
                std::vector<const void*> B;
                std::vector<float*> C;
                for (size_t t = 0; t < (same_size ? 1u : 3u); t++) {
                    A.clear(); B.clear(); C.clear();
                    for (size_t i = 0; i < loops; i++) {
                        if (!same_size && i % 3 != t)
                            continue;
                        A.push_back(taskInput(i));
                        B.push_back(packed_qkv_mlas.ptr() + packed_qkv_ofs[i % 3] + ((i / 3) % num_heads) * packed_qkv_step[i % 3]);
                        C.push_back(taskOutput(i));
                    }
                    const int N = (int)qkv_head_sizes[t];
                    projected = mlasSgemmPackedBatch(A.size(), (int)seq_len, N, (int)input_hidden_size, 1.f,
                                                     A.data(), (int)input_hidden_size, B.data(), 1.f, C.data(), N);
                    CV_Assert(projected);
                }
                if (do_rotary) {
                    parallel_for_(Range(0, (int)loops), [&](const Range &r) {
                        for (int i = r.start; i < r.end; i++)
                            applyRotary(i);
                    });
                }
            }
#endif
            if (!projected) {
                double nstripes = loops * seq_len * qkv_head_sizes[0] * input_hidden_size * (1 / 1024.0);
                parallel_for_(Range(0, (int)loops), [&](const Range &r) {
                    for (int i = r.start; i < r.end; i++) {
                        fillBias(i);
                        const size_t head_size = qkv_head_sizes[i % 3];
                        auto *packed_weight = packed_weights[i % 3] + packed_weights_size[i % 3] * ((i / 3) % num_heads);
                        // Single-threaded gemm; outer parallel_for_ already partitions over (batch, head, qkv).
                        fastGemm(false, seq_len, head_size, input_hidden_size,
                                1.f, taskInput(i), input_hidden_size,
                                packed_weight, 1.f, taskOutput(i), head_size, opt);
                        applyRotary(i);
                    }
                }, nstripes);
            }
        }

        if (useFlashAttention(inputs.size(), (int)batch_size, (int)seq_len, (int)qkv_head_sizes[2]))
        {
            const int B   = (int)batch_size;
            const int H   = (int)num_heads;
            const int S   = (int)seq_len;
            const int Dqk = (int)qkv_head_sizes[0];
            const int Dv  = (int)qkv_head_sizes[2];
            // ORT-style default tiling, clamped to the actual sequence length.
            const int q_block  = std::min(256, S);
            const int kv_block = std::min(256, S);
            const int threads  = std::max(1, cv::getNumThreads());

            const size_t per_thread =
                mlasFlashAttentionBufferBytesPerThread(q_block, kv_block, Dv);
            flash_scratch.resize((size_t)threads * per_thread);

            if (mlasFlashAttention(Q, K, V,
                                    outputs[0].ptr<float>(),
                                    B, H, S, S, Dqk, Dv, scale,
                                    q_block, kv_block,
                                    flash_scratch.data(), threads))
            {
                return;  // fast path done; outputs[0] is fully written
            }
        }

        // Compute Softmax(scale * MatMul(Q, K))
        auto &attention_prob = internals[1];
        // getMemoryShapes() plans dense buffers whenever FlashAttention is not used
        CV_CheckEQ(attention_prob.total(), (size_t)(batch_size * num_heads * seq_len * seq_len), "");
        {
            auto *output = attention_prob.ptr<float>();

            auto loops = batch_size * num_heads;
            auto seq_len_square = seq_len * seq_len;
            auto qk_head_size = qkv_head_sizes[0];
            auto qk_inner_size = seq_len * qk_head_size;

            // One batched fastGemmBatch call over all (batch, head) pairs — wrapping per-head fastGemm in parallel_for_ caused nested parallelism (inner MLAS gemm issued its own parallel_for_, serializing on most threads).
            std::vector<size_t> qk_a_offs(loops), qk_b_offs(loops), qk_c_offs(loops);
            for (int i = 0; i < (int)loops; i++) {
                qk_a_offs[i] = (size_t)i * qk_inner_size;
                qk_b_offs[i] = (size_t)i * qk_inner_size;
                qk_c_offs[i] = (size_t)i * seq_len_square;
            }
            opt.multi_thread = true;
            // ldb0=1, ldb1=qk_head_size signals trans_b for K (stored as [seq_len, qk_head_size]) to fastGemmBatch.
            fastGemmBatch(loops, qk_a_offs.data(), qk_b_offs.data(), qk_c_offs.data(),
                          (int)seq_len, (int)seq_len, (int)qk_head_size,
                          scale, Q, (int)qk_head_size, 1,
                          K, 1, (int)qk_head_size, 0.f,
                          output, (int)seq_len, opt);

            // Additive mask broadcast-aligned right to [B,H,S,S] (size-1 dims broadcast); only present when the fusion pass attached an external mask input.
            int num_non_blob_inputs = (int)inputs.size();
            bool has_mask = (!blobs.empty() && num_non_blob_inputs >= 2) ||
                            (blobs.empty() && num_non_blob_inputs >= 4);
            if (has_mask) {
                const Mat &mask_mat = blobs.empty() ? inputs[3] : inputs[1];
                CV_CheckTypeEQ(mask_mat.type(), CV_32F,
                               "DNN/Attention: mask must be float32");

                const auto mask_shape = shape(mask_mat);
                const int mask_ndim = mask_shape.dims;
                auto fmt_shape = [&]() {
                    std::ostringstream ss;
                    for (int d = 0; d < mask_ndim; d++) { if (d) ss << ","; ss << mask_shape[d]; }
                    return ss.str();
                };
                // Right-align mask dims to (B, H, Q=S, K=S); missing leading dims default to 1.
                auto get_dim = [&](int t) -> int {
                    int md = mask_ndim - 4 + t;
                    return (md >= 0) ? mask_shape[md] : 1;
                };
                // Extra leading dims beyond rank 4 must all be 1 (no broadcasting beyond [B,H,S,S]).
                for (int d = 0; d < mask_ndim - 4; d++) {
                    if (mask_shape[d] != 1)
                        CV_Error(Error::StsNotImplemented,
                                 cv::format("DNN/Attention: unsupported mask shape [%s] (B=%d, H=%d, S=%d)",
                                            fmt_shape().c_str(), (int)batch_size, (int)num_heads, (int)seq_len));
                }
                const int dim_b = get_dim(0);
                const int dim_h = get_dim(1);
                const int dim_q = get_dim(2);
                const int dim_k = get_dim(3);
                auto check_bcast = [&](int md, int td, const char* axis) {
                    if (md != 1 && md != td)
                        CV_Error(Error::StsNotImplemented,
                                 cv::format("DNN/Attention: mask shape [%s] not broadcastable to [%d,%d,%d,%d] (axis %s)",
                                            fmt_shape().c_str(), (int)batch_size, (int)num_heads,
                                            (int)seq_len, (int)seq_len, axis));
                };
                check_bcast(dim_b, (int)batch_size, "batch");
                check_bcast(dim_h, (int)num_heads, "head");
                check_bcast(dim_q, (int)seq_len,   "query");
                check_bcast(dim_k, (int)seq_len,   "key");

                const size_t mask_b_stride = (dim_b == 1) ? 0 : (size_t)dim_h * dim_q * dim_k;
                const size_t mask_h_stride = (dim_h == 1) ? 0 : (size_t)dim_q * dim_k;
                const size_t mask_q_stride = (dim_q == 1) ? 0 : (size_t)dim_k;
                const bool   mask_k_bcast  = (dim_k == 1);

                const float *mask_data = mask_mat.ptr<const float>();
                parallel_for_(Range(0, (int)loops), [&](const Range &r) {
                    for (int i = r.start; i < r.end; i++) {
                        const int b = i / (int)num_heads;
                        const int h = i % (int)num_heads;
                        const float *m_bh = mask_data + b * mask_b_stride + h * mask_h_stride;
                        float *prob = output + i * seq_len_square;
                        for (size_t row = 0; row < seq_len; row++) {
                            const float *m = m_bh + row * mask_q_stride;
                            float *p = prob + row * seq_len;
                            if (mask_k_bcast) {
                                float v = m[0];
                                for (size_t j = 0; j < seq_len; j++) p[j] += v;
                            } else {
                                for (size_t j = 0; j < seq_len; j++) p[j] += m[j];
                            }
                        }
                    }
                }, loops * seq_len * (1 / 1024.0));
            }

            softmax(attention_prob, attention_prob, shape(attention_prob).size() - 1);
        }

        // output = attention_prob @ V, then transpose back to [B, S, H*D]
        auto &output_buffer = internals[2];
        CV_CheckEQ(output_buffer.total(), (size_t)(batch_size * num_heads * seq_len * qkv_head_sizes[2]), "");
        {
            auto *output = outputs[0].ptr<float>();
            auto *output_buff = output_buffer.ptr<float>();
            const auto *prob = attention_prob.ptr<const float>();

            auto loops = batch_size * num_heads;
            auto prob_inner_size = seq_len * seq_len;
            auto v_head_size = qkv_head_sizes[2];
            auto v_inner_size = seq_len * v_head_size;

            // Batched fastGemmBatch over (batch, head) — same nested-parallelism rationale as QK^T above.
            std::vector<size_t> av_a_offs(loops), av_b_offs(loops), av_c_offs(loops);
            for (int i = 0; i < (int)loops; i++) {
                av_a_offs[i] = (size_t)i * prob_inner_size;
                av_b_offs[i] = (size_t)i * v_inner_size;
                av_c_offs[i] = (size_t)i * v_inner_size;
            }
            opt.multi_thread = true;
            fastGemmBatch(loops, av_a_offs.data(), av_b_offs.data(), av_c_offs.data(),
                          (int)seq_len, (int)v_head_size, (int)seq_len,
                          1.f, prob, (int)seq_len, 1,
                          V, (int)v_head_size, 1, 0.f,
                          output_buff, (int)v_head_size, opt);

            // Transpose [B*H, S, D] -> [B, S, H*D] in place via per-(batch, head) memcpy strips.
            parallel_for_(Range(0, (int)loops), [&] (const Range &r) {
                for (int i = r.start; i < r.end; i++) {
                    const int output_offset = i * (int)v_inner_size;
                    const int batch_index = static_cast<int>(i / num_heads);
                    const int head_index  = static_cast<int>(i % num_heads);
                    const float *src = output_buff + output_offset;
                    float *dst = output + (batch_index * (int)seq_len * (int)num_heads + head_index) * (int)v_head_size;
                    for (int j = 0; j < (int)seq_len; j++) {
                        std::memcpy(dst, src, v_head_size * sizeof(float));
                        src += v_head_size;
                        dst += qkv_hidden_sizes[2];
                    }
                }
            }, loops * seq_len * v_head_size * (1 / 1024.0));
        }
    }

 private:
    bool useFlashAttention(size_t inputCount, int batch, int sequence, int valueHeadSize) const
    {
        const bool hasMask = (!blobs.empty() && inputCount >= 2) ||
                             (blobs.empty() && inputCount >= 4);
        return mlasAvailable() && !hasMask && batch > 0 && num_heads > 0 && sequence > 0 &&
               qkv_head_sizes[0] > 0 && valueHeadSize > 0;
    }

    size_t num_heads;
    std::vector<size_t> qkv_hidden_sizes; // order: {qk_hidden_size, qk_hidden_size, v_hidden_size}
    float scale;
    size_t output_ndims;

    std::vector<size_t> qkv_head_sizes; // order: {qk_head_size, qk_head_size, v_head_size}

    size_t batch_size;
    size_t seq_len;
    size_t input_hidden_size;
    size_t hidden_size;

    bool do_rotary;
    std::vector<int> qkv_batch, qkv_seq;  // token grouping from the attention fusion
    std::vector<float> packed_weight_q;
    std::vector<float> packed_weight_k;
    std::vector<float> packed_weight_v;
    Mat packed_qkv_mlas;  // used instead of packed_weight_q/k/v when MLAS is available
    size_t packed_qkv_ofs[3] = {0, 0, 0}, packed_qkv_step[3] = {0, 0, 0};
    std::vector<unsigned char> flash_scratch;

    FastGemmOpt opt;
};

Ptr<AttentionLayer> AttentionLayer::create(const LayerParams &params) {
    return makePtr<AttentionLayerImpl>(params);
}

}} // cv::dnn
