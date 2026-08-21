// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include <cmath>

namespace cv {
namespace dnn {

/*
    Implementation of LinearAttention, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__LinearAttention.html

    Opset 27 is covered.
*/
// Recurrent (O(T)) attention over a fixed-size state S of shape [Dk, Dv], maintained
// per (batch, kv-head). For each time step t the state is updated and read out:
//   linear      : S += k_t (x) v_t
//   delta       : S += k_t (x) (beta_t * (v_t - S^T k_t))
//   gated       : S  = S * exp(decay_t); S += k_t (x) v_t
//   gated_delta : S  = S * exp(decay_t); S += k_t (x) (beta_t * (v_t - S^T k_t))   (default)
//   out_t = scale * (q_t . S)
// GQA/MQA: q has q_num_heads, the state has kv_num_heads; the state is shared across
// the q_num_heads / kv_num_heads queries in each group.
// chunk_size is a prefill tuning hint only; ignored.

class LinearAttentionLayerImpl CV_FINAL : public LinearAttentionLayer
{
public:
    LinearAttentionLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        q_num_heads  = params.get<int>("q_num_heads", 0);
        kv_num_heads = params.get<int>("kv_num_heads", 0);
        scale        = params.get<float>("scale", 0.f);
        // Spec: scale 0.0 means "derive 1/sqrt(d_k)", so test the value, not presence.
        has_scale    = (scale != 0.f);
        update_rule  = params.get<std::string>("update_rule", "gated_delta");
        CV_Check(update_rule, update_rule == "linear" || update_rule == "gated" ||
                              update_rule == "delta"  || update_rule == "gated_delta",
                 "LinearAttention: unknown update_rule");
        CV_CheckGT(q_num_heads, 0, "LinearAttention: q_num_heads is required");
        CV_CheckGT(kv_num_heads, 0, "LinearAttention: kv_num_heads is required");
        CV_CheckEQ(q_num_heads % kv_num_heads, 0, "LinearAttention: q_num_heads must be divisible by kv_num_heads");
        use_decay = (update_rule == "gated" || update_rule == "gated_delta");
        use_delta = (update_rule == "delta" || update_rule == "gated_delta");
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    // Optional inputs arrive as empty Mats in fixed slots: 3=past_state, 4=decay, 5=beta.
    static bool present(const std::vector<MatShape>& in, size_t i) { return in.size() > i && in[i].dims > 0; }
    static bool present(const std::vector<Mat>& in, size_t i)      { return in.size() > i && !in[i].empty(); }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int /*requiredInternals*/,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_CheckGE(inputs.size(), (size_t)3, "LinearAttention needs query, key, value");
        CV_CheckType(inputs[0], inputs[0] == CV_32F || inputs[0] == CV_16F, "LinearAttention: only FP32/FP16 are supported");
        outputs.assign(requiredOutputs, inputs[0]);
        internals.clear();
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int /*requiredOutputs*/,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        CV_CheckGE(inputs.size(), (size_t)3, "LinearAttention needs query, key, value");
        CV_CheckEQ(inputs[0].dims, 3, "LinearAttention: query must be 3D [batch, seq, q_num_heads*head_size]");

        const int batch = inputs[0][0];
        const int seq   = inputs[0][1];
        const int dk    = inputs[0][2] / q_num_heads;
        const int dv    = inputs[2][2] / kv_num_heads;
        CV_CheckEQ(inputs[1][2] / kv_num_heads, dk, "LinearAttention: key head_size must equal query head_size");

        outputs.assign(1, MatShape{batch, seq, q_num_heads * dv});   // output
        outputs.push_back(MatShape{batch, kv_num_heads, dk, dv});    // present_state
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays /*internals_arr*/) CV_OVERRIDE
    {
        std::vector<Mat> rawInputs, rawOutputs;
        inputs_arr.getMatVector(rawInputs);
        outputs_arr.getMatVector(rawOutputs);

        // Compute in fp32; convert fp16 tensors in/out (the no-op forward_fallback path is unusable here).
        const bool is_fp16 = rawInputs[0].depth() == CV_16F;
        std::vector<Mat> in32, out32;
        if (is_fp16)
        {
            in32.resize(rawInputs.size());
            for (size_t i = 0; i < rawInputs.size(); ++i)
                if (!rawInputs[i].empty()) rawInputs[i].convertTo(in32[i], CV_32F);
            out32.resize(rawOutputs.size());
            for (size_t i = 0; i < rawOutputs.size(); ++i)
                out32[i].create(rawOutputs[i].dims, rawOutputs[i].size.p, CV_32F);
        }
        std::vector<Mat>& inputs  = is_fp16 ? in32  : rawInputs;
        std::vector<Mat>& outputs = is_fp16 ? out32 : rawOutputs;

        const Mat& query = inputs[0];
        const Mat& key   = inputs[1];
        const Mat& value = inputs[2];
        const bool has_past  = present(inputs, 3);
        const bool has_decay = present(inputs, 4) && use_decay;
        const bool has_beta  = present(inputs, 5) && use_delta;

        const int batch = query.size[0];
        const int seq   = query.size[1];
        const int Hq    = q_num_heads;
        const int Hkv   = kv_num_heads;
        const int group = Hq / Hkv;
        const int Dk    = query.size[2] / Hq;
        const int Dv    = value.size[2] / Hkv;

        const float scl = has_scale ? scale : 1.0f / std::sqrt(static_cast<float>(Dk));

        // decay: per-kv-head vector of length Dk, or a single value broadcast over Dk (per_head_decay).
        const int decayDim = has_decay ? inputs[4].size[2] / Hkv : 0;
        // beta: one scalar per kv-head, or one scalar broadcast over all heads.
        const bool betaPerHead = has_beta && inputs[5].size[2] >= Hkv;

        const float* Qp = query.ptr<float>();
        const float* Kp = key.ptr<float>();
        const float* Vp = value.ptr<float>();
        const float* Dp = has_decay ? inputs[4].ptr<float>() : nullptr;
        const float* Bp = has_beta  ? inputs[5].ptr<float>() : nullptr;
        const float* Pp = has_past  ? inputs[3].ptr<float>() : nullptr;

        float* Op = outputs[0].ptr<float>();
        float* Sp = outputs[1].ptr<float>();

        const size_t qStride = (size_t)seq * Hq * Dk;   // per batch
        const size_t kStride = (size_t)seq * Hkv * Dk;
        const size_t vStride = (size_t)seq * Hkv * Dv;
        const size_t oStride = (size_t)seq * Hq * Dv;
        const size_t stateSz = (size_t)Dk * Dv;

        parallel_for_(Range(0, batch * Hkv), [&](const Range& r)
        {
            std::vector<float> retrieved(Dv);
            for (int bh = r.start; bh < r.end; ++bh)
            {
                const int b = bh / Hkv;
                const int h = bh % Hkv;

                // state[i*Dv + j], initialised from past_state or zeros.
                float* S = Sp + (size_t)bh * stateSz;
                if (has_past)
                    std::memcpy(S, Pp + (size_t)bh * stateSz, stateSz * sizeof(float));
                else
                    std::memset(S, 0, stateSz * sizeof(float));

                for (int t = 0; t < seq; ++t)
                {
                    const float* k_t = Kp + b * kStride + (size_t)t * Hkv * Dk + (size_t)h * Dk;
                    const float* v_t = Vp + b * vStride + (size_t)t * Hkv * Dv + (size_t)h * Dv;

                    // 1) forget gate: S[i,:] *= exp(decay_i). Resolve the per-head-scalar vs
                    //    per-Dk-vector layout outside the i-loop so it stays branchless.
                    if (has_decay)
                    {
                        const float* d_t = Dp + b * (size_t)seq * Hkv * decayDim + (size_t)t * Hkv * decayDim + (size_t)h * decayDim;
                        if (decayDim == 1)
                        {
                            const float gate = std::exp(d_t[0]);
                            for (int i = 0; i < Dk; ++i)
                            {
                                float* Si = S + (size_t)i * Dv;
                                for (int j = 0; j < Dv; ++j) Si[j] *= gate;
                            }
                        }
                        else
                        {
                            for (int i = 0; i < Dk; ++i)
                            {
                                const float gate = std::exp(d_t[i]);
                                float* Si = S + (size_t)i * Dv;
                                for (int j = 0; j < Dv; ++j) Si[j] *= gate;
                            }
                        }
                    }

                    // 2) write: outer-product update, optionally delta-corrected
                    if (has_beta)
                    {
                        // retrieved = S^T k_t   ([Dv])
                        for (int j = 0; j < Dv; ++j) retrieved[j] = 0.f;
                        for (int i = 0; i < Dk; ++i)
                        {
                            const float ki = k_t[i];
                            const float* Si = S + (size_t)i * Dv;
                            for (int j = 0; j < Dv; ++j) retrieved[j] += Si[j] * ki;
                        }
                        const float beta = Bp[b * (size_t)seq * (betaPerHead ? Hkv : 1) + (size_t)t * (betaPerHead ? Hkv : 1) + (betaPerHead ? h : 0)];
                        for (int i = 0; i < Dk; ++i)
                        {
                            const float ki = k_t[i];
                            float* Si = S + (size_t)i * Dv;
                            for (int j = 0; j < Dv; ++j)
                                Si[j] += ki * (beta * (v_t[j] - retrieved[j]));
                        }
                    }
                    else
                    {
                        for (int i = 0; i < Dk; ++i)
                        {
                            const float ki = k_t[i];
                            float* Si = S + (size_t)i * Dv;
                            for (int j = 0; j < Dv; ++j) Si[j] += ki * v_t[j];
                        }
                    }

                    // 3) read-out for every query head in this kv group: out = scale * (q_t . S)
                    for (int g = 0; g < group; ++g)
                    {
                        const int n = h * group + g;
                        const float* q_t = Qp + b * qStride + (size_t)t * Hq * Dk + (size_t)n * Dk;
                        float* o_t = Op + b * oStride + (size_t)t * Hq * Dv + (size_t)n * Dv;
                        for (int j = 0; j < Dv; ++j) o_t[j] = 0.f;
                        for (int i = 0; i < Dk; ++i)
                        {
                            const float qi = q_t[i];
                            const float* Si = S + (size_t)i * Dv;
                            for (int j = 0; j < Dv; ++j) o_t[j] += qi * Si[j];
                        }
                        for (int j = 0; j < Dv; ++j) o_t[j] *= scl;
                    }
                }
            }
        });

        if (is_fp16)
            for (size_t i = 0; i < rawOutputs.size(); ++i)
                out32[i].convertTo(rawOutputs[i], CV_16F);
    }

    int64 getFLOPS(const std::vector<MatShape>& inputs, const std::vector<MatShape>& /*outputs*/) const CV_OVERRIDE
    {
        const int64 batch = inputs[0][0], seq = inputs[0][1];
        const int64 Dk = inputs[0][2] / q_num_heads, Dv = inputs[2][2] / kv_num_heads;
        // per step, per kv-head: ~2 outer products (write) + one read per q head
        int64 perStep = kv_num_heads * CV_BIG_INT(4) * Dk * Dv + q_num_heads * CV_BIG_INT(2) * Dk * Dv;
        return batch * seq * perStep;
    }

private:
    int q_num_heads = 0;
    int kv_num_heads = 0;
    float scale = 0.f;
    bool has_scale = false;
    std::string update_rule;
    bool use_decay = false;
    bool use_delta = false;
};

Ptr<LinearAttentionLayer> LinearAttentionLayer::create(const LayerParams& params)
{
    return makePtr<LinearAttentionLayerImpl>(params);
}

}} // namespace cv::dnn
