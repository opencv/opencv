// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "cpu_kernels/mlas_gemm.hpp"

namespace cv { namespace dnn {

// Scaled-Dot-Product Attention on pre-projected Q, K^T, V. Fuses
// QK^T -> [scale] -> softmax -> ·V -> Transpose -> Reshape into one
// MlasFlashAttention call. Used by the SR-attention fusion pass.
class SDPALayerImpl CV_FINAL : public SDPALayer {
public:
    SDPALayerImpl(const LayerParams &params) {
        setParamsFrom(params);
        scale = params.get<float>("scale", 1.f);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), 3u, "DNN/SDPA: expects Q, K^T, V");
        const auto &q  = inputs[0];   // (B, H, S_q,  D)
        const auto &kT = inputs[1];   // (B, H, D,    S_kv)
        const auto &v  = inputs[2];   // (B, H, S_kv, D_v)
        CV_CheckEQ(q.dims, 4, "DNN/SDPA: Q must be 4D");
        CV_CheckEQ(kT.dims, 4, "DNN/SDPA: K must be 4D");
        CV_CheckEQ(v.dims, 4, "DNN/SDPA: V must be 4D");
        const int B  = q[0], H = q[1], S_q = q[2], D = q[3];
        const int S_kv = v[2], D_v = v[3];
        CV_CheckEQ(kT[0], B,    "DNN/SDPA: K batch");
        CV_CheckEQ(kT[1], H,    "DNN/SDPA: K heads");
        CV_CheckEQ(kT[2], D,    "DNN/SDPA: K^T inner dim must equal Q's head_dim");
        CV_CheckEQ(kT[3], S_kv, "DNN/SDPA: K^T outer dim must equal V's seq len");
        CV_CheckEQ(v[0], B,     "DNN/SDPA: V batch");
        CV_CheckEQ(v[1], H,     "DNN/SDPA: V heads");

        outputs.assign(1, MatShape{B, S_q, H * D_v});
        // Scratch for K^T re-transposed to (B, H, S_kv, D).
        internals.assign(1, MatShape{B * H * S_kv * D});
        return false;
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE {
        const auto &q = inputs[0];
        const auto &v = inputs[2];
        const int64 B = q[0], H = q[1], S_q = q[2], D = q[3];
        const int64 S_kv = v[2], D_v = v[3];
        return B * H * (CV_BIG_INT(2) * S_q * S_kv * D + 4 * S_q * S_kv
                        + CV_BIG_INT(2) * S_q * S_kv * D_v);
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        const Mat& Q  = inputs[0];   // (B, H, S_q,  D)
        const Mat& KT = inputs[1];   // (B, H, D,    S_kv)
        const Mat& V  = inputs[2];   // (B, H, S_kv, D_v)
        Mat&       Y  = outputs[0];  // (B, S_q, H*D_v)

        const int B   = Q.size[0];
        const int H   = Q.size[1];
        const int S_q = Q.size[2];
        const int D   = Q.size[3];
        const int S_kv = V.size[2];
        const int D_v  = V.size[3];

        // Re-transpose K^T (B,H,D,S_kv) -> K (B,H,S_kv,D) for MlasFlashAttention.
        Mat &K = internals[0];
        const float* kT_data = KT.ptr<const float>();
        float*       k_data  = K.ptr<float>();
        const size_t kT_bhd_stride = (size_t)D * S_kv;
        const size_t k_bhd_stride  = (size_t)S_kv * D;
        cv::parallel_for_(cv::Range(0, B * H), [&](const cv::Range& r) {
            for (int bh = r.start; bh < r.end; bh++) {
                const float* src = kT_data + (size_t)bh * kT_bhd_stride;
                float*       dst = k_data  + (size_t)bh * k_bhd_stride;
                for (int s = 0; s < S_kv; s++) {
                    for (int d = 0; d < D; d++) {
                        dst[(size_t)s * D + d] = src[(size_t)d * S_kv + s];
                    }
                }
            }
        });

        const int q_block  = std::min(256, S_q);
        const int kv_block = std::min(256, S_kv);
        const int threads  = std::max(1, cv::getNumThreads());

        const size_t per_thread =
            mlasFlashAttentionBufferBytesPerThread(q_block, kv_block, D_v);
        if (mlasAvailable() && per_thread > 0) {
            flash_scratch.resize((size_t)threads * per_thread);
            const float effective_scale = (scale > 0.f) ? (1.f / scale) : 1.f;
            if (mlasFlashAttention(Q.ptr<float>(), K.ptr<float>(), V.ptr<float>(),
                                   Y.ptr<float>(),
                                   B, H, S_q, S_kv, D, D_v, effective_scale,
                                   q_block, kv_block,
                                   flash_scratch.data(), threads))
            {
                return;
            }
        }

        forward_fallback_impl(Q, K, V, Y, B, H, S_q, S_kv, D, D_v);
    }

private:
    // Unfused SDPA, used when MLAS isn't available. Not perf-critical.
    void forward_fallback_impl(const Mat& Q, const Mat& K, const Mat& V, Mat& Y,
                               int B, int H, int S_q, int S_kv, int D, int D_v) const
    {
        const float* q_p = Q.ptr<const float>();
        const float* k_p = K.ptr<const float>();
        const float* v_p = V.ptr<const float>();
        float*       y_p = Y.ptr<float>();
        const float inv_scale = (scale > 0.f) ? (1.f / scale) : 1.f;

        cv::parallel_for_(cv::Range(0, B * H), [&](const cv::Range& r) {
            std::vector<float> attn((size_t)S_q * S_kv);
            for (int bh = r.start; bh < r.end; bh++) {
                const int b = bh / H, h = bh % H;
                const float* Q_bh = q_p + (size_t)bh * S_q  * D;
                const float* K_bh = k_p + (size_t)bh * S_kv * D;
                const float* V_bh = v_p + (size_t)bh * S_kv * D_v;
                float*       Y_b  = y_p + (size_t)b * S_q * H * D_v;

                for (int i = 0; i < S_q; i++) {
                    for (int j = 0; j < S_kv; j++) {
                        float s = 0.f;
                        for (int d = 0; d < D; d++)
                            s += Q_bh[(size_t)i * D + d] * K_bh[(size_t)j * D + d];
                        attn[(size_t)i * S_kv + j] = s * inv_scale;
                    }
                    float mx = attn[(size_t)i * S_kv];
                    for (int j = 1; j < S_kv; j++)
                        if (attn[(size_t)i * S_kv + j] > mx) mx = attn[(size_t)i * S_kv + j];
                    float sum = 0.f;
                    for (int j = 0; j < S_kv; j++) {
                        float e = std::exp(attn[(size_t)i * S_kv + j] - mx);
                        attn[(size_t)i * S_kv + j] = e;
                        sum += e;
                    }
                    float inv_sum = 1.f / sum;
                    for (int j = 0; j < S_kv; j++)
                        attn[(size_t)i * S_kv + j] *= inv_sum;

                    // out[b, i, h, :] = attn[i, :] @ V_bh
                    float* out_row = Y_b + ((size_t)i * H + h) * D_v;
                    for (int d = 0; d < D_v; d++) out_row[d] = 0.f;
                    for (int j = 0; j < S_kv; j++) {
                        float a = attn[(size_t)i * S_kv + j];
                        const float* V_row = V_bh + (size_t)j * D_v;
                        for (int d = 0; d < D_v; d++)
                            out_row[d] += a * V_row[d];
                    }
                }
            }
        });
    }

    float scale;
    std::vector<unsigned char> flash_scratch;
};

Ptr<SDPALayer> SDPALayer::create(const LayerParams &params) {
    return makePtr<SDPALayerImpl>(params);
}

}} // cv::dnn
