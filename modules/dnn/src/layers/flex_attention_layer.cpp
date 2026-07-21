// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "cpu_kernels/fast_gemm.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include <cmath>

namespace cv {
namespace dnn {

/*
    Implementation of FlexAttention (domain ai.onnx.preview, opset 1).
    Spec: https://github.com/onnx/onnx/blob/main/docs/Operators-preview.md#aionnxpreviewFlexAttention
    Supported: score_mod / prob_mod sub-graphs (inlined by the importer); mask_mod is
    rejected with StsNotImplemented.
*/
// Scaled dot-product attention with GQA and optional score_mod / prob_mod
// sub-graphs that rewrite the whole [B, H, L, S] score / probability tensor:
//   scores = scale * (Q . expand_kv(K)^T);  scores = score_mod(scores)
//   probs  = softmax(scores, axis=-1);       probs  = prob_mod(probs)
//   Y      = probs . expand_kv(V)
// The sub-graphs carry no control flow, so the importer inlines them as ordinary
// graph nodes and cuts this op into three stages executed by this layer:
//   "full" : Q, K, V -> Y            (no sub-graphs)
//   "qk"   : Q, K    -> scores       (feeds score_mod / Softmax)
//   "av"   : probs, V-> Y            (consumes prob_mod / Softmax output)

class FlexAttentionLayerImpl CV_FINAL : public FlexAttentionLayer
{
public:
    FlexAttentionLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        scale     = params.get<float>("scale", 0.f);
        has_scale = params.has("scale");
        stage     = params.get<std::string>("stage", "full");
        CV_Check(stage, stage == "full" || stage == "qk" || stage == "av", "FlexAttention: bad stage");
        opt.init();
    }

    bool supportBackend(int backendId) CV_OVERRIDE { return backendId == DNN_BACKEND_OPENCV; }

    void getTypes(const std::vector<MatType>& inputs, const int requiredOutputs, const int /*ri*/,
                  std::vector<MatType>& outputs, std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_CheckType(inputs[0], inputs[0] == CV_32F || inputs[0] == CV_64F || inputs[0] == CV_16F,
                     "FlexAttention: only FP32/FP64/FP16 are supported");
        outputs.assign(requiredOutputs, inputs[0]);
        internals.clear();
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs, const int /*ro*/,
                         std::vector<MatShape>& outputs, std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs[0].dims, 4, "FlexAttention: inputs must be 4D [batch, heads, seq, head_size]");
        const int B = inputs[0][0], Hq = inputs[0][1], L = inputs[0][2];
        if (stage == "qk")            // Q[B,Hq,L,D], K[B,Hkv,S,D] -> scores[B,Hq,L,S]
            outputs.assign(1, MatShape{B, Hq, L, inputs[1][2]});
        else if (stage == "av")       // probs[B,Hq,L,S], V[B,Hkv,S,Dv] -> Y[B,Hq,L,Dv]
            outputs.assign(1, MatShape{B, Hq, L, inputs[1][3]});
        else                          // Q,K,V -> Y[B,Hq,L,Dv]
            outputs.assign(1, MatShape{B, Hq, L, inputs[2][3]});
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays /*internals*/) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        if (inputs[0].depth() == CV_64F)
            run<double>(inputs, outputs, false);        // scalar path (fastGemmBatch is float-only)
        else if (inputs[0].depth() == CV_16F)
            runFloat(inputs, outputs, true);
        else
            runFloat(inputs, outputs, false);
    }

private:
    // fp32/fp16 path: batched GEMM via fastGemmBatch (MLAS-accelerated when built with
    // HAVE_MLAS). fp16 is computed in fp32. K/V are shared across each GQA group through the
    // per-head offset arithmetic (n -> n/group), matching the ONNX Attention layer.
    void runFloat(std::vector<Mat>& rawIn, std::vector<Mat>& rawOut, bool fp16)
    {
        std::vector<Mat> in, out;
        if (fp16)
        {
            in.resize(rawIn.size());
            for (size_t i = 0; i < rawIn.size(); ++i)
                if (!rawIn[i].empty()) rawIn[i].convertTo(in[i], CV_32F);
            out.resize(rawOut.size());
            out[0].create(rawOut[0].dims, rawOut[0].size.p, CV_32F);
        }
        std::vector<Mat>& I = fp16 ? in : rawIn;
        std::vector<Mat>& O = fp16 ? out : rawOut;

        if (stage == "qk")
            qkGemm(I[0], I[1], O[0]);
        else if (stage == "av")
            avGemm(I[0], I[1], O[0]);
        else
            fullFloat(I[0], I[1], I[2], O[0]);

        if (fp16) O[0].convertTo(rawOut[0], CV_16F);
    }

    // scores[B,Hq,L,Skv] = scl * Q * K^T (K^T via ldb0=1,ldb1=D).
    void qkGemm(const Mat& Q, const Mat& K, Mat& S)
    {
        const int B = Q.size[0], Hq = Q.size[1], L = Q.size[2], D = Q.size[3];
        const int Hkv = K.size[1], Skv = K.size[2], group = Hq / Hkv;
        const float scl = has_scale ? scale : (float)(1.0 / std::sqrt((double)D));

        const size_t batch = (size_t)B * Hq;
        std::vector<size_t> qo(batch), ko(batch), so(batch);
        for (int b = 0; b < B; ++b)
            for (int n = 0; n < Hq; ++n)
            {
                const size_t bn = (size_t)b * Hq + n;
                qo[bn] = ((size_t)b * Hq + n) * L * D;
                ko[bn] = ((size_t)b * Hkv + n / group) * Skv * D;
                so[bn] = ((size_t)b * Hq + n) * L * Skv;
            }
        fastGemmBatch(batch, qo.data(), ko.data(), so.data(),
                      L, Skv, D, scl, Q.ptr<float>(), D, 1, K.ptr<float>(), 1, D,
                      0.f, S.ptr<float>(), Skv, opt);
    }

    // Y[B,Hq,L,Dv] = P * V.
    void avGemm(const Mat& P, const Mat& V, Mat& Y)
    {
        const int B = P.size[0], Hq = P.size[1], L = P.size[2], Skv = P.size[3];
        const int Hkv = V.size[1], Dv = V.size[3], group = Hq / Hkv;

        const size_t batch = (size_t)B * Hq;
        std::vector<size_t> po(batch), vo(batch), yo(batch);
        for (int b = 0; b < B; ++b)
            for (int n = 0; n < Hq; ++n)
            {
                const size_t bn = (size_t)b * Hq + n;
                po[bn] = ((size_t)b * Hq + n) * L * Skv;
                vo[bn] = ((size_t)b * Hkv + n / group) * Skv * Dv;
                yo[bn] = ((size_t)b * Hq + n) * L * Dv;
            }
        fastGemmBatch(batch, po.data(), vo.data(), yo.data(),
                      L, Dv, Skv, 1.f, P.ptr<float>(), Skv, 1, V.ptr<float>(), Dv, 1,
                      0.f, Y.ptr<float>(), Dv, opt);
    }

    void fullFloat(const Mat& Q, const Mat& K, const Mat& V, Mat& Y)
    {
        const int B = Q.size[0], Hq = Q.size[1], L = Q.size[2], Skv = K.size[2];
        int ssz[4] = {B, Hq, L, Skv};
        Mat scores(4, ssz, CV_32F);
        qkGemm(Q, K, scores);
        softmaxLastDim(scores, (int)((size_t)B * Hq * L), Skv);
        avGemm(scores, V, Y);
    }

    static void softmaxLastDim(Mat& S, int rows, int cols)
    {
        float* base = S.ptr<float>();
        parallel_for_(Range(0, rows), [&](const Range& r) {
            for (int i = r.start; i < r.end; ++i)
            {
                float* row = base + (size_t)i * cols;
                float mx = row[0];
                for (int j = 1; j < cols; ++j) mx = std::max(mx, row[j]);
                float sum = 0.f;
                for (int j = 0; j < cols; ++j) { const float e = std::exp(row[j] - mx); row[j] = e; sum += e; }
                const float inv = sum > 0.f ? 1.f / sum : 0.f;
                for (int j = 0; j < cols; ++j) row[j] *= inv;
            }
        });
    }

    // ---- CV_64F reference path (scalar templates; fastGemmBatch is float-only) ----
    template<typename T>
    void run(std::vector<Mat>& rawIn, std::vector<Mat>& rawOut, bool fp16)
    {
        // fp16 is computed in fp32 with conversion in/out.
        std::vector<Mat> in, out;
        if (fp16)
        {
            in.resize(rawIn.size());
            for (size_t i = 0; i < rawIn.size(); ++i)
                if (!rawIn[i].empty()) rawIn[i].convertTo(in[i], CV_32F);
            out.resize(rawOut.size());
            out[0].create(rawOut[0].dims, rawOut[0].size.p, CV_32F);
        }
        std::vector<Mat>& I = fp16 ? in : rawIn;
        std::vector<Mat>& O = fp16 ? out : rawOut;

        if (stage == "qk")
            qk<T>(I[0], I[1], O[0]);
        else if (stage == "av")
            av<T>(I[0], I[1], O[0]);
        else
            full<T>(I[0], I[1], I[2], O[0]);

        if (fp16) O[0].convertTo(rawOut[0], CV_16F);
    }

    // scores[b,n,l,s] = scale * sum_d Q[b,n,l,d] * K[b, n/group, s, d]
    template<typename T>
    void qk(const Mat& Q, const Mat& K, Mat& S)
    {
        const int B = Q.size[0], Hq = Q.size[1], L = Q.size[2], D = Q.size[3];
        const int Hkv = K.size[1], Skv = K.size[2], group = Hq / Hkv;
        const T scl = static_cast<T>(has_scale ? scale : 1.0 / std::sqrt((double)D));
        parallel_for_(Range(0, B * Hq), [&](const Range& r) {
            for (int bn = r.start; bn < r.end; ++bn) {
                const int b = bn / Hq, n = bn % Hq, h = n / group;
                const T* q = Q.ptr<T>() + ((size_t)b * Hq + n) * L * D;
                const T* k = K.ptr<T>() + ((size_t)b * Hkv + h) * Skv * D;
                T* s = S.ptr<T>() + ((size_t)b * Hq + n) * L * Skv;
                for (int l = 0; l < L; ++l)
                    for (int j = 0; j < Skv; ++j) {
                        T acc = 0;
                        for (int d = 0; d < D; ++d) acc += q[l * D + d] * k[j * D + d];
                        s[l * Skv + j] = acc * scl;
                    }
            }
        });
    }

    // Y[b,n,l,:] = sum_s P[b,n,l,s] * V[b, n/group, s, :]
    template<typename T>
    void av(const Mat& P, const Mat& V, Mat& Y)
    {
        const int B = P.size[0], Hq = P.size[1], L = P.size[2], Skv = P.size[3];
        const int Hkv = V.size[1], Dv = V.size[3], group = Hq / Hkv;
        parallel_for_(Range(0, B * Hq), [&](const Range& r) {
            for (int bn = r.start; bn < r.end; ++bn) {
                const int b = bn / Hq, n = bn % Hq, h = n / group;
                const T* p = P.ptr<T>() + ((size_t)b * Hq + n) * L * Skv;
                const T* v = V.ptr<T>() + ((size_t)b * Hkv + h) * Skv * Dv;
                T* y = Y.ptr<T>() + ((size_t)b * Hq + n) * L * Dv;
                for (int l = 0; l < L; ++l) {
                    for (int c = 0; c < Dv; ++c) y[l * Dv + c] = 0;
                    for (int j = 0; j < Skv; ++j) {
                        const T pw = p[l * Skv + j];
                        for (int c = 0; c < Dv; ++c) y[l * Dv + c] += pw * v[j * Dv + c];
                    }
                }
            }
        });
    }

    template<typename T>
    void full(const Mat& Q, const Mat& K, const Mat& V, Mat& Y)
    {
        const int B = Q.size[0], Hq = Q.size[1], L = Q.size[2], D = Q.size[3];
        const int Hkv = K.size[1], Skv = K.size[2], Dv = V.size[3], group = Hq / Hkv;
        const T scl = static_cast<T>(has_scale ? scale : 1.0 / std::sqrt((double)D));
        parallel_for_(Range(0, B * Hq), [&](const Range& r) {
            std::vector<T> s(Skv);
            for (int bn = r.start; bn < r.end; ++bn) {
                const int b = bn / Hq, n = bn % Hq, h = n / group;
                const T* q = Q.ptr<T>() + ((size_t)b * Hq + n) * L * D;
                const T* k = K.ptr<T>() + ((size_t)b * Hkv + h) * Skv * D;
                const T* v = V.ptr<T>() + ((size_t)b * Hkv + h) * Skv * Dv;
                T* y = Y.ptr<T>() + ((size_t)b * Hq + n) * L * Dv;
                for (int l = 0; l < L; ++l) {
                    T mx = -std::numeric_limits<T>::infinity();
                    for (int j = 0; j < Skv; ++j) {
                        T acc = 0;
                        for (int d = 0; d < D; ++d) acc += q[l * D + d] * k[j * D + d];
                        s[j] = acc * scl;
                        mx = std::max(mx, s[j]);
                    }
                    T sum = 0;
                    for (int j = 0; j < Skv; ++j) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
                    const T inv = sum > 0 ? (T)1 / sum : 0;
                    for (int c = 0; c < Dv; ++c) y[l * Dv + c] = 0;
                    for (int j = 0; j < Skv; ++j) {
                        const T pw = s[j] * inv;
                        for (int c = 0; c < Dv; ++c) y[l * Dv + c] += pw * v[j * Dv + c];
                    }
                }
            }
        });
    }

    float scale = 0.f;
    bool has_scale = false;
    std::string stage;
    FastGemmOpt opt;
};

Ptr<FlexAttentionLayer> FlexAttentionLayer::create(const LayerParams& params)
{
    return makePtr<FlexAttentionLayerImpl>(params);
}

}} // namespace cv::dnn
