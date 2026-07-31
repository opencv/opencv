// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <cmath>
#include "opencv2/core/utils/logger.hpp"
namespace cv { namespace dnn {

// Operator spec: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.GroupQueryAttention
class GroupQueryAttentionLayerImpl CV_FINAL : public GroupQueryAttentionLayer {
public:
    int num_heads = 0;
    int kv_num_heads = 0;
    float scale = 0.f;
    int local_window_size = -1;
    float softcap = 0.f;
    bool do_rotary = false;
    bool rotary_interleaved = false;
    Ptr<RotaryEmbeddingLayer> ropeQ, ropeK;

    GroupQueryAttentionLayerImpl(const LayerParams& params) {
        setParamsFrom(params);
        num_heads = params.get<int>("num_heads");
        kv_num_heads = params.get<int>("kv_num_heads");
        scale = params.get<float>("scale", 0.f);
        local_window_size = params.get<int>("local_window_size", -1);
        softcap = params.get<float>("softcap", 0.f);
        do_rotary = params.get<int>("do_rotary", 0) != 0;
        rotary_interleaved = params.get<int>("rotary_interleaved", 0) != 0;
        CV_CheckGT(num_heads, 0, "GroupQueryAttention: num_heads must be > 0");
        CV_CheckGT(kv_num_heads, 0, "GroupQueryAttention: kv_num_heads must be > 0");
        CV_CheckEQ(num_heads % kv_num_heads, 0, "GroupQueryAttention: num_heads must be a multiple of kv_num_heads");

        if (do_rotary) {
            LayerParams lpQ;
            lpQ.set("num_heads", num_heads);
            lpQ.set("interleaved", rotary_interleaved ? 1 : 0);
            ropeQ = RotaryEmbeddingLayer::create(lpQ);

            LayerParams lpK;
            lpK.set("num_heads", kv_num_heads);
            lpK.set("interleaved", rotary_interleaved ? 1 : 0);
            ropeK = RotaryEmbeddingLayer::create(lpK);
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
                          const int requiredOutputs,
                          const int requiredInternals,
                          std::vector<MatType>& outputs,
                          std::vector<MatType>& internals) const CV_OVERRIDE {
        CV_CheckType(inputs[0], inputs[0] == CV_32F || inputs[0] == CV_16F, "");
        outputs.assign(3, inputs[0]);
        internals.assign(requiredInternals, inputs[0]);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape>& outputs,
                                 std::vector<MatShape>& internals) const CV_OVERRIDE {
        CV_CheckGE((int)inputs.size(), 9, "GroupQueryAttention: expects 9 inputs");
        const MatShape& q = inputs[0];
        CV_CheckEQ(q.dims, 3, "GroupQueryAttention: query must be 3D (B,S,H*D)");
        CV_CheckEQ(q[2] % num_heads, 0, "GroupQueryAttention: hidden size must be divisible by num_heads");
        int B = q[0], S = q[1];
        int D = q[2] / num_heads;
        int Sp = 0;
        const MatShape& pastKey = inputs[3];
        if (pastKey.dims == 4) Sp = pastKey[2];

        outputs.resize(3);
        outputs[0] = MatShape{B, S, num_heads * D};
        outputs[1] = MatShape{B, kv_num_heads, Sp + S, D};
        outputs[2] = MatShape{B, kv_num_heads, Sp + S, D};

        internals.assign(1, MatShape{B, num_heads, S, D});     // Q
        internals.push_back(MatShape{B, kv_num_heads, S, D});  // Knew
        internals.push_back(MatShape{B, kv_num_heads, S, D});  // Vnew
        internals.push_back(MatShape{B, num_heads, S, D});     // outHeadsMajor
        return false;
    }

    static void splitHeads(const Mat& x, int B, int S, int nH, int D, Mat& out) {
        CV_Assert(x.isContinuous());
        CV_Assert(out.isContinuous());
        const float* src = x.ptr<float>();
        float* dst = out.ptr<float>();
        parallel_for_(Range(0, B * S), [&](const Range& r) {
            for (int bs = r.start; bs < r.end; ++bs) {
                const int b = bs / S;
                const int s = bs % S;
                const float* row = src + (size_t)bs * nH * D;
                for (int h = 0; h < nH; ++h) {
                    std::memcpy(dst + (((size_t)b * nH + h) * S + s) * D, row + (size_t)h * D, sizeof(float) * D);
                }
            }
        });
    }

    void applyRotary(const Ptr<RotaryEmbeddingLayer>& rope, Mat& x, int B, int nH, int S, int D,
                     const Mat& cosCache, const Mat& sinCache, const Mat& positionIds) const {
        int sizes4[4] = {B, nH, S, D};

        std::vector<Mat> ropeInputs = {x, cosCache, sinCache, positionIds};
        std::vector<Mat> ropeOutputs = {Mat(4, sizes4, CV_32F)};
        int dhalf = static_cast<int>(cosCache.size[cosCache.dims - 1]);
        std::vector<Mat> ropeInternals = {
            Mat(std::vector<int>{B, S, dhalf}, CV_32F),
            Mat(std::vector<int>{B, S, dhalf}, CV_32F),
        };
        rope->forward(ropeInputs, ropeOutputs, ropeInternals);
        ropeOutputs[0].copyTo(x);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();

        if (inputs_arr.depth() == CV_16F) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);
        CV_Assert(internals.size() == 4);
        Mat& Q = internals[0];
        Mat& Knew = internals[1];
        Mat& Vnew = internals[2];
        Mat& outHeadsMajor = internals[3];

        const Mat& query = inputs[0];
        const Mat& key = inputs[1];
        const Mat& value = inputs[2];
        const Mat& pastKey = inputs[3];
        const Mat& pastValue = inputs[4];
        const Mat& seqlensK = inputs[5];
        const Mat& cosCache = inputs[7];
        const Mat& sinCache = inputs[8];

        CV_Assert(query.isContinuous() && key.isContinuous() && value.isContinuous());
        CV_CheckType(seqlensK.depth(), seqlensK.depth() == CV_32S || seqlensK.depth() == CV_64S,
                     "GroupQueryAttention: seqlens_k must be CV_32S or CV_64S");

        const int B = query.size[0];
        const int S = query.size[1];
        const int D = query.size[2] / num_heads;
        const int Sp = (pastKey.dims == 4) ? pastKey.size[2] : 0;
        const int Skv = Sp + S;
        const int groupSize = num_heads / kv_num_heads;

        splitHeads(query, B, S, num_heads, D, Q);
        splitHeads(key, B, S, kv_num_heads, D, Knew);
        splitHeads(value, B, S, kv_num_heads, D, Vnew);

        Mat positionIds(std::vector<int>{B, S}, CV_MAKETYPE(CV_64S, 1));
        std::vector<int> validLen(B), padOffset(B);
        {
            int64_t* posPtr = positionIds.ptr<int64_t>();
            for (int b = 0; b < B; ++b) {
                int64_t sk = 0;
                if (seqlensK.depth() == CV_32S) {
                    sk = seqlensK.ptr<int32_t>()[b];
                } else {
                    sk = seqlensK.ptr<int64_t>()[b];
                }
                validLen[b] = static_cast<int>(sk) + 1;
                CV_CheckLE(validLen[b], Skv, "GroupQueryAttention: seqlens_k exceeds total KV buffer length");
                padOffset[b] = Skv - validLen[b];
                int64_t base = sk - S + 1;
                for (int i = 0; i < S; ++i) {
                    int64_t pos = base + i;
                    posPtr[(size_t)b * S + i] = std::max<int64_t>(pos, 0);
                }
            }
        }

        if (do_rotary) {
            applyRotary(ropeQ, Q, B, num_heads, S, D, cosCache, sinCache, positionIds);
            applyRotary(ropeK, Knew, B, kv_num_heads, S, D, cosCache, sinCache, positionIds);
        }

        Mat& presentKey = outputs[1];
        Mat& presentValue = outputs[2];
        {
            float* pk = presentKey.ptr<float>();
            float* pv = presentValue.ptr<float>();
            if (Sp > 0) CV_Assert(pastKey.isContinuous() && pastValue.isContinuous());
            const float* pastKeyPtr = (Sp > 0) ? pastKey.ptr<float>() : nullptr;
            const float* pastValuePtr = (Sp > 0) ? pastValue.ptr<float>() : nullptr;
            for (int b = 0; b < B; ++b) {
                for (int h = 0; h < kv_num_heads; ++h) {
                    float* dstK = pk + (((size_t)b * kv_num_heads + h) * Skv) * D;
                    float* dstV = pv + (((size_t)b * kv_num_heads + h) * Skv) * D;
                    if (Sp > 0) {
                        const float* srcK = pastKeyPtr + (((size_t)b * kv_num_heads + h) * Sp) * D;
                        const float* srcV = pastValuePtr + (((size_t)b * kv_num_heads + h) * Sp) * D;
                        std::memcpy(dstK, srcK, sizeof(float) * Sp * D);
                        std::memcpy(dstV, srcV, sizeof(float) * Sp * D);
                    }
                    const float* newK = Knew.ptr<float>() + (((size_t)b * kv_num_heads + h) * S) * D;
                    const float* newV = Vnew.ptr<float>() + (((size_t)b * kv_num_heads + h) * S) * D;
                    std::memcpy(dstK + (size_t)Sp * D, newK, sizeof(float) * S * D);
                    std::memcpy(dstV + (size_t)Sp * D, newV, sizeof(float) * S * D);
                }
            }
        }

        const float effScale = (scale > 0.f) ? scale : (1.f / std::sqrt(static_cast<float>(D)));

        parallel_for_(Range(0, B * num_heads), [&](const Range& r) {
            std::vector<float> scores(Skv);
            for (int bh = r.start; bh < r.end; ++bh) {
                const int b = bh / num_heads;
                const int h = bh % num_heads;
                const int kvh = h / groupSize;
                const int validLenB = validLen[b];
                const int padOffB = padOffset[b];

                const float* Qbh = Q.ptr<float>() + (size_t)bh * S * D;
                const float* Kbh = presentKey.ptr<float>() + (((size_t)b * kv_num_heads + kvh) * Skv) * D;
                const float* Vbh = presentValue.ptr<float>() + (((size_t)b * kv_num_heads + kvh) * Skv) * D;
                float* outBh = outHeadsMajor.ptr<float>() + (size_t)bh * S * D;

                for (int i = 0; i < S; ++i) {
                    const int queryPos = validLenB - S + i;
                    int hi = padOffB + queryPos;
                    int lo = padOffB;
                    if (local_window_size >= 0) lo = std::max(lo, hi - local_window_size);

                    const float* Qi = Qbh + (size_t)i * D;
                    float mx = -FLT_MAX;
                    for (int j = lo; j <= hi; ++j) {
                        const float* Kj = Kbh + (size_t)j * D;
                        float s = 0.f;
                        for (int d = 0; d < D; ++d) s += Qi[d] * Kj[d];
                        s *= effScale;
                        if (softcap > 0.f) s = softcap * std::tanh(s / softcap);
                        scores[j] = s;
                        if (s > mx) mx = s;
                    }
                    float sum = 0.f;
                    for (int j = lo; j <= hi; ++j) {
                        float e = std::exp(scores[j] - mx);
                        scores[j] = e;
                        sum += e;
                    }
                    const float invSum = 1.f / sum;
                    float* outRow = outBh + (size_t)i * D;
                    for (int d = 0; d < D; ++d) outRow[d] = 0.f;
                    for (int j = lo; j <= hi; ++j) {
                        const float a = scores[j] * invSum;
                        const float* Vj = Vbh + (size_t)j * D;
                        for (int d = 0; d < D; ++d) outRow[d] += a * Vj[d];
                    }
                }
            }
        });

        Mat& output = outputs[0];
        float* outPtr = output.ptr<float>();
        parallel_for_(Range(0, B * num_heads), [&](const Range& r) {
            for (int bh = r.start; bh < r.end; ++bh) {
                const int b = bh / num_heads;
                const int h = bh % num_heads;
                const float* src = outHeadsMajor.ptr<float>() + (size_t)bh * S * D;
                for (int s = 0; s < S; ++s) {
                    float* dst = outPtr + (((size_t)b * S + s) * num_heads + h) * D;
                    std::memcpy(dst, src + (size_t)s * D, sizeof(float) * D);
                }
            }
        });
    }
};

Ptr<GroupQueryAttentionLayer> GroupQueryAttentionLayer::create(const LayerParams& params) {
    return makePtr<GroupQueryAttentionLayerImpl>(params);
}

}} // namespace cv::dnn
