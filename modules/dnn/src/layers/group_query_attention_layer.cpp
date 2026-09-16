// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "cpu_kernels/fast_gemm.hpp"
#include "cpu_kernels/fast_attn.hpp"
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
    // Set by the importer when past_key has a fixed sequence length, i.e. the export
    // preallocates one KV buffer and rewrites it in place. Then present aliases that
    // buffer (same length as past) instead of being past concatenated with the new
    // tokens, and seqlens_k rather than the buffer size says how much of it is live.
    bool shared_kv_buffer = false;
    Ptr<RotaryEmbeddingLayer> ropeQ, ropeK;
    int ropeRotaryDim = -1;   // channel count the cached rotary layers were built for
    FastGemmOpt opt;
    // Kept across calls so a decode step, which re-derives them every token as the cache
    // grows, reuses the capacity instead of heap-allocating four vectors per forward().
    std::vector<size_t> qOff, kvOff, aOff, oOff;

    GroupQueryAttentionLayerImpl(const LayerParams& params) {
        setParamsFrom(params);
        num_heads = params.get<int>("num_heads");
        kv_num_heads = params.get<int>("kv_num_heads");
        scale = params.get<float>("scale", 0.f);
        local_window_size = params.get<int>("local_window_size", -1);
        softcap = params.get<float>("softcap", 0.f);
        do_rotary = params.get<int>("do_rotary", 0) != 0;
        rotary_interleaved = params.get<int>("rotary_interleaved", 0) != 0;
        shared_kv_buffer = params.get<int>("shared_kv_buffer", 0) != 0;
        CV_CheckGT(num_heads, 0, "GroupQueryAttention: num_heads must be > 0");
        CV_CheckGT(kv_num_heads, 0, "GroupQueryAttention: kv_num_heads must be > 0");
        CV_CheckEQ(num_heads % kv_num_heads, 0, "GroupQueryAttention: num_heads must be a multiple of kv_num_heads");

        // The rotary layers are built in ensureRope() instead of here: how many channels they
        // rotate follows from the cos_cache width, which is only known once inputs arrive.
        opt.init();
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
                          const int requiredOutputs,
                          const int requiredInternals,
                          std::vector<MatType>& outputs,
                          std::vector<MatType>& internals) const CV_OVERRIDE {
        // fp32 only. The kernel below has no half path, and routing CV_16F to
        // Layer::forward_fallback would not give it one: that only converts for
        // DNN_TARGET_OPENCL_FP16, and on any other target it calls the old-style
        // forward(vector<Mat*>&, ...) overload, which this layer does not implement and
        // whose base version is an empty compatibility stub -- the outputs would come back
        // untouched with no error. Unreachable while the engine widens halves to float
        // (enableFP16 is hardcoded false, opencv/opencv#26196), so claim what is true and
        // let it fail loudly here if that changes.
        CV_CheckType(inputs[0], inputs[0] == CV_32F, "GroupQueryAttention: only CV_32F is supported");
        outputs.assign(3, inputs[0]);
        internals.assign(requiredInternals, inputs[0]);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape>& outputs,
                                 std::vector<MatShape>& internals) const CV_OVERRIDE {
        // cos_cache / sin_cache (7, 8) only exist when the export sets do_rotary=1; a
        // sliding-window-only node stops at total_sequence_length.
        CV_CheckGE((int)inputs.size(), 7, "GroupQueryAttention: expects at least 7 inputs");
        if (do_rotary)
            CV_CheckGE((int)inputs.size(), 9,
                       "GroupQueryAttention: do_rotary=1 needs cos_cache and sin_cache");
        const MatShape& q = inputs[0];
        CV_CheckEQ(q.dims, 3, "GroupQueryAttention: query must be 3D (B,S,H*D)");
        CV_CheckEQ(q[2] % num_heads, 0, "GroupQueryAttention: hidden size must be divisible by num_heads");
        int B = q[0], S = q[1];
        int D = q[2] / num_heads;
        int Sp = 0;
        const MatShape& pastKey = inputs[3];
        if (pastKey.dims == 4) Sp = pastKey[2];

        // A shared buffer is rewritten in place, so present keeps the buffer's length; a
        // growing cache appends, so present is past + query. seqlens_k decides which slots
        // are live, but its value is not known during shape inference, which is why the
        // mode comes in as an attribute rather than being derived here.
        const int presentLen = shared_kv_buffer ? Sp : Sp + S;
        if (shared_kv_buffer)
            CV_CheckGE(Sp, S, "GroupQueryAttention: shared KV buffer is shorter than the query");

        outputs.resize(3);
        outputs[0] = MatShape{B, S, num_heads * D};
        outputs[1] = MatShape{B, kv_num_heads, presentLen, D};
        outputs[2] = MatShape{B, kv_num_heads, presentLen, D};

        internals.assign(1, MatShape{B, num_heads, S, D});     // Q
        internals.push_back(MatShape{B, kv_num_heads, S, D});  // Knew
        internals.push_back(MatShape{B, kv_num_heads, S, D});  // Vnew
        internals.push_back(MatShape{B, num_heads, S, presentLen});  // attention scores
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

    // RotaryEmbeddingLayer falls back to rotating the whole head when it is given no
    // rotary_embedding_dim, and then gathers head_size/2 floats per (b, s) row -- more than
    // a narrower cos_cache's rows hold, which overruns the scratch buffers sized below.
    // Pass the width explicitly so the gather and the buffers agree, and so partial rotary
    // (rotary_dim < head size) rotates the leading channels and copies the rest through,
    // which is what onnxruntime does with the same cache.
    void ensureRope(int rotaryDim) {
        if (ropeQ && ropeRotaryDim == rotaryDim)
            return;
        LayerParams lpQ;
        lpQ.set("num_heads", num_heads);
        lpQ.set("interleaved", rotary_interleaved ? 1 : 0);
        lpQ.set("rotary_embedding_dim", rotaryDim);
        ropeQ = RotaryEmbeddingLayer::create(lpQ);

        LayerParams lpK;
        lpK.set("num_heads", kv_num_heads);
        lpK.set("interleaved", rotary_interleaved ? 1 : 0);
        lpK.set("rotary_embedding_dim", rotaryDim);
        ropeK = RotaryEmbeddingLayer::create(lpK);

        ropeRotaryDim = rotaryDim;
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

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);
        CV_Assert(internals.size() == 4);
        Mat& Q = internals[0];
        Mat& Knew = internals[1];
        Mat& Vnew = internals[2];
        Mat& attnScores = internals[3];

        const Mat& query = inputs[0];
        const Mat& key = inputs[1];
        const Mat& value = inputs[2];
        const Mat& pastKey = inputs[3];
        const Mat& pastValue = inputs[4];
        const Mat& seqlensK = inputs[5];
        // cos_cache / sin_cache (7, 8) are read in the rotary branch below; a node without
        // do_rotary stops at total_sequence_length and never provides them.

        CV_Assert(query.isContinuous() && key.isContinuous() && value.isContinuous());
        CV_CheckType(seqlensK.depth(), seqlensK.depth() == CV_32S || seqlensK.depth() == CV_64S,
                     "GroupQueryAttention: seqlens_k must be CV_32S or CV_64S");

        const int B = query.size[0];
        const int S = query.size[1];
        const int D = query.size[2] / num_heads;
        const int Sp = (pastKey.dims == 4) ? pastKey.size[2] : 0;
        const int Skv = shared_kv_buffer ? Sp : Sp + S;   // length of present / attention span
        const int groupSize = num_heads / kv_num_heads;

        splitHeads(query, B, S, num_heads, D, Q);
        splitHeads(key, B, S, kv_num_heads, D, Knew);
        splitHeads(value, B, S, kv_num_heads, D, Vnew);

        Mat positionIds(std::vector<int>{B, S}, CV_MAKETYPE(CV_64S, 1));
        // validLen[b] is how much of the cache is live after this step; pastLen[b] is where
        // this step's new tokens land, i.e. the live length before it. Live tokens always
        // occupy [0, validLen) -- the tail, not the head, is the unwritten part.
        std::vector<int> validLen(B), pastLen(B);
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
                CV_CheckLE(validLen[b], Skv, "GroupQueryAttention: seqlens_k exceeds the KV buffer length");
                pastLen[b] = validLen[b] - S;
                CV_CheckGE(pastLen[b], 0, "GroupQueryAttention: seqlens_k is smaller than the query length");
                if (!shared_kv_buffer)
                    CV_CheckEQ(pastLen[b], Sp,
                        "GroupQueryAttention: seqlens_k disagrees with past_key length for a "
                        "growing cache; a preallocated buffer needs a static past_key dimension");
                int64_t base = sk - S + 1;
                for (int i = 0; i < S; ++i) {
                    int64_t pos = base + i;
                    posPtr[(size_t)b * S + i] = std::max<int64_t>(pos, 0);
                }
            }
        }

        if (do_rotary) {
            const Mat& cosCache = inputs[7];
            const Mat& sinCache = inputs[8];
            // Each cos_cache row holds one value per rotated pair, so it covers 2*width channels.
            const int rotaryDim = 2 * static_cast<int>(cosCache.size[cosCache.dims - 1]);
            CV_CheckLE(rotaryDim, D,
                "GroupQueryAttention: cos_cache is wider than half the head size");
            ensureRope(rotaryDim);
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
                    // Carry the live history over. For a growing cache that is all Sp slots;
                    // for a shared buffer only [0, pastLen) is live, the rest is about to be
                    // overwritten or zeroed.
                    const int carry = shared_kv_buffer ? pastLen[b] : Sp;
                    if (carry > 0) {
                        const float* srcK = pastKeyPtr + (((size_t)b * kv_num_heads + h) * Sp) * D;
                        const float* srcV = pastValuePtr + (((size_t)b * kv_num_heads + h) * Sp) * D;
                        std::memcpy(dstK, srcK, sizeof(float) * carry * D);
                        std::memcpy(dstV, srcV, sizeof(float) * carry * D);
                    }
                    const float* newK = Knew.ptr<float>() + (((size_t)b * kv_num_heads + h) * S) * D;
                    const float* newV = Vnew.ptr<float>() + (((size_t)b * kv_num_heads + h) * S) * D;
                    std::memcpy(dstK + (size_t)pastLen[b] * D, newK, sizeof(float) * S * D);
                    std::memcpy(dstV + (size_t)pastLen[b] * D, newV, sizeof(float) * S * D);
                    // onnxruntime leaves the not-yet-written tail of a shared buffer zeroed;
                    // attention never reads it, but present is an output, so match it.
                    if (validLen[b] < Skv) {
                        const size_t tail = sizeof(float) * (size_t)(Skv - validLen[b]) * D;
                        std::memset(dstK + (size_t)validLen[b] * D, 0, tail);
                        std::memset(dstV + (size_t)validLen[b] * D, 0, tail);
                    }
                }
            }
        }

        const float effScale = (scale > 0.f) ? scale : (1.f / std::sqrt(static_cast<float>(D)));
        Mat& output = outputs[0];

        // Q is [B, num_heads, S, D]; present K/V are [B, kv_num_heads, Skv, D]. One batched
        // GEMM entry per (batch, query head); the K/V offsets fold the grouping in by mapping
        // query head -> h / groupSize, so grouped heads read the shared KV head directly.
        qOff.resize((size_t)B * num_heads); kvOff.resize((size_t)B * num_heads);
        aOff.resize((size_t)B * num_heads); oOff.resize((size_t)B * num_heads);
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < num_heads; ++h) {
                const size_t idx = (size_t)b * num_heads + h;
                qOff[idx] = idx * S * D;
                kvOff[idx] = ((size_t)b * kv_num_heads + h / groupSize) * Skv * D;
                aOff[idx] = idx * S * Skv;
                // Straight into the [B, S, num_heads * D] output, so no un-permute pass.
                oOff[idx] = (size_t)b * S * num_heads * D + (size_t)h * D;
            }
        }

        // scores = (Q . K^T) * scale
        fastGemmBatch(B * num_heads, qOff.data(), kvOff.data(), aOff.data(),
                      S, Skv, D, effScale,
                      Q.ptr<float>(), D, 1,
                      presentKey.ptr<float>(), 1, D,
                      0.f, attnScores.ptr<float>(), Skv, opt);

        // Without a sliding window and with every batch's KV span full (the usual prefill and
        // decode steps), the allowed range is exactly [0, Sp + i], which is what the fused
        // kernel's own is_causal/past_seq_len bound already computes -- so skip the mask
        // entirely. That also lets softcap fuse into the same pass, because is_causal trims by
        // loop bound and fills the tail with min_val afterwards rather than before softcap.
        bool needMask = local_window_size > 0;
        for (int b = 1; b < B && !needMask; ++b)
            needMask = (pastLen[b] != pastLen[0]);

        if (!needMask) {
            fused_softmax_softcap_mask(attnScores, Mat(), softcap, softcap > 0.f, 9.f, -FLT_MAX,
                                       /*has_mask*/ false, /*is_causal*/ true,
                                       /*past_seq_len*/ pastLen[0]);
        } else {
            // is_causal takes one past length for the whole call, so a batch whose sequences
            // are at different lengths, or a sliding window, has to be masked explicitly.
            Mat mask(std::vector<int>{B, 1, S, Skv}, CV_8U, Scalar(0));
            for (int b = 0; b < B; ++b) {
                for (int i = 0; i < S; ++i) {
                    const int hi = pastLen[b] + i;
                    int lo = 0;
                    // local_window_size counts the window itself, so it spans local_window_size
                    // tokens ending at hi (verified against onnxruntime for several sizes), not
                    // local_window_size + 1. <= 0 means no window, matching the -1 default.
                    if (local_window_size > 0) lo = std::max(lo, hi - local_window_size + 1);
                    uchar* row = mask.ptr<uchar>(b, 0, i);
                    for (int j = std::max(lo, 0); j <= hi && j < Skv; ++j)
                        row[j] = 1;
                }
            }

            // Here softcap must run before the mask, not with it: this path applies the mask
            // first, and softcap would then squash the masked -FLT_MAX to -softcap, a finite
            // score that survives the softmax. So cap in a separate no-mask pass.
            if (softcap > 0.f)
                fused_softmax_softcap_mask(attnScores, Mat(), softcap, true, 9.f, -FLT_MAX,
                                           /*has_mask*/ false, /*is_causal*/ false,
                                           /*past_seq_len*/ 0, /*do_softmax*/ false);

            fused_softmax_softcap_mask(attnScores, mask, 0.f, false, 9.f, -FLT_MAX,
                                       /*has_mask*/ true, /*is_causal*/ false);
        }

        // out = probs . V
        fastGemmBatch(B * num_heads, aOff.data(), kvOff.data(), oOff.data(),
                      S, D, Skv, 1.f,
                      attnScores.ptr<float>(), Skv, 1,
                      presentValue.ptr<float>(), D, 1,
                      0.f, output.ptr<float>(), num_heads * D, opt);
    }
};

Ptr<GroupQueryAttentionLayer> GroupQueryAttentionLayer::create(const LayerParams& params) {
    return makePtr<GroupQueryAttentionLayerImpl>(params);
}

}} // namespace cv::dnn
