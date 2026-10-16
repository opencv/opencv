// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "../layers/conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include "conv2_int8_kernels.hpp"

namespace cv {
namespace dnn {

static void quantizeRowInt8(const float* src, uint8_t* dst, int len, float inv_sc, int zp, bool is_u8)
{
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VECSZ = VTraits<v_float32>::vlanes();
    v_float32 v_inv = vx_setall_f32(inv_sc);
    v_int32 v_zp = vx_setall_s32(zp);
    for (; j <= len - 4 * VECSZ; j += 4 * VECSZ) {
        v_int32 q0 = v_add(v_round(v_mul(vx_load(src + j), v_inv)), v_zp);
        v_int32 q1 = v_add(v_round(v_mul(vx_load(src + j + VECSZ), v_inv)), v_zp);
        v_int32 q2 = v_add(v_round(v_mul(vx_load(src + j + 2 * VECSZ), v_inv)), v_zp);
        v_int32 q3 = v_add(v_round(v_mul(vx_load(src + j + 3 * VECSZ), v_inv)), v_zp);
        v_int16 h0 = v_pack(q0, q1), h1 = v_pack(q2, q3);
        if (is_u8)
            v_store(dst + j, v_pack_u(h0, h1));
        else
            v_store((schar*)dst + j, v_pack(h0, h1));
    }
#endif
    for (; j < len; j++) {
        int q = cvRound(src[j] * inv_sc) + zp;
        dst[j] = is_u8 ? saturate_cast<uchar>(q) : (uint8_t)saturate_cast<schar>(q);
    }
}

static inline void copyBytes(uint8_t* dst, const uint8_t* src, int n)
{
    if (n >= 16) {
        int i = 0;
        for (; i + 16 <= n; i += 16)
            memcpy(dst + i, src + i, 16);
        if (i < n)
            memcpy(dst + n - 16, src + n - 16, 16);
    } else if (n >= 8) {
        memcpy(dst, src, 8);
        memcpy(dst + n - 8, src + n - 8, 8);
    } else {
        for (int i = 0; i < n; i++)
            dst[i] = src[i];
    }
}

static MatShape getWpackShapeInt8(const MatShape& wshape, int ngroups, int C0)
{
    CV_Assert(wshape.dims >= 3);
    int K = wshape[0], Cg = wshape[1];
    int ksize = (int)(wshape.total()) / (K * Cg);
    CV_Assert(K % ngroups == 0);
    int Kg = K / ngroups, K0 = C0;
    int Kblk = (Kg + K0 - 1) / K0;
    int C1Max = 0;
    for (int g = 0; g < ngroups; ++g) {
        int c_start = g * Cg;
        int c00 = c_start & (C0 - 1);
        int cblocks = (c00 + Cg + C0 - 1) / C0;
        C1Max = std::max(C1Max, cblocks);
    }
    return MatShape({ngroups, Kblk, ksize, C1Max, C0 * K0}, DATA_LAYOUT_UNKNOWN);
}

static void repackConvWeightsInt8(const Mat& weights, Mat& Wpack, int ngroups, int C0_)
{
    CV_Assert(weights.isContinuous());
    CV_Assert(weights.type() == CV_8SC1);
    CV_Assert(ngroups > 0);
    CV_Assert((C0_ & (C0_ - 1)) == 0 && C0_ >= 4);

    MatShape wshape = weights.shape();
    CV_Assert(wshape.dims >= 3);

    int K = wshape[0];
    CV_Assert(K % ngroups == 0);

    if (!Wpack.isContinuous())
        Wpack.release();

    MatShape wpackShape = getWpackShapeInt8(wshape, ngroups, C0_);
    Wpack.create(wpackShape, CV_8SC1);
    Wpack.setZero();

    parallel_for_(Range(0, K), [&](const Range& range) {
        int Cg = wshape[1], Kg = K / ngroups;
        int ksize = wpackShape[2], C1Max = wpackShape[3];
        int C0 = C0_, K0 = C0;
        const int8_t* wdata = weights.ptr<int8_t>();
        int8_t* Wpackdata = Wpack.ptr<int8_t>();

        for (int k = range.start; k < range.end; ++k) {
            int g = k / Kg;
            int kin = k - g * Kg;
            int kblk = kin / K0;
            int k0 = kin & (K0 - 1);

            int c_start = g * Cg;
            int c00 = c_start & (C0 - 1);

            for (int c = 0; c < Cg; ++c) {
                int ch = c00 + c;
                int c1 = ch / C0;
                int c0 = ch & (C0 - 1);

                const int8_t* wptr = wdata + ((k * Cg + c) * ksize);
                int8_t* wpackptr = Wpackdata +
                    (((g * (int64_t)wpackShape[1] + kblk) * ksize * C1Max + c1) * C0 + c0) * K0 + k0;
                for (int i = 0; i < ksize; ++i) {
                    wpackptr[i * (C1Max * C0 * K0)] = wptr[i];
                }
            }
        }
    });
}

static void repackWeightsForVNNI(const Mat& wpack, int ngroups, int Kg, int Cg,
                                  const Mat& biasInt32,
                                  Mat& wpackVNNI, Mat& biasVNNI)
{
    constexpr int C0 = 8, K0 = 8;
    MatShape ws = wpack.shape();
    int Kblk = ws[1], ksize = ws[2], C1Max = ws[3];

    wpackVNNI.create(ws, CV_8SC1);
    const int8_t* src = wpack.ptr<int8_t>();
    int8_t* dst = wpackVNNI.ptr<int8_t>();

    int blockSize = C0 * K0;
    int totalBlocks = (int)(ws.total() / blockSize);

    parallel_for_(Range(0, totalBlocks), [&](const Range& range) {
        for (int b = range.start; b < range.end; b++) {
            const int8_t* s = src + (size_t)b * blockSize;
            int8_t* d = dst + (size_t)b * blockSize;
            for (int k = 0; k < K0; k++) {
                d[k*4 + 0] = s[0*K0 + k];
                d[k*4 + 1] = s[1*K0 + k];
                d[k*4 + 2] = s[2*K0 + k];
                d[k*4 + 3] = s[3*K0 + k];
                d[32 + k*4 + 0] = s[4*K0 + k];
                d[32 + k*4 + 1] = s[5*K0 + k];
                d[32 + k*4 + 2] = s[6*K0 + k];
                d[32 + k*4 + 3] = s[7*K0 + k];
            }
        }
    });

    int K = ngroups * Kg;
    biasVNNI.create({K}, CV_32SC1);
    const int32_t* bsrc = biasInt32.ptr<int32_t>();
    int32_t* bdst = biasVNNI.ptr<int32_t>();

    parallel_for_(Range(0, K), [&](const Range& range) {
        for (int k = range.start; k < range.end; k++) {
            int g = k / Kg;
            int kin = k - g * Kg;
            int kblk = kin / K0;
            int k0 = kin & (K0 - 1);
            int c_start = g * Cg;
            int c00 = c_start & (C0 - 1);
            int cblocks = (c00 + Cg + C0 - 1) / C0;

            const int8_t* wbase = src + (size_t)((g * Kblk + kblk) * ksize * C1Max) * C0 * K0;
            int32_t wsum = 0;
            for (int ki = 0; ki < ksize; ki++)
                for (int c1 = 0; c1 < cblocks; c1++)
                    for (int c0 = 0; c0 < C0; c0++)
                        wsum += (int)wbase[((size_t)ki * C1Max + c1) * C0 * K0 + c0 * K0 + k0];
            bdst[k] = bsrc[k] - 128 * wsum;
        }
    });
}

class Conv2Int8LayerImpl CV_FINAL : public Conv2Int8Layer
{
public:
    Mat weights;       // repacked int8 weights in block format
    Mat weightsVNNI;   // VNNI-transposed weights (pre-computed)
    Mat biasInt32;
    Mat biasVNNI;
    Mat biasU8;
    Mat biasBaselineU8;
    Mat outMultiplier; // float32 per-channel output multiplier
    MatShape wshape0;  // original weight shape (K x Cg x kH x kW)
    MatShape prevInpshape;
    ConvState cs;

    Mat activationLUT;
    Ptr<ActivationLayerInt8> activ;

    Mat quantizedInput;

    bool useWindows = false;
    Mat winWeights, winWeightsVNNI;
    ConvState winCs;
    Mat paddedInput;
    Mat windowInput;

    bool addResidual;
    bool inputIsU8;

    Conv2Int8LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        auto_pad = getAutoPadding(params);
        ceil_mode = params.get<bool>("ceil_mode", false);
        strides = params.getVector<int>("stride");
        dilations = params.getVector<int>("dilation");
        pads = params.getVector<int>("pad");
        ngroups = params.get<int>("group", 1);

        input_sc = params.get<float>("input_scale", 1.f);
        input_zp = params.get<int>("input_zeropoint", 0);
        output_sc = params.get<float>("scales", 1.f);
        output_zp = params.get<int>("zeropoints", 0);
        per_channel = params.get<bool>("per_channel", true);
        inputIsU8 = params.get<bool>("input_is_u8", false);

        addResidual = false;

        if (!blobs.empty()) {
            CV_Assert(blobs.size() >= 3);
            wshape0 = blobs[0].shape();
            biasInt32 = blobs[1];
            outMultiplier = blobs[2];
        }
    }

    Conv2Int8LayerImpl(const Conv2Int8Params& p)
    {
        name = p.name;
        type = "Conv2Int8";
        strides = p.strides;
        dilations = p.dilations;
        pads = p.pads;
        ngroups = p.ngroups;
        auto_pad = p.auto_pad;
        ceil_mode = p.ceil_mode;
        input_sc = p.input_sc;
        input_zp = p.input_zp;
        output_sc = p.output_sc;
        output_zp = p.output_zp;
        per_channel = p.per_channel;
        inputIsU8 = p.input_is_u8;
        float_input = p.float_input;
        addResidual = false;

        if (!p.weights.empty()) {
            wshape0 = p.weights.shape();
            biasInt32 = p.bias;
            outMultiplier = p.outputMultiplier;
            blobs = { p.weights, p.bias, p.outputMultiplier };
        }
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs, const int,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        int outtype;
        if (float_input)
            outtype = inputIsU8 ? CV_8UC1 : CV_8SC1;
        else
            outtype = !inputs.empty() ? inputs[0] : CV_8SC1;
        outputs.assign(requiredOutputs, outtype);
        internals.clear();
    }

    bool getMemoryShapes(const std::vector<MatShape>& inpshapes,
                         const int,
                         std::vector<MatShape>& outshapes,
                         std::vector<MatShape>& tempshapes) const CV_OVERRIDE
    {
        size_t ninputs = inpshapes.size();
        CV_Assert(ninputs >= 1);

        std::vector<int> emptyKernelShape;
        MatShape inpshape = inpshapes[0];
        if (float_input && inpshape.layout != DATA_LAYOUT_BLOCK) {
            if (inpshape.layout == DATA_LAYOUT_UNKNOWN)
                inpshape.layout = DATA_LAYOUT_NCHW;
            int C0 = getNetImpl(this)->defaultC0;
            inpshape = inpshape.toLayout(DATA_LAYOUT_BLOCK, C0);
        }
        outshapes.assign(1, convInferShape(inpshape, wshape0, emptyKernelShape,
                                           ngroups, strides, dilations,
                                           pads, auto_pad, ceil_mode));
        tempshapes.clear();
        return true;
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                   std::vector<DataLayout>& desiredInputs,
                   const int requiredOutputs,
                   std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        size_t ninputs = actualInputs.size();
        CV_Assert(ninputs >= 1u && requiredOutputs == 1u);
        desiredInputs = actualInputs;
        desiredInputs[0] = float_input ? DATA_LAYOUT_NCHW : DATA_LAYOUT_BLOCK;
        for (size_t i = 1; i < ninputs; i++)
            desiredInputs[i] = DATA_LAYOUT_UNKNOWN;
        outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
        return getNetImpl(this)->defaultC0;
    }

    void setWeightsInt8(const Mat& w_q, const Mat& bias, const Mat& multiplier, int C0)
    {
        CV_Assert(w_q.type() == CV_8SC1);
        wshape0 = w_q.shape();
        biasInt32 = bias;
        outMultiplier = multiplier;

        repackConvWeightsInt8(w_q, weights, ngroups, C0);
    }

    bool fuseAddResidual(Arg /*residual*/)
    {
        addResidual = true;
        return true;
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        Ptr<ActivationLayerInt8> activ_int8 = layer.dynamicCast<ActivationLayerInt8>();
        if (!activ_int8.empty()) {
            activ = activ_int8;
            if (!activ_int8->activationLUT.empty())
                activationLUT = activ_int8->activationLUT;
            else if (!activ_int8->blobs.empty())
                activ_int8->blobs[0].copyTo(activationLUT);
            return true;
        }
        return false;
    }

    void quantizeInterleaveBlock(const Mat& fp32_nchw, Mat& int8_block, int C0) const
    {
        MatShape nchw = fp32_nchw.shape();
        if (nchw.layout == DATA_LAYOUT_UNKNOWN)
            nchw.layout = DATA_LAYOUT_NCHW;
        CV_Assert(nchw.dims >= 2);
        int N = nchw[0], C = nchw[1];
        int planesize = 1;
        for (int d = 2; d < nchw.dims; d++) planesize *= nchw[d];
        int C1 = (C + C0 - 1) / C0;

        MatShape blkshape = nchw.toLayout(DATA_LAYOUT_BLOCK, C0);
        int8_block.fit(blkshape, inputIsU8 ? CV_8UC1 : CV_8SC1);

        float inv_sc = input_sc > 0.f ? 1.f / input_sc : 0.f;
        int zp = input_zp;
        const bool is_u8 = inputIsU8;
        const float* src = fp32_nchw.ptr<float>();
        uint8_t* dst = int8_block.ptr<uint8_t>();

        const int PIX_STRIPE = 2048;
        const int nstripes = (planesize + PIX_STRIPE - 1) / PIX_STRIPE;

        parallel_for_(Range(0, N * C1 * nstripes), [&](const Range& range) {
            const int BLK = 256;
            AutoBuffer<uint8_t> qbuf_(C0 * BLK);
            uint8_t* qbuf = qbuf_.data();
            for (int t = range.start; t < range.end; t++) {
                int nc = t / nstripes, stripe = t - nc * nstripes;
                int n = nc / C1, c1 = nc - n * C1;
                int c_base = c1 * C0;
                int nzc = std::min(C0, C - c_base);
                const float* src_block = src + ((size_t)n * C + c_base) * planesize;
                uint8_t* dst_block = dst + ((size_t)n * C1 + c1) * planesize * C0;
                int i1 = std::min((stripe + 1) * PIX_STRIPE, planesize);
                for (int i = stripe * PIX_STRIPE; i < i1; i += BLK) {
                    int len = std::min(BLK, i1 - i);
                    for (int c = 0; c < nzc; c++)
                        quantizeRowInt8(src_block + (size_t)c * planesize + i, qbuf + c * BLK,
                                        len, inv_sc, zp, is_u8);
                    uint8_t* dst_px = dst_block + (size_t)i * C0;
                    if (nzc < C0)
                        memset(dst_px, 0, (size_t)len * C0);
                    for (int c = 0; c < nzc; c++) {
                        const uint8_t* q = qbuf + c * BLK;
                        for (int j = 0; j < len; j++)
                            dst_px[(size_t)j * C0 + c] = q[j];
                    }
                }
            }
        });
    }

    bool canUseWindowPacking(const MatShape& outshape, int C0) const
    {
        static const bool enabled =
            utils::getConfigurationParameterBool("OPENCV_DNN_INT8_CONV_WINDOW_PACKING", true);
        if (!enabled || !float_input || ngroups != 1 || blobs.empty() || C0 != 8 ||
            wshape0.dims != 4 || outshape.dims != 5)
            return false;
        int C = wshape0[1], kh = wshape0[2], kw = wshape0[3];
        if (C > 4 || kh * kw == 1)
            return false;
        size_t Cw8 = (size_t)(kh * kw * C + C0 - 1) / C0 * C0;
        size_t nbytes = (size_t)outshape[0] * outshape[2] * outshape[3] * Cw8;
        return nbytes <= ((size_t)64 << 20);
    }

    void initWindowConv(const MatShape& outshape, int C0)
    {
        int K = wshape0[0], C = wshape0[1], kh = wshape0[2], kw = wshape0[3];
        int Cw8 = (kh * kw * C + C0 - 1) / C0 * C0;
        if (winWeights.empty()) {
            const Mat& w = blobs[0];
            CV_Assert(w.type() == CV_8SC1 && w.isContinuous() && w.total() == (size_t)K * C * kh * kw);
            Mat w1x1(std::vector<int>{K, Cw8, 1, 1}, CV_8SC1, Scalar(0));
            const int8_t* src = w.ptr<int8_t>();
            int8_t* dst = w1x1.ptr<int8_t>();
            for (int k = 0; k < K; k++)
                for (int c = 0; c < C; c++)
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++)
                            dst[(size_t)k * Cw8 + (ky * kw + kx) * C + c] =
                                src[(((size_t)k * C + c) * kh + ky) * kw + kx];
            repackConvWeightsInt8(w1x1, winWeights, 1, C0);
            Mat unusedBias;
            repackWeightsForVNNI(winWeights, 1, K, Cw8, biasInt32, winWeightsVNNI, unusedBias);
        }
        int N = outshape[0], Ho = outshape[2], Wo = outshape[3];
        MatShape wininp = MatShape({N, Cw8, Ho, Wo}, DATA_LAYOUT_NCHW).toLayout(DATA_LAYOUT_BLOCK, C0);
        winCs.initConv(wininp, MatShape({K, Cw8, 1, 1}, DATA_LAYOUT_UNKNOWN), outshape, 1,
                       {1, 1}, {1, 1}, {0, 0, 0, 0}, AUTO_PAD_NONE, false,
                       FAST_ACTIV_NONE, nullptr, {});
    }

    void packWindows(const Mat& fp32_nchw, int C0)
    {
        MatShape nchw = fp32_nchw.shape();
        CV_Assert(nchw.dims == 4 && C0 == 8);
        const int N = nchw[0], C = nchw[1], H = nchw[2], W = nchw[3];
        const int kh = cs.kshape[1], kw = cs.kshape[2];
        const int sy = cs.strides[1], sx = cs.strides[2];
        const int dy = cs.dilations[1], dx = cs.dilations[2];
        const int padT = cs.pads[1], padL = cs.pads[2];
        const int Ho = cs.outshape[2], Wo = cs.outshape[3];
        const int Hp = std::max(padT + H + cs.pads[1 + ConvState::MAX_CONV_DIMS], (Ho - 1) * sy + (kh - 1) * dy + 1);
        const int Wp = std::max(padL + W + cs.pads[2 + ConvState::MAX_CONV_DIMS], (Wo - 1) * sx + (kw - 1) * dx + 1);
        const int Cw = kh * kw * C, Cw8 = (Cw + C0 - 1) / C0 * C0, C1w = Cw8 / C0;

        const float inv_sc = input_sc > 0.f ? 1.f / input_sc : 0.f;
        const int zp = input_zp;
        const bool is_u8 = inputIsU8;
        const uint8_t zpbyte = (uint8_t)input_zp;
        const float* src = fp32_nchw.ptr<float>();
        const size_t prow = (size_t)Wp * C;

        paddedInput.fit({N * Hp, (int)prow}, CV_8UC1);
        uint8_t* pbuf = paddedInput.ptr<uint8_t>();
        parallel_for_(Range(0, N * Hp), [&](const Range& range) {
            AutoBuffer<uint8_t> qbuf_((size_t)C * W);
            uint8_t* qbuf = qbuf_.data();
            for (int t = range.start; t < range.end; t++) {
                int n = t / Hp, y = t - n * Hp - padT;
                uint8_t* drow = pbuf + (size_t)t * prow;
                if (y < 0 || y >= H) {
                    memset(drow, zpbyte, prow);
                    continue;
                }
                memset(drow, zpbyte, (size_t)padL * C);
                memset(drow + (size_t)(padL + W) * C, zpbyte, (size_t)(Wp - padL - W) * C);
                for (int c = 0; c < C; c++)
                    quantizeRowInt8(src + (((size_t)n * C + c) * H + y) * W, qbuf + (size_t)c * W,
                                    W, inv_sc, zp, is_u8);
                uint8_t* d = drow + (size_t)padL * C;
                for (int x = 0; x < W; x++)
                    for (int c = 0; c < C; c++)
                        d[x * C + c] = qbuf[(size_t)c * W + x];
            }
        });

        MatShape winshape = MatShape({N, Cw8, Ho, Wo}, DATA_LAYOUT_NCHW).toLayout(DATA_LAYOUT_BLOCK, C0);
        windowInput.fit(winshape, inputIsU8 ? CV_8UC1 : CV_8SC1);
        uint8_t* wdst = windowInput.ptr<uint8_t>();
        const size_t plane = (size_t)Ho * Wo * C0;
        parallel_for_(Range(0, N * Ho), [&](const Range& range) {
            AutoBuffer<uint8_t> win_(Cw8);
            uint8_t* win = win_.data();
            memset(win + Cw, 0, Cw8 - Cw);
            for (int t = range.start; t < range.end; t++) {
                int n = t / Ho, oy = t - n * Ho;
                const uint8_t* srow = pbuf + ((size_t)n * Hp + (size_t)oy * sy) * prow;
                uint8_t* drow = wdst + (size_t)n * C1w * plane + (size_t)oy * Wo * C0;
                for (int ox = 0; ox < Wo; ox++) {
                    const uint8_t* s = srow + (size_t)ox * sx * C;
                    for (int ky = 0; ky < kh; ky++) {
                        const uint8_t* sk = s + (size_t)ky * dy * prow;
                        uint8_t* wk = win + ky * kw * C;
                        if (dx == 1) {
                            copyBytes(wk, sk, kw * C);
                        } else {
                            for (int kx = 0; kx < kw; kx++)
                                for (int c = 0; c < C; c++)
                                    wk[kx * C + c] = sk[(size_t)kx * dx * C + c];
                        }
                    }
                    uint8_t* d = drow + (size_t)ox * C0;
                    for (int c1 = 0; c1 < C1w; c1++)
                        memcpy(d + c1 * plane, win + c1 * 8, 8);
                }
            }
        });
    }

    void forward(InputArrayOfArrays input_arrs,
                 OutputArrayOfArrays output_arrs,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        int ninputs = (int)input_arrs.total();
        CV_Assert(ninputs >= 1);

        Mat inp_ref = input_arrs.getMat(0);

        MatShape inpshape;
        if (float_input) {
            CV_Assert(inp_ref.type() == CV_32F);
            MatShape nchw = inp_ref.shape();
            if (nchw.layout == DATA_LAYOUT_UNKNOWN)
                nchw.layout = DATA_LAYOUT_NCHW;
            inpshape = nchw.toLayout(DATA_LAYOUT_BLOCK, getNetImpl(this)->defaultC0);
        } else {
            CV_Assert(inp_ref.type() == CV_8SC1 || inp_ref.type() == CV_8UC1);
            inpshape = inp_ref.shape();
        }
        CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);

        int C0 = inpshape.back();

        if (weights.empty() && !blobs.empty()) {
            repackConvWeightsInt8(blobs[0], weights, ngroups, C0);
        }

        if (weightsVNNI.empty() && !weights.empty()) {
            int K = wshape0[0];
            int Cg = wshape0[1];
            int Kg = K / ngroups;
            repackWeightsForVNNI(weights, ngroups, Kg, Cg,
                                  biasInt32, weightsVNNI, biasVNNI);

            if (inputIsU8) {
                biasU8.create({K}, CV_32SC1);
                biasBaselineU8.create({K}, CV_32SC1);
                const int32_t* bI = biasInt32.ptr<int32_t>();
                const int32_t* bV = biasVNNI.ptr<int32_t>();
                int32_t* bU = biasU8.ptr<int32_t>();
                int32_t* bB = biasBaselineU8.ptr<int32_t>();
                for (int k = 0; k < K; k++) {
                    int32_t wsum = (bI[k] - bV[k]) / 128;
                    bU[k] = bI[k] - input_zp * wsum;
                    bB[k] = bI[k] + (128 - input_zp) * wsum;
                }
            }
        }

        Mat residual;
        const void* resptr = nullptr;
        if (addResidual) {
            residual = input_arrs.getMat(ninputs - 1);
            resptr = residual.data;
            ninputs--;
        }

        std::vector<int> emptyKernelShape;
        MatShape outshape = convInferShape(inpshape, wshape0, emptyKernelShape,
                                           ngroups, strides, dilations,
                                           pads, auto_pad, ceil_mode);
        int outtype = float_input ? (inputIsU8 ? CV_8UC1 : CV_8SC1) : inp_ref.type();

        int outkind = output_arrs.kind();
        Mat out;

        if (outkind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = output_arrs.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, outtype);
            out = outs[0];
        } else {
            std::vector<UMat>& outs = output_arrs.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outshape, outtype);
            out.fit(outshape, outtype);
        }

        if (inpshape != prevInpshape) {
            cs.initConv(inpshape, wshape0, outshape, ngroups,
                        strides, dilations, pads, auto_pad, ceil_mode,
                        FAST_ACTIV_NONE, nullptr, {});

            if (cs.depthwise) {
                cs.wshape = getWpackShapeInt8(wshape0, ngroups, C0);
            }
            useWindows = canUseWindowPacking(outshape, C0);
            if (useWindows)
                initWindowConv(outshape, C0);
            prevInpshape = inpshape;
        }

        const Mat* inp = &inp_ref;
        if (useWindows) {
            packWindows(inp_ref, C0);
            inp = &windowInput;
        } else if (float_input) {
            quantizeInterleaveBlock(inp_ref, quantizedInput, C0);
            inp = &quantizedInput;
        }
        const Mat& wts = useWindows ? winWeights : weights;
        const Mat& wtsVNNI = useWindows ? winWeightsVNNI : weightsVNNI;

        const int8_t* lutptr = !activationLUT.empty() ? activationLUT.ptr<int8_t>() : nullptr;

        const int* effectiveBiasVNNI = inputIsU8
            ? (biasU8.empty()   ? nullptr : biasU8.ptr<int>())
            : (biasVNNI.empty() ? nullptr : biasVNNI.ptr<int>());

        const int* baselineBias = (inputIsU8 && !biasBaselineU8.empty())
            ? biasBaselineU8.ptr<int>()
            : biasInt32.ptr<int>();

        convInt8Block(inp->data, resptr, out.data, useWindows ? winCs : cs,
                      wts.data,
                      wtsVNNI.empty() ? nullptr : wtsVNNI.data,
                      baselineBias,
                      effectiveBiasVNNI,
                      outMultiplier.ptr<float>(),
                      input_zp, output_zp,
                      lutptr, inputIsU8, false);

        if (outkind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = output_arrs.getUMatVecRef();
            out.copyTo(outs[0]);
        }
    }
};

Ptr<Conv2Int8Layer> Conv2Int8Layer::create(const LayerParams& params)
{
    return Ptr<Conv2Int8Layer>(new Conv2Int8LayerImpl(params));
}

Ptr<Conv2Int8Layer> Conv2Int8Layer::create(const Conv2Int8Params& params)
{
    return Ptr<Conv2Int8Layer>(new Conv2Int8LayerImpl(params));
}

} // namespace dnn
} // namespace cv
