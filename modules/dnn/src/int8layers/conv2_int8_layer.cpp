// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "../layers/conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include "conv2_int8_kernels.simd.hpp"
#include "int8layers/conv2_int8_kernels.simd_declarations.hpp"

namespace cv {
namespace dnn {

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

static void convInt8Block(const void* inp_, const void* residual_,
                          void* out_, const ConvState& cs,
                          const void* weights_,
                          const void* weightsVNNI_,
                          const int* bias, const int* biasVNNI_,
                          const float* multiplier,
                          int inp_zp, int out_zp,
                          const int8_t* activLUT,
                          bool inputIsU8)
{
#if CV_TRY_AVX2
    if (cv::checkHardwareSupport(CV_CPU_AVX2)) {
        opt_AVX2::convInt8Block(inp_, residual_, out_, cs, weights_,
                                weightsVNNI_, bias, biasVNNI_,
                                multiplier, inp_zp, out_zp, activLUT, inputIsU8);
        return;
    }
#endif
    CV_CPU_CALL_BASELINE(convInt8Block, (inp_, residual_, out_, cs, weights_,
                                         weightsVNNI_, bias, biasVNNI_,
                                         multiplier, inp_zp, out_zp, activLUT, inputIsU8));
}


class Conv2Int8LayerImpl CV_FINAL : public Conv2Int8Layer
{
public:
    Mat weights;       // repacked int8 weights in block format
    Mat weightsVNNI;   // VNNI-transposed weights (pre-computed)
    Mat biasInt32;     // int32 fused bias
    Mat biasVNNI;      // int32 VNNI-adjusted bias (pre-computed)
    Mat outMultiplier; // float32 per-channel output multiplier
    MatShape wshape0;  // original weight shape (K x Cg x kH x kW)
    MatShape prevInpshape;
    ConvState cs;

    Mat activationLUT;
    Ptr<ActivationLayerInt8> activ;

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

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs, const int,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        // Input can be CV_8SC1 or CV_8UC1; output matches input type
        int outtype = !inputs.empty() ? inputs[0] : CV_8SC1;
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
        outshapes.assign(1, convInferShape(inpshapes[0], wshape0, emptyKernelShape,
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
        desiredInputs[0] = DATA_LAYOUT_BLOCK;
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
            if (!activ_int8->blobs.empty())
                activ_int8->blobs[0].convertTo(activationLUT, CV_8S);
            return true;
        }
        return false;
    }

    void forward(InputArrayOfArrays input_arrs,
                 OutputArrayOfArrays output_arrs,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        int ninputs = (int)input_arrs.total();
        CV_Assert(ninputs >= 1);

        const Mat& inp = input_arrs.getMat(0);
        MatShape inpshape = inp.shape();
        CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);
        CV_Assert(inp.type() == CV_8SC1 || inp.type() == CV_8UC1);

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
        int outtype = inp.type();

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
            prevInpshape = inpshape;
        }

        const int8_t* lutptr = !activationLUT.empty() ? activationLUT.ptr<int8_t>() : nullptr;

        convInt8Block(inp.data, resptr, out.data, cs,
                      weights.data,
                      weightsVNNI.empty() ? nullptr : weightsVNNI.data,
                      biasInt32.ptr<int>(),
                      biasVNNI.empty() ? nullptr : biasVNNI.ptr<int>(),
                      outMultiplier.ptr<float>(),
                      input_zp, output_zp,
                      lutptr, inputIsU8);

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

} // namespace dnn
} // namespace cv
