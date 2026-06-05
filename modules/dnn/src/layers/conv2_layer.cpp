// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include "conv2_common.hpp"
#include "cpu_kernels/mlas_gemm.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include <algorithm>
#include <cstring>

namespace cv
{
namespace dnn
{

/*
    Convolution layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Conv.html

    Opset's 1 to 22 are covered.
*/

class Conv2LayerImpl : public Conv2Layer
{
public:
    Conv2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        auto_pad = getAutoPadding(params);
        ceil_mode = params.get<bool>("ceil_mode", false);
        strides = params.getVector<int>("stride");
        dilations = params.getVector<int>("dilation");
        pads = params.getVector<int>("pad");
        ngroups = params.get<int>("group", 1);
        fusedBatchNorm = false;
        fastActivation = FAST_ACTIV_NONE;
        activationFunc = nullptr;
        addResidual = false;
    }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "group: " << ngroups << ",\n";

        if (!wshape0.empty()) {
            prindent(strm, indent);
            strm << "ksize: [";
            for (int k = 0; k < wshape0.dims; k++)
                strm << (k > 0 ? ", " : "") << wshape0[k];
            strm << "],\n";
        }

        prindent(strm, indent);
        strm << "stride: [";
        for (size_t k = 0; k < strides.size(); k++)
            strm << (k > 0 ? ", " : "") << strides[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "dilation: [";
        for (size_t k = 0; k < dilations.size(); k++)
            strm << (k > 0 ? ", " : "") << dilations[k];
        strm << "],\n";

        prindent(strm, indent);
        strm << "pad: [";
        for (size_t k = 0; k < pads.size(); k++)
            strm << (k > 0 ? ", " : "") << pads[k];
        strm << "],\n";

        if (fusedBatchNorm) {
            prindent(strm, indent);
            strm << "batch_norm: true,\n";
        }

        if (fastActivation != FAST_ACTIV_NONE || activationFunc != nullptr || !activ.empty()) {
            prindent(strm, indent);
            strm << "fused_activation: " <<
                (fastActivation != FAST_ACTIV_NONE ? fastActivationToString(fastActivation) :
                 activationFunc != nullptr ? "ActivationFunc" :
                 activ->type) << ",\n";
        }

        if (addResidual) {
            prindent(strm, indent);
            strm << "addResidual: true,\n";
        }

        if (activ) {
            prindent(strm, indent);
            strm << "activation: " << activ->name << ",\n";
        }

        return strm;
    }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual void setWeights(InputArray weights_arr, InputArray bias_arr,
                            int C0, int accuracy) CV_OVERRIDE
    {
        Mat weights_ = weights_arr.getMat();
        Mat bias_ = bias_arr.getMat();
        CV_Assert(!weights_.empty());
        int wtype0 = weights_.type();
        CV_Assert(wtype0 == CV_32F || wtype0 == CV_16F || wtype0 == CV_16BF);
        CV_Assert(accuracy == -1 || accuracy == CV_32F);
        int wtype = accuracy < 0 ? CV_32F : accuracy;

        wshape0 = weights_.shape();
        bool depthwise = ngroups == wshape0[0] && wshape0[1] == 1;

        if (depthwise) {
            repackDepthwiseConvWeights(weights_, weights, wtype, C0);
        } else {
            repackConvWeights(weights_, weights, wtype, ngroups, C0);
        }

        if (!bias_.empty()) {
            CV_Assert(bias_.isContinuous() && bias_.total() == wshape0[0]);
            bias_.convertTo(bias, CV_32F);
        }

        // Pre-pack for MLAS 1x1 SGEMM (1x1 dense, stride-1, FP32, Cout*Cin
        // >= 256*256 so the reorder cost is amortized).
        mlas_packed_B_.release();
        mlas_packed_M_ = mlas_packed_K_ = 0;
        if (!depthwise && ngroups == 1 && wtype0 == CV_32F &&
            wtype == CV_32F && mlasAvailable())
        {
            bool ksize_all_one = wshape0.dims >= 3;
            for (int k = 2; k < wshape0.dims; k++)
                if (wshape0[k] != 1) { ksize_all_one = false; break; }
            bool strides_all_one = !strides.empty();
            for (int s : strides) if (s != 1) { strides_all_one = false; break; }
            const int Cout = wshape0[0];
            const int Cin  = wshape0[1];
            const bool big_enough = (int64_t)Cout * (int64_t)Cin >= (int64_t)(256 * 256);
            if (ksize_all_one && strides_all_one && big_enough) {
                size_t pack_bytes = mlasSgemmPackBSize(false, true, Cout, Cin);
                if (pack_bytes > 0 && pack_bytes <= (size_t)INT_MAX) {
                    mlas_packed_B_.create(1, (int)pack_bytes, CV_8U);
                    // Weight is (Cout, Cin, 1, ..., 1) contiguous; reshape to
                    // (Cout, Cin) and PackB with trans_b=true.
                    Mat W2D = weights_.reshape(1, Cout);
                    if (!mlasSgemmPackB(false, true, Cout, Cin,
                                        W2D.ptr<float>(), Cin,
                                        mlas_packed_B_.data))
                    {
                        mlas_packed_B_.release();
                    } else {
                        mlas_packed_M_ = Cout;
                        mlas_packed_K_ = Cin;
                    }
                }
            }
        }
    }

    void fuseBatchNormWeights(const BatchNorm2Layer* bn)
    {
        Mat bn_scale, bn_bias;
        bn->getScaleBias(bn_scale, bn_bias);

        CV_Assert(bn_scale.isContinuous() && bn_bias.isContinuous());
        CV_Assert(bn_scale.type() == CV_32F && bn_bias.type() == CV_32F);
        CV_Assert(bn_scale.total() == bn_bias.total());
        int K = (int)bn_scale.total();
        CV_Assert(bias.empty() || (bias.type() == CV_32F && bias.total() == (size_t)K));
        const float* bias_data = bias.data ? bias.ptr<float>() : nullptr;

        fusedScale.fit(1, &K, CV_32F);
        fusedBias.fit(1, &K, CV_32F);

        const float* bn_scale_data = bn_scale.ptr<float>();
        const float* bn_bias_data = bn_bias.ptr<float>();
        float* fused_scale_data = fusedScale.ptr<float>();
        float* fused_bias_data = fusedBias.ptr<float>();

        // (sum(x*w) + bias)*bn_scale + bn_bias => sum(x*w)*fused_scale + fused_bias,
        // where fused_scale = bn_scale and fused_bias = bias*bn_scale + bn_bias.
        for (size_t i = 0; i < K; i++) {
            fused_scale_data[i] = bn_scale_data[i];
            fused_bias_data[i] = (bias_data ? bn_scale_data[i]*bias_data[i] : 0.f) + bn_bias_data[i];
        }
    }

    virtual bool fuseBatchNorm(const Ptr<Layer>& bnlayer) override
    {
        BatchNorm2Layer* bn = dynamic_cast<BatchNorm2Layer*>(bnlayer.get());
        // addResidual means the graph order is conv->add->bn: the residual is added
        // before BN, but the conv applies the fused BN scale only to conv(x), not to
        // the residual. Refuse so BN stays separate and runs on (conv + residual).
        if (fusedBatchNorm || !bn || bn->inputs.size() > 1 ||
            fastActivation != FAST_ACTIV_NONE || !activ.empty() || addResidual)
            return false;
        fuseBatchNormWeights(bn);
        fusedBatchNorm = true;
        return true;
    }

    virtual bool fuseAddBias(InputArray arr) CV_OVERRIDE
    {
        if (inputs.size() > 1 || fusedBatchNorm || addResidual)
            return false;
        Mat new_bias = arr.getMat();
        CV_Assert(new_bias.isContinuous() && new_bias.dims == 1);
        if (new_bias.type() != CV_32F) {
            Mat temp;
            new_bias.convertTo(temp, CV_32F);
            new_bias = temp;
            CV_Assert(new_bias.type() == CV_32F);
        }
        if (!bias.empty()) {
            CV_Assert(bias.shape() == new_bias.shape());
            add(bias, new_bias, bias);
        } else {
            new_bias.copyTo(bias);
        }
        return true;
    }

    virtual bool fuseActivation(const Ptr<Layer>& activlayer) override
    {
        ActivationLayer* activ_ptr = dynamic_cast<ActivationLayer*>(activlayer.get());
        if (!activ_ptr || fastActivation != FAST_ACTIV_NONE ||
            activationFunc != nullptr || !activ.empty())
            return false;

        ReLULayer* activRelu = dynamic_cast<ReLULayer*>(activ_ptr);
        ReLU6Layer* activClip = dynamic_cast<ReLU6Layer*>(activ_ptr);
        ChannelsPReLULayer* activPRelu = dynamic_cast<ChannelsPReLULayer*>(activ_ptr);
        if (activRelu) {
            float alpha = activRelu->negativeSlope;
            if (alpha == 0.f) {
                fastActivation = FAST_ACTIV_RELU;
            } else {
                fastActivation = FAST_ACTIV_LEAKY_RELU;
                activParams = {alpha};
            }
        } else if (activClip && activClip->minValue == 0.f) {
            fastActivation = FAST_ACTIV_CLIP;
            activParams = {activClip->minValue, activClip->maxValue};
        } else if (activPRelu && activPRelu->blobs.size() == 1) {
            fastActivation = FAST_ACTIV_PRELU;
            const Mat& slopes = activPRelu->blobs[0];
            int slopesType = slopes.type();
            CV_Assert_N((slopesType == CV_32F || slopesType == CV_16F || slopesType == CV_16BF),
                        slopes.isContinuous());
            int nslopes = int(slopes.total());
            Mat(1, &nslopes, slopesType, (void*)slopes.data).convertTo(activParams, CV_32F);
        } else {
            activationFunc = activ_ptr->getActivationFunc(CV_32F, activParams);
            if (!activationFunc)
                return false;
        }
        return true;
    }

    virtual bool fuseAddResidual(Arg residual) CV_OVERRIDE
    {
        if (activ.empty() && fastActivation == FAST_ACTIV_NONE &&
            activationFunc == nullptr && !addResidual && residual.idx >= 0) {
            addResidual = true;
            inputs.push_back(residual);
            return true;
        }
        return false;
    }

    virtual int64_t getFLOPS(const std::vector<MatShape>& inputs,
                             const std::vector<MatShape>& outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 1);
        CV_Assert(outputs.size() == 1);
        // probably, there should be a coefficient in the case of complex reduction functions
        MatShape inpshape = inputs[0], wshape = inputs.size() > 1 ? inputs[1] : wshape0;
        int C = inpshape[1]*inpshape.back();
        size_t ksize = wshape.total();
        return (int64_t)((inputs[0].total()/C)*ksize/ngroups);
    }

    virtual void getTypes(const std::vector<MatType>& inptypes,
                          const int, const int,
                          std::vector<MatType>& outtypes,
                          std::vector<MatType>& temptypes) const CV_OVERRIDE
    {
        int ninputs = (int)inptypes.size();
        CV_Assert(ninputs >= 1);

        outtypes.assign(1, inferType(inptypes[0]));
        temptypes.clear();
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inpshapes,
                                 const int,
                                 std::vector<MatShape> &outshapes,
                                 std::vector<MatShape> &tempshapes) const CV_OVERRIDE
    {
        size_t ninputs = inpshapes.size();
        if (addResidual)
            ninputs--;
        CV_Assert(ninputs >= 1);

        MatShape wshape = ninputs > 1 ? inpshapes[1] : wshape0;
        outshapes.assign(1, convInferShape(inpshapes[0], wshape, emptyKernelShape,
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
        if (addResidual && ninputs > 1)
            desiredInputs[ninputs - 1] = DATA_LAYOUT_BLOCK;
        outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
        return getNetImpl(this)->defaultC0;
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays input_arrs,
                 OutputArrayOfArrays output_arrs,
                 OutputArrayOfArrays temp_arrs) CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        std::vector<Mat>* temp_mats = &temp_arrs.getMatVecRef();
        temp_mats->resize(2);
        int ninputs = (int)input_arrs.total();
        CV_Assert(ninputs >= 1);
        const Mat& inp = input_arrs.getMat(0);
        Mat residual;
        const void* resptr = nullptr;
        int inptype = inp.type();
        MatShape inpshape = inp.shape();
        CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);
        CV_Assert(inp.isContinuous());

        if (addResidual) {
            residual = input_arrs.getMat(ninputs-1);
            resptr = residual.data;
            ninputs--;
        }

        bool dynamicWeights = false;
        for (int i = 1; i < ninputs; i++) {
            if (!netimpl_->isConstArg(inputs[i]))
                dynamicWeights = true;
        }
        if (dynamicWeights || weights.empty()) {
            setWeights(input_arrs.getMat(1), ninputs > 2 ? input_arrs.getMat(2) : Mat(),
                       inpshape.back(), netimpl_->accuracy);
        }

        MatShape outshape = convInferShape(inpshape, wshape0, emptyKernelShape,
                                           ngroups, strides, dilations,
                                           pads, auto_pad, ceil_mode);
        int outtype = inferType(inptype);
        int C0 = inpshape.back();
        int outkind = output_arrs.kind();
        CV_Assert(outkind == _InputArray::STD_VECTOR_MAT ||
                  outkind == _InputArray::STD_VECTOR_UMAT);

        if (addResidual && (residual.size != outshape || residual.type() != outtype))
        {
            CV_Error(Error::StsBadArg,
                    "residual added after convolution must have the same shape and the "
                    "same type as the convolution output. If this error occurs, the only "
                    "solution for now is to edit the model and add 'Expand' and/or 'Cast' "
                    "operators to make the residual tensor match the convolution shape and type");
        }

        int nspatialdims = inpshape.dims - 3;
        CV_Assert(wshape0.dims == nspatialdims+2);

        if (inpshape != prevInpshape) {
            cs.initConv(inpshape, wshape0, outshape, ngroups,
                        strides, dilations, pads, auto_pad, ceil_mode,
                        fastActivation, activationFunc, activParams);
            prevInpshape = inpshape;
        }

        const float* scale_data = nullptr;
        const float* bias_data = bias.ptr<float>();

        if (fusedBatchNorm) {
            scale_data = fusedScale.ptr<float>();
            bias_data = fusedBias.ptr<float>();
        }

        std::vector<Mat>* outs = nullptr;
        std::vector<UMat>* uouts = nullptr;
        Mat out;

        if (outkind == _InputArray::STD_VECTOR_MAT) {
            outs = &output_arrs.getMatVecRef();
            outs->resize(1);
            outs->at(0).fit(outshape, outtype);
            out = outs->at(0);
        } else {
            uouts = &output_arrs.getUMatVecRef();
            uouts->resize(1);
            uouts->at(0).fit(outshape, outtype);
            out.fit(outshape, outtype);
        }

        const void* inptr = inp.data;
        void* outptr = out.data;
        const void* wptr = weights.data;

        // MLAS 1x1 SGEMM path. Skipped for small spatial: the gather/scatter
        // tax exceeds MLAS's SGEMM speedup over the in-place NCHWc8 kernel.
        if (mlas1x1Enabled() && inptype == CV_32F &&
            !activationFunc && !addResidual && inpshape.back() == 8)
        {
            const int ndims = inpshape.dims;
            int HW = 1;
            for (int i = 2; i < ndims - 1; i++) HW *= inpshape[i];
            constexpr int MLAS_MIN_SPATIAL = 256;
            if (HW >= MLAS_MIN_SPATIAL) {
                forwardMlas1x1(inp, out);
                if (uouts) out.copyTo(uouts->at(0));
                if (dynamicWeights) weights.release();
                return;
            }
        }

        ConvFunc func = cs.depthwise ? getDepthwiseConvFunc(inptype) : getConvFunc(inptype, C0);
        CV_Assert(func != nullptr);
        func(inptr, resptr, outptr, cs, wptr, scale_data, bias_data);

        if (uouts) {
            out.copyTo(uouts->at(0));
        }

        if (dynamicWeights) {
            // to keep memory footprint low in the case of
            // very rare situation of dynamic convolution weights,
            // we release temporarily allocated and reordered copy of the weights
            weights.release();
        }
    }

    // True iff setWeights() armed the MLAS 1x1 SGEMM path for this conv.
    bool mlas1x1Enabled() const {
        return !mlas_packed_B_.empty() && !addResidual && mlasAvailable();
    }

    // NCHWc8 -> row-major (M_chunk, Cin); (c1-outer, mi-inner) so each
    // thread streams one C-block.
    static void gatherNCHWcToNHWC(const float* inp, int C1, int HW,
                                  int p_start, int p_end,
                                  float* dst, int Cin)
    {
        const int M = p_end - p_start;
        cv::parallel_for_(cv::Range(0, C1), [&](const cv::Range& r) {
            for (int c1 = r.start; c1 < r.end; c1++) {
                const int c_base = c1 * 8;
                const int n_valid_in = std::min(8, Cin - c_base);
                const float* src = inp + (size_t)c1 * (size_t)HW * 8
                                       + (size_t)p_start * 8;
                float* dst_col = dst + c_base;
                int mi = 0;
                if (n_valid_in == 8) {
#if CV_SIMD256 && defined(__AVX2__)
                    for (; mi + 8 <= M; mi += 8) {
                        __m256 r0 = _mm256_loadu_ps(src + 0 * 8);
                        __m256 r1 = _mm256_loadu_ps(src + 1 * 8);
                        __m256 r2 = _mm256_loadu_ps(src + 2 * 8);
                        __m256 r3 = _mm256_loadu_ps(src + 3 * 8);
                        __m256 r4 = _mm256_loadu_ps(src + 4 * 8);
                        __m256 r5 = _mm256_loadu_ps(src + 5 * 8);
                        __m256 r6 = _mm256_loadu_ps(src + 6 * 8);
                        __m256 r7 = _mm256_loadu_ps(src + 7 * 8);
                        _mm256_storeu_ps(dst_col + (size_t)(mi + 0) * Cin, r0);
                        _mm256_storeu_ps(dst_col + (size_t)(mi + 1) * Cin, r1);
                        _mm256_storeu_ps(dst_col + (size_t)(mi + 2) * Cin, r2);
                        _mm256_storeu_ps(dst_col + (size_t)(mi + 3) * Cin, r3);
                        _mm256_storeu_ps(dst_col + (size_t)(mi + 4) * Cin, r4);
                        _mm256_storeu_ps(dst_col + (size_t)(mi + 5) * Cin, r5);
                        _mm256_storeu_ps(dst_col + (size_t)(mi + 6) * Cin, r6);
                        _mm256_storeu_ps(dst_col + (size_t)(mi + 7) * Cin, r7);
                        src += 8 * 8;
                    }
#endif
                    for (; mi < M; mi++) {
                        std::memcpy(dst_col + (size_t)mi * Cin, src, 8 * sizeof(float));
                        src += 8;
                    }
                } else {
                    const size_t copy_bytes = (size_t)n_valid_in * sizeof(float);
                    for (; mi < M; mi++) {
                        std::memcpy(dst_col + (size_t)mi * Cin, src, copy_bytes);
                        src += 8;
                    }
                }
            }
        });
    }

    // Row-major (M_chunk, Cout) -> NCHWc8 with fused BN + bias + activation
    // in the same pass; writes stream within one C-block.
    void scatterAndActivate(const float* src, int M, int p_start,
                            int K1, int HW, float* out) const
    {
        const int Cout = mlas_packed_M_;
        const float* biasptr = bias.empty() ? nullptr : bias.ptr<float>();
        const float* scaleptr = fusedBatchNorm ? fusedScale.ptr<float>() : nullptr;
        const float* fbiasptr = fusedBatchNorm ? fusedBias.ptr<float>() : biasptr;
        const FastActivation act = fastActivation;
        // PReLU has per-channel slopes; LEAKY_RELU broadcasts a single one.
        const float* alphaptr = (act == FAST_ACTIV_PRELU) ? activParams.data() : nullptr;
        const float leaky_alpha = (act == FAST_ACTIV_LEAKY_RELU) ? activParams[0] : 0.f;
        const float clip_min = (act == FAST_ACTIV_CLIP) ? activParams[0] : -FLT_MAX;
        const float clip_max = (act == FAST_ACTIV_CLIP) ? activParams[1] :  FLT_MAX;

        cv::parallel_for_(cv::Range(0, K1), [&](const cv::Range& r) {
            for (int c1 = r.start; c1 < r.end; c1++) {
                const int c_base = c1 * 8;
                const int n_valid = std::min(8, Cout - c_base);
                float* dst = out + (size_t)c1 * (size_t)HW * 8
                                 + (size_t)p_start * 8;
                const float* row = src + c_base;

                CV_DECL_ALIGNED(32) float scalebuf[8] = {0};
                CV_DECL_ALIGNED(32) float biasbuf[8]  = {0};
                CV_DECL_ALIGNED(32) float alphabuf[8] = {0};
                for (int c0 = 0; c0 < n_valid; c0++) {
                    scalebuf[c0] = scaleptr ? scaleptr[c_base + c0] : 1.f;
                    biasbuf[c0]  = fbiasptr ? fbiasptr[c_base + c0] : 0.f;
                    alphabuf[c0] = alphaptr ? alphaptr[c_base + c0] : leaky_alpha;
                }

#if CV_SIMD256 && defined(__AVX2__)
                if (n_valid == 8) {
                    const __m256 vscale = _mm256_loadu_ps(scalebuf);
                    const __m256 vbias  = _mm256_loadu_ps(biasbuf);
                    const __m256 valpha = _mm256_loadu_ps(alphabuf);
                    const __m256 vzero  = _mm256_setzero_ps();
                    const __m256 vlo    = _mm256_set1_ps(clip_min);
                    const __m256 vhi    = _mm256_set1_ps(clip_max);
                    for (int mi = 0; mi < M; mi++) {
                        __m256 v = _mm256_loadu_ps(row + (size_t)mi * Cout);
                        if (scaleptr) {
                            v = _mm256_fmadd_ps(v, vscale, vbias);
                        } else if (fbiasptr) {
                            v = _mm256_add_ps(v, vbias);
                        }
                        if (act == FAST_ACTIV_RELU) {
                            v = _mm256_max_ps(v, vzero);
                        } else if (act == FAST_ACTIV_LEAKY_RELU || act == FAST_ACTIV_PRELU) {
                            __m256 neg = _mm256_mul_ps(v, valpha);
                            __m256 mask = _mm256_cmp_ps(v, vzero, _CMP_GE_OQ);
                            v = _mm256_blendv_ps(neg, v, mask);
                        } else if (act == FAST_ACTIV_CLIP) {
                            v = _mm256_min_ps(_mm256_max_ps(v, vlo), vhi);
                        }
                        _mm256_storeu_ps(dst + (size_t)mi * 8, v);
                    }
                } else
#endif
                {
                    for (int mi = 0; mi < M; mi++) {
                        for (int c0 = 0; c0 < 8; c0++) {
                            float v = (c0 < n_valid) ? row[(size_t)mi * Cout + c0] : 0.f;
                            if (scaleptr) v = v * scalebuf[c0] + biasbuf[c0];
                            else if (fbiasptr) v = v + biasbuf[c0];
                            if (act == FAST_ACTIV_RELU) v = v > 0.f ? v : 0.f;
                            else if (act == FAST_ACTIV_LEAKY_RELU || act == FAST_ACTIV_PRELU)
                                v = v > 0.f ? v : v * alphabuf[c0];
                            else if (act == FAST_ACTIV_CLIP)
                                v = std::min(std::max(v, clip_min), clip_max);
                            dst[(size_t)mi * 8 + c0] = v;
                        }
                    }
                }
            }
        });
        if (activationFunc) {
            float* dst = out;
            int total = K1 * HW * 8;
            activationFunc(dst, dst, total, activParams.data());
        }
    }

    // Chunked 1x1 SGEMM-as-conv: gather NCHWc8 -> SGEMM -> scatter+activate
    // per spatial chunk. Scratch is ~CHUNK*(Cin+Cout)*4 bytes.
    void forwardMlas1x1(const Mat& inp, Mat& out)
    {
        const MatShape& inpshape = inp.shape();
        const MatShape& outshape = out.shape();
        const int Cout = mlas_packed_M_;
        const int Cin  = mlas_packed_K_;
        const int ndims = inpshape.dims;
        const int N = inpshape[0];
        const int H = ndims >= 5 ? inpshape[ndims - 3] : 1;
        const int W = inpshape[ndims - 2];
        const int HW = H * W;
        const int C1_in  = (Cin  + 7) / 8;
        const int C1_out = (Cout + 7) / 8;

        CV_Assert(inpshape.back() == 8 && outshape.back() == 8);
        CV_Assert(C1_in == inpshape[1]);

        const int CHUNK = std::min(HW, 1024);
        const float* inpdata = inp.ptr<float>();
        float* outdata = out.ptr<float>();

        // Per-layer scratch, grown lazily.
        if ((int)scratch_A_.total() < CHUNK * Cin)
            scratch_A_.create(1, CHUNK * Cin, CV_32F);
        if ((int)scratch_C_.total() < CHUNK * Cout)
            scratch_C_.create(1, CHUNK * Cout, CV_32F);

        const size_t in_batch_stride  = (size_t)C1_in  * (size_t)HW * 8;
        const size_t out_batch_stride = (size_t)C1_out * (size_t)HW * 8;

        for (int n = 0; n < N; n++) {
            const float* inp_n = inpdata + (size_t)n * in_batch_stride;
            float* out_n       = outdata + (size_t)n * out_batch_stride;

            for (int p_start = 0; p_start < HW; p_start += CHUNK) {
                int p_end = std::min(p_start + CHUNK, HW);
                int M = p_end - p_start;

                gatherNCHWcToNHWC(inp_n, C1_in, HW, p_start, p_end,
                                  scratch_A_.ptr<float>(), Cin);

                // C = A @ W^T (W is (Cout, Cin), packed with trans_b=true).
                bool ok = mlasSgemmPacked(false, true, M, Cout, Cin,
                                          1.0f,
                                          scratch_A_.ptr<float>(), Cin,
                                          mlas_packed_B_.data,
                                          0.0f,
                                          scratch_C_.ptr<float>(), Cout);
                CV_Assert(ok);

                scatterAndActivate(scratch_C_.ptr<float>(), M, p_start,
                                   C1_out, HW, out_n);
            }
        }
    }

    std::vector<int> emptyKernelShape;
    Ptr<Layer> activ, batchNorm;
    Mat weights, bias, fusedScale, fusedBias;
    MatShape wshape0, prevInpshape;
    ConvState cs;
    bool fusedBatchNorm;
    FastActivation fastActivation;
    ActivationFunc activationFunc;
    std::vector<float> activParams;
    bool addResidual;

    // MLAS 1x1 fast path: prepacked weight + per-layer scratch.
    Mat mlas_packed_B_;
    Mat scratch_A_;
    Mat scratch_C_;
    int mlas_packed_M_ = 0;   // == Cout
    int mlas_packed_K_ = 0;   // == Cin
};

Ptr<Conv2Layer> Conv2Layer::create(const LayerParams& params)
{
    return Ptr<Conv2Layer>(new Conv2LayerImpl(params));
}

}}
