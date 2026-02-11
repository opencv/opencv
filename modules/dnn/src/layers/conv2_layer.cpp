// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include "conv2_common.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{
namespace dnn
{

/*
    Convolution layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Conv.html

    Opset's 1 to 22 are covered.
*/

static void refConv32f(const void* inp__, const void* residual__, void* out__,
                       const ConvState& cs, const void* weights__,
                       const float* scale__, const float* bias__,
                       const int32_t* inpofs__, const int32_t* ofsofs__)
{
    int C0_ = cs.inpshape.back();
    int K1_ = cs.outshape[1];

    CV_Assert(C0_ == 8);
    CV_Assert(cs.activation == nullptr || cs.fastActivation == FAST_ACTIV_NONE);
    CV_Assert(0 <= cs.nspatialdims && cs.nspatialdims <= ConvState::MAX_CONV_DIMS);
    
    parallel_for_(Range(0, K1_), [&](const Range& range) {
        int sdims = cs.nspatialdims;
        int C0 = C0_, K0 = C0_;
        int K1 = K1_, C1 = cs.inpshape[1];
        int K = cs.outshape.C, C = cs.inpshape.C;
        int N = cs.inpshape[0];
        
        int Di = sdims > 2 ? cs.inpshape[sdims - 1] : 1;
        int Hi = sdims > 1 ? cs.inpshape[sdims] : 1;
        int Wi = cs.inpshape[sdims + 1];
        int D = sdims > 2 ? cs.outshape[sdims - 1] : 1;
        int H = sdims > 1 ? cs.outshape[sdims] : 1;
        int W = cs.outshape[sdims + 1];
        int iplanesize = Di*Hi*Wi*C0;
        int planesize = D*H*W*C0;
        
        int ngroups = cs.ngroups;
        int Cg = C/ngroups, Kg = K/ngroups;
        int WCg = (Cg + C0 - 1)/C0;
        int Sz = cs.strides[0], Sy = cs.strides[1], Sx = cs.strides[2];
        int Dz = cs.dilations[0], Dy = cs.dilations[1], Dx = cs.dilations[2];
        int padz = cs.pads[0], pady = cs.pads[1], padx = cs.pads[2];
        
        const float* scale_ = scale__;
        const float* bias_ = bias__;
        constexpr int BLOCK_SIZE = 8;
        
        AutoBuffer<float> sumbuf(BLOCK_SIZE*K0*3);
        
        int k1start = range.start, nk1 = range.end;
        constexpr int C0 = 8, K0 = C0;
        int planesize = 1, iplanesize = 1, ksize = 1;
        int nspatialdims = cs.nspatialdims;
        int C1 = cs.inpshape[1], K1 = cs.outshape[1];
        int ngroups = cs.ngroups, K1g = K1/ngroups, C1g = C1/ngroups;
        int nC = C1g*ksize*C0*K0;
        
        float* sum = sumbuf.data();
        float* scale = sum + BLOCK_SIZE*K0;
        float* bias = sum + BLOCK_SIZE*K0*2;
        const float* inptrs[BLOCK_SIZE];
        const int32_t* ofsptrs[BLOCK_SIZE];
        FastActivation fastActivation = cs.fastActivation;
        const float* activParams = cs.activParams;
        activation_func_t activation = cs.activation;
        float maxval = fastActivation == FAST_ACTIV_CLIP ? activParams[1] : FLT_MAX;
        float alpha = fastActivation == FAST_ACTIV_LEAKY_RELU ? activParams[0] :
                    fastActivation == FAST_ACTIV_NONE ? 1.f : 0.f;

        for (int j = 0; j < BLOCK_SIZE*K0; j++) {
            scale[j] = 1.f;
            bias[j] = 0.f;
        }

        for (int k1 = range.start; k1 < range.end; k1++) {
            for (int n = 0; n < N; n++) {
                float* out = (float*)out__ + n*k1*planesize*K0;
                const float* resptr = residual__ ? (const float*)residual__ + n*k1*planesize*K0 : nullptr;
                
                
                
            }
            int n = nk/K1, k1 = nk - n*K1;
            int g = k1/K1g;
            
            const float* inp0 = (const float*)inp__ + (n*C1 + g*C1g)*iplanesize*C0;
            
            const float* wptr = (const float*)weights__ + k1*nC;

            if (scale_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        scale[b*K0 + k] = scale_[k1*K0 + k];
            }

            if (bias_) {
                for (int b = 0; b < BLOCK_SIZE; b++)
                    for (int k = 0; k < K0; k++)
                        bias[b*K0 + k] = bias_[k1*K0 + k];
            }

            for (int xy0 = 0; xy0 < planesize; xy0 += BLOCK_SIZE, out += K0*BLOCK_SIZE,
                 resptr += (resptr ? K0*BLOCK_SIZE : 0)) {
                int j = 0, blocksize = std::min(planesize - xy0, BLOCK_SIZE);

                for (; j < blocksize; j++) {
                    int jj = (xy0 + j)*2;
                    inptrs[j] = inp0 + ofsofs_[jj];
                    ofsptrs[j] = inpofs_ + ofsofs_[jj+1];
                }

                if (j < BLOCK_SIZE) {
                    const float* last_inptr = inptrs[blocksize-1];
                    const int32_t* last_ofsptr = ofsptrs[blocksize-1];
                    for (; j < BLOCK_SIZE; j++) {
                        inptrs[j] = last_inptr;
                        ofsptrs[j] = last_ofsptr;
                    }
                }

                for (int i = 0; i < BLOCK_SIZE*K0; i++)
                    sum[i] = 0.f;

                for (int c1 = 0, i = 0; c1 < nC; c1 += K0*C0, i++) {
                    for (j = 0; j < BLOCK_SIZE; j++) {
                        int32_t ofs_ij = ofsptrs[j][i];
                        const float* x = &inptrs[j][std::max(ofs_ij, 0)];
                        float mij = (float)(ofs_ij >= 0);
                        for (int c0 = 0; c0 < C0; c0++) {
                            float xc = x[c0]*mij;
                            for (int k = 0; k < K0; k++) {
                                float w = wptr[c1 + c0*K0 + k];
                                sum[K0*j + k] += xc*w;
                            }
                        }
                    }
                }

                if (activation) {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            sum[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            sum[j] = v;
                        }
                    }
                    activation(sum, out, blocksize*K0, activParams);
                } else {
                    if (resptr) {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j] + resptr[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    } else {
                        for (j = 0; j < blocksize*K0; j++) {
                            float v = sum[j]*scale[j] + bias[j];
                            v = std::min(v*(v >= 0 ? 1.f : alpha), maxval);
                            out[j] = v;
                        }
                    }
                }
            }
        }
    });
}

/*
static void refConv2d(const Mat& inp, const Mat& weights, const MatShape& wshape0, const Mat& bias, Mat& out,
                      const int* strides, const int* dilations,
                      const int* pads, FastActivation activation,
                      float activParam=0.f)
{
    CV_Assert(inp.type() == CV_32F);

    MatShape inpshape = inp.size;
    int ndims = inpshape.dims;
    CV_Assert(ndims == 4 && wshape0.dims == 4);

    MatShape outshape = fastConv2dInferShape(inpshape, wshape0, strides, dilations, pads);
    CV_Assert(wshape0[0] == outshape[1] && inpshape[0] == outshape[0]);

    out.fit(outshape, inp.type());

    parallel_for_(Range(0, wshape0[0]), [&](const Range& range) {
        int C0 = weights.size.back();
        int K = wshape0[0], WC = wshape0[1], Hk = wshape0[2], Wk = wshape0[3];
        int N = inpshape[0], C = inpshape[1];
        int Hi = inpshape[2], Wi = inpshape[3];
        int H0 = outshape[2], W0 = outshape[3];
        int ngroups = C/WC;
        int Cg = C/ngroups, Kg = K/ngroups;
        int WCg = (Cg + C0 - 1)/C0;
        int Sy = strides[1], Sx = strides[2];
        int Dy = dilations[1], Dx = dilations[2];
        int pady = pads[1], padx = pads[2];
        float minval = activation == FAST_ACTIV_RELU || activation == FAST_ACTIV_CLIP ? 0.f : -FLT_MAX;
        float maxval = activation == FAST_ACTIV_CLIP ? activParam : FLT_MAX;
        const float* wptr = weights.ptr<float>();
        const float* biasptr = bias.ptr<float>();
        const float* inptr = inp.ptr<float>();
        float* outptr = out.ptr<float>();

        for (int k = range.start; k < range.end; k++) {
            int k1 = k / C0, k0 = k % C0;
            int g = k/Kg;
            for (int n = 0; n < N; n++) {
                for (int y0 = 0; y0 < H0; y0++) {
                    for (int x0 = 0; x0 < W0; x0++) {
                        int yi_ = y0*Sy - pady;
                        int xi_ = x0*Sx - padx;
                        float s = 0.f;

                        for (int ky = 0; ky < Hk; ky++) {
                            int yi = yi_ + ky*Dy;
                            if (yi < 0 || yi >= Hi)
                                continue;

                            for (int kx = 0; kx < Wk; kx++) {
                                int xi = xi_ + kx*Dx;
                                if (xi < 0 || xi >= Wi)
                                    continue;
                                for (int c = 0; c < Cg; c++) {
                                    size_t ofs = (((n*ngroups + g)*Cg + c)*Hi + yi)*Wi + xi;
                                    float inpval = inptr[ofs];
                                    int c1 = c / C0, c0 = c % C0;
                                    float w = wptr[(((k1*WCg + c1)*Hk+ky)*Wk + kx)*C0*C0 + c0*C0 + k0];
                                    s += inpval*w;
                                }
                            }
                        }
                        if (biasptr)
                            s += biasptr[k];
                        outptr[((n*K + k)*H0 + y0)*W0 + x0] = std::min(std::max(s, minval), maxval);
                    }
                }
            }
        }
    });
}*/

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
        fused_batch_norm = false;
        fused_batch_norm = false;
        fast_activation = FAST_ACTIV_NONE;
        memset(fast_activ_params, 0, sizeof(fast_activ_params));
        add_residual = false;
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

        if (fused_batch_norm) {
            prindent(strm, indent);
            strm << "batch_norm: true,\n";
        }
        
        if (fast_activation != FAST_ACTIV_NONE || !activ.empty()) {
            prindent(strm, indent);
            strm << "fused_activation: " <<
                (fast_activation != FAST_ACTIV_NONE ? fastActivationToString(fast_activation) :
                 activ->type) << ",\n";
        }

        if (add_residual) {
            prindent(strm, indent);
            strm << "add_residual: true,\n";
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
        MatShape wshape1 = wshape0;
        bool depthwise = ngroups == wshape0[0] && wshape0[1] == 1;

        if (depthwise) {
            wshape1.layout = DATA_LAYOUT_BLOCK;
            wshape1.C = wshape1[0];
            wshape1[0] = (wshape1[0] + C0 - 1)/C0;
            for (int i = 2; i < wshape1.dims; i++)
                wshape1[i-1] = wshape1[i];
            wshape1[wshape1.dims-1] = C0;
            weights.fit(wshape1, wtype);

            repackDepthwiseConvWeights(weights_.data, wtype0, weights.data, wtype, wshape0, C0);
        } else {
            wshape1.dims += 2;
            wshape1[wshape1.dims-1] = wshape1[wshape1.dims-2] = C0;
            wshape1[0] = (wshape1[0] + C0 - 1)/C0;
            wshape1[1] = (wshape1[1] + C0 - 1)/C0;
            weights.fit(wshape1, wtype);

            repackConvWeights(weights_.data, wtype0, weights.data, wtype, wshape0, C0);
        }

        if (!bias_.empty()) {
            CV_Assert(bias_.isContinuous() && bias_.total() == wshape0[0]);
            bias_.convertTo(bias, CV_32F);
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

        fused_scale.fit(1, &K, CV_32F);
        fused_bias.fit(1, &K, CV_32F);

        const float* bn_scale_data = bn_scale.ptr<float>();
        const float* bn_bias_data = bn_bias.ptr<float>();
        float* fused_scale_data = fused_scale.ptr<float>();
        float* fused_bias_data = fused_bias.ptr<float>();

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
        if (fused_batch_norm || !bn || bn->inputs.size() > 1)
            return false;
        fuseBatchNormWeights(bn);
        fused_batch_norm = true;
        return true;
    }

    virtual bool fuseAddBias(InputArray arr) CV_OVERRIDE
    {
        if (inputs.size() > 1 || fused_batch_norm || add_residual)
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
        if (!activ_ptr || fast_activation != FAST_ACTIV_NONE)
            return false;
        ReLULayer* relu = dynamic_cast<ReLULayer*>(activ_ptr);
        if (relu) {
            float alpha = relu->negativeSlope;
            if (alpha > 0.f) {
                fast_activation = FAST_ACTIV_LEAKY_RELU;
                fast_activ_params[0] = alpha;
            } else {
                fast_activation = FAST_ACTIV_RELU;
            }
        } else {
            //activ = activlayer;
            return false;
        }
        return true;
    }
    
    virtual bool fuseAddResidual(Arg residual) CV_OVERRIDE
    {
        if (activ.empty() && fast_activation == FAST_ACTIV_NONE && !add_residual && residual.idx >= 0) {
            add_residual = true;
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
        if (add_residual)
            ninputs--;
        CV_Assert(ninputs >= 1);

        MatShape wshape = ninputs > 1 ? inpshapes[1] : wshape0;
        outshapes.assign(1, convInferShape(inpshapes[0], wshape, empty_kernel_shape,
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

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays input_arrs,
                 OutputArrayOfArrays output_arrs,
                 OutputArrayOfArrays temp_arrs) CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        DataLayout origLayout = netimpl_->originalLayout;
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
        
        if (add_residual) {
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

        MatShape outshape = convInferShape(inpshape, wshape0, empty_kernel_shape,
                                           ngroups, strides, dilations,
                                           pads, auto_pad, ceil_mode);
        int outtype = inferType(inptype);
        int C0 = inpshape.back();
        int outkind = output_arrs.kind();
        CV_Assert(outkind == _InputArray::STD_VECTOR_MAT ||
                  outkind == _InputArray::STD_VECTOR_UMAT);
        
        if (add_residual && (residual.size != outshape || residual.type() != outtype))
        {
            CV_Error(Error::StsBadArg,
                    "residual added after convolution must have the same shape and the "
                    "same type as the convolution output. If this error occurs, the only "
                    "solution for now is to edit the model and add 'Expand' and/or 'Cast' "
                    "operators to make the residual tensor match the convolution shape and type");
        }

        int nspatialdims = inpshape.dims - 3;
        CV_Assert(wshape0.dims == nspatialdims+2);

        cs.initConv(inpshape, wshape0, outshape, ngroups,
                    strides, dilations, pads, auto_pad, ceil_mode,
                    fast_activation, fast_activ_params, ConvState::MAX_ACTIV_PARAMS);
        
        const float* scale_data = nullptr;
        const float* bias_data = bias.ptr<float>();

        if (fused_batch_norm) {
            scale_data = fused_scale.ptr<float>();
            bias_data = fused_bias.ptr<float>();
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

        if (cs.depthwise) {
            DepthwiseConvFunc func = getDepthwiseConvFunc(inptype);
            CV_Assert(func != nullptr);

            func(inptr, resptr, outptr, cs, wptr, scale_data, bias_data);
        } else {
            // [TODO] add special 1x1 convolution kernels that don't need inpofs & ofsofs
            if (inpofs.empty() || !cs.sameShape(prev_cs)) {
                initConvTables(cs, inpofs, ofsofs);
                prev_cs = cs;
            }

            {
                ConvFunc func = cs.unevenGroupedConv ? refConv32f : getConvFunc(inptype, C0);
                CV_Assert(func != nullptr);
                func(inptr, resptr, outptr, cs, wptr, scale_data, bias_data,
                     inpofs.data(), ofsofs.data());

#if 0
                Mat inp0, out0, temp;
                transformLayout(inp, inp0, DATA_LAYOUT_NCHW,
                                DATA_LAYOUT_NCHW, inp.size.C);
                refConv2d(inp0, weights, wshape0, bias, out0,
                          cs.strides, cs.dilations, cs.pads,
                          fast_activation, 0.f);
                transformLayout(out0, temp, DATA_LAYOUT_BLOCK,
                                DATA_LAYOUT_NCHW, out.size.back());
                double err = norm(temp, out, NORM_INF);
                CV_Assert(err < 1e-4);
#endif
            }
        }

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

    std::vector<int> empty_kernel_shape;
    Ptr<Layer> activ, batchNorm;
    Mat weights, bias, fused_scale, fused_bias;
    MatShape wshape0;
    ConvState cs, prev_cs;
    std::vector<int32_t> inpofs;
    std::vector<int32_t> ofsofs;
    bool fused_batch_norm;
    FastActivation fast_activation;
    float fast_activ_params[ConvState::MAX_ACTIV_PARAMS];
    bool add_residual;
};

Ptr<Conv2Layer> Conv2Layer::create(const LayerParams& params)
{
    return Ptr<Conv2Layer>(new Conv2LayerImpl(params));
}

}}
