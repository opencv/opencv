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

#if 0
static MatShape fastConv2dInferShape(const MatShape& inpshape, const MatShape& wshape,
                                     const int* strides,
                                     const int* dilations,
                                     const int* pads)
{
    int ndims = inpshape.dims, wdims = wshape.dims;
    CV_Assert(ndims == 4 && wdims == 4);

    int N = inpshape[0], C = inpshape[1], H = inpshape[2], W = inpshape[3];
    int K = wshape[0], WCg = wshape[1], Hk = wshape[2], Wk = wshape[3];
    int ngroups = C/WCg;

    int pad_y = pads[1] + pads[4];
    int pad_x = pads[2] + pads[5];
    int H0 = (H + pad_y - dilations[1]*(Hk - 1) - 1)/strides[1] + 1;
    int W0 = (W + pad_x - dilations[2]*(Wk - 1) - 1)/strides[2] + 1;
    MatShape outshape(4);
    outshape.layout = inpshape.layout;
    outshape[0] = inpshape[0];
    outshape[1] = K;
    outshape[2] = H0;
    outshape[3] = W0;

    return outshape;
}

static void refConv2d(const Mat& inp, const Mat& weights0, const Mat& weights, const Mat& bias, Mat& out,
                      const int* strides, const int* dilations,
                      const int* pads, FastActivation activation,
                      float activParam=0.f)
{
    CV_Assert(inp.type() == CV_32F);

    MatShape inpshape = inp.size, wshape = weights0.size;
    int ndims = inpshape.dims, wdims = wshape.dims;
    CV_Assert(ndims == 4 && wdims == 4);

    MatShape outshape = fastConv2dInferShape(inpshape, wshape, strides, dilations, pads);
    CV_Assert(wshape[0] == outshape[1] && inpshape[0] == outshape[0]);

    out.fit(outshape, inp.type());

    parallel_for_(Range(0, wshape[0]), [&](const Range& range) {
        int C0 = weights.size.back();
        int K = wshape[0], WC = wshape[1], Hk = wshape[2], Wk = wshape[3];
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
        const float* wptr0 = weights0.ptr<float>();
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
                                    float w0 = wptr0[((k*Cg + c)*Hk+ky)*Wk + kx];
                                    float w = wptr[(((k1*WCg + c1)*Hk+ky)*Wk + kx)*C0*C0 + c0*C0 + k0];
                                    CV_Assert(w == w0);
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
}
#endif

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
        add_residual = false;
        fused_batch_norm = false;
        fast_activation = FAST_ACTIV_NONE;
        memset(fast_activ_params, 0, sizeof(fast_activ_params));
    }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        prindent(strm, indent);
        strm << "group: " << ngroups << ",\n";

        /*prindent(strm, indent);
        strm << "ksizes: [";
        for (int k = 0; k < wshape0.ndims; k++)
            strm << (k > 0 ? ", " : "") << wshape0.size[k];
        strm << "],\n";*/

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
        if (dynamic_cast<ReLULayer*>(activ_ptr)) {
            fast_activation = FAST_ACTIV_RELU;
        }
        activ = activlayer;
        return true;
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

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        int ninputs = (int)inputs_arr.total();
        CV_Assert(ninputs >= 1);
        const Mat& inp = inputs_arr.getMat(0);
        Mat residual;
        const void* resptr = nullptr;
        int inptype = inp.type();
        MatShape inpshape = inp.shape();
        CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);
        CV_Assert(inp.isContinuous());

        if (add_residual) {
            residual = inputs_arr.getMat(ninputs-1);
            resptr = residual.data;
            ninputs--;
        }

        bool dynamic_weights = false;
        for (int i = 1; i < ninputs; i++) {
            if (!netimpl_->isConstArg(inputs[i]))
                dynamic_weights = true;
        }
        if (dynamic_weights || weights.empty()) {
            setWeights(inputs_arr.getMat(1), ninputs > 2 ? inputs_arr.getMat(2) : Mat(),
                       inpshape.back(), netimpl_->accuracy);
        }

        MatShape outshape = convInferShape(inpshape, wshape0, empty_kernel_shape,
                                           ngroups, strides, dilations,
                                           pads, auto_pad, ceil_mode);
        int outtype = inferType(inptype);
        int C0 = inpshape.back();
        int outkind = outputs_arr.kind();
        CV_Assert(outkind == _InputArray::STD_VECTOR_MAT ||
                  outkind == _InputArray::STD_VECTOR_UMAT);

        if (add_residual) {
            CV_Assert(outshape == residual.shape());
            CV_Assert(outtype == residual.type());
        }

        int nspatialdims = inpshape.dims - 3;
        CV_Assert(wshape0.dims == nspatialdims+2);

        cs.initConv(inpshape, wshape0, outshape, ngroups,
                    strides, dilations, pads, auto_pad, ceil_mode,
                    fast_activation, fast_activ_params, 0);
        //bool conv1x1 = cs.kshape[0]*cs.kshape[1]*cs.kshape[2] == 1;
        bool depthwise = ngroups == cs.inpshape.C;

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
            outs = &outputs_arr.getMatVecRef();
            outs->resize(1);
            outs->at(0).fit(outshape, outtype);
            out = outs->at(0);
        } else {
            uouts = &outputs_arr.getUMatVecRef();
            uouts->resize(1);
            uouts->at(0).fit(outshape, outtype);
            out.fit(outshape, outtype);
        }

        const void* inptr = inp.data;
        void* outptr = out.data;
        const void* wptr = weights.data;

        if (depthwise) {
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
                ConvFunc func = getConvFunc(inptype, C0);
                CV_Assert(func != nullptr);
                func(inptr, resptr, outptr, cs, wptr, scale_data, bias_data,
                     inpofs.data(), ofsofs.data());
            }
        }

        if (uouts) {
            out.copyTo(uouts->at(0));
        }

        if (dynamic_weights) {
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
};

Ptr<Conv2Layer> Conv2Layer::create(const LayerParams& params)
{
    return Ptr<Conv2Layer>(new Conv2LayerImpl(params));
}

}}
