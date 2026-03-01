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

    std::vector<int> empty_kernel_shape;
    Ptr<Layer> activ, batchNorm;
    Mat weights, bias, fused_scale, fused_bias;
    MatShape wshape0;
    ConvState cs;
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
