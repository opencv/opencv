// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include "conv2_common.hpp"

namespace cv
{
namespace dnn
{

/*
    ONNX Det operator
    Spec: https://onnx.ai/onnx/operators/onnx__ConvTranspose.html
    Supported opsets: 1-22
*/

class ConvTranspose2LayerImpl : public ConvTranspose2Layer
{
public:
    ConvTranspose2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        auto_pad = getAutoPadding(params);
        strides = params.getVector<int>("stride");
        dilations = params.getVector<int>("dilation");
        pads = params.getVector<int>("pad");
        adjust_pads = params.getVector<int>("adj");
        ngroups = params.get<int>("group", 1);
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

        if (!adjust_pads.empty()) {
            prindent(strm, indent);
            strm << "adj: [";
            for (size_t k = 0; k < adjust_pads.size(); k++)
                strm << (k > 0 ? ", " : "") << adjust_pads[k];
            strm << "],\n";
        }

        return strm;
    }

    int inferType(int inptype0) const
    {
        return inptype0;
    }

    virtual void setWeights(InputArray weights_arr, InputArray bias_arr,
                            int C0, int /*accuracy*/) CV_OVERRIDE
    {
        Mat rawWeights = weights_arr.getMat();
        Mat rawBias = bias_arr.getMat();
        CV_Assert(!rawWeights.empty());

        wshape0 = rawWeights.shape();

        Mat wfloat;
        if (rawWeights.type() != CV_32F)
            rawWeights.convertTo(wfloat, CV_32F);
        else
            wfloat = rawWeights;
        repackDeconvWeights(wfloat, weights, CV_32F, ngroups, C0);

        if (!rawBias.empty())
            rawBias.convertTo(bias, CV_32F);
    }

    virtual bool fuseAddBias(InputArray arr) CV_OVERRIDE
    {
        if (inputs.size() > 1)
            return false;
        Mat new_bias = arr.getMat();
        CV_Assert(new_bias.isContinuous() && new_bias.dims == 1);
        if (new_bias.type() != CV_32F) {
            Mat temp;
            new_bias.convertTo(temp, CV_32F);
            new_bias = temp;
        }
        if (!bias.empty()) {
            CV_Assert(bias.shape() == new_bias.shape());
            add(bias, new_bias, bias);
        } else {
            new_bias.copyTo(bias);
        }
        return true;
    }

    virtual int64_t getFLOPS(const std::vector<MatShape>& inputs,
                             const std::vector<MatShape>& outputs) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 1);
        CV_Assert(outputs.size() == 1);
        MatShape wshape = inputs.size() > 1 ? inputs[1] : wshape0;
        int K = wshape[1] * ngroups;
        size_t ksize = wshape.total() / (wshape[0] * wshape[1]);
        int C = inputs[0][1] * inputs[0].back();
        return (int64_t)((inputs[0].total() / C) * ksize * K);
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
                                 std::vector<MatShape>& outshapes,
                                 std::vector<MatShape>& tempshapes) const CV_OVERRIDE
    {
        size_t ninputs = inpshapes.size();
        CV_Assert(ninputs >= 1);

        MatShape wshape = ninputs > 1 ? inpshapes[1] : wshape0;
        outshapes.assign(1, deconvInferShape(inpshapes[0], wshape, emptyKernelShape,
                                             ngroups, strides, dilations,
                                             pads, adjust_pads, auto_pad));
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

    void finalize(InputArrayOfArrays, OutputArrayOfArrays) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays input_arrs,
                 OutputArrayOfArrays output_arrs,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        int ninputs = (int)input_arrs.total();
        CV_Assert(ninputs >= 1);
        const Mat& inp = input_arrs.getMat(0);
        int inptype = inp.type();
        MatShape inpshape = inp.shape();
        CV_Assert(inpshape.layout == DATA_LAYOUT_BLOCK);
        CV_Assert(inp.isContinuous());

        bool dynamicWeights = false;
        for (int i = 1; i < ninputs; i++) {
            if (!netimpl_->isConstArg(inputs[i]))
                dynamicWeights = true;
        }
        if (dynamicWeights || weights.empty()) {
            setWeights(input_arrs.getMat(1), ninputs > 2 ? input_arrs.getMat(2) : Mat(),
                       inpshape.back(), netimpl_->accuracy);
        }

        MatShape outshape = deconvInferShape(inpshape, wshape0, emptyKernelShape,
                                             ngroups, strides, dilations,
                                             pads, adjust_pads, auto_pad);

        // compute actual pads for SAME/VALID auto-padding
        int nsd = inpshape.dims - 3;
        std::vector<int> pads_resolved = pads;
        if (auto_pad != AUTO_PAD_NONE) {
            pads_resolved.resize(nsd * 2, 0);
            for (int i = 0; i < nsd; i++) {
                int inpsz   = inpshape[2 + i];
                int outsz   = outshape[2 + i];
                int adj_i   = adjust_pads.empty() ? 0 : adjust_pads[i];
                int stride  = strides.empty() ? 1 : strides[i];
                int dil     = dilations.empty() ? 1 : dilations[i];
                int ki      = wshape0[2 + i];
                int total   = (inpsz - 1) * stride + dil * (ki - 1) + 1 + adj_i - outsz;
                int pb;
                if (auto_pad == AUTO_PAD_SAME_UPPER && stride <= ki * dil) {
                    pb = std::max((total - (outsz - 1 + stride) % stride) / 2, 0);
                } else {
                    pb = total / 2;
                }
                pads_resolved[i]       = pb;
                pads_resolved[nsd + i] = total - pb;
            }
        }

        int outtype = inferType(inptype);

        if (inpshape != prevInpshape) {
            cs.initDeconv(inpshape, wshape0, outshape, ngroups,
                          strides, dilations, pads_resolved);
            prevInpshape = inpshape;
        }

        int outkind = output_arrs.kind();
        CV_Assert(outkind == _InputArray::STD_VECTOR_MAT ||
                  outkind == _InputArray::STD_VECTOR_UMAT);

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

        DeconvFunc func = getDeconvFunc(inptype);
        CV_Assert(func != nullptr);
        const float* bias_data = bias.empty() ? nullptr : bias.ptr<float>();
        func(inp.data, nullptr, out.data, cs, weights.data, nullptr, bias_data);

        if (uouts) {
            out.copyTo(uouts->at(0));
        }

        if (dynamicWeights) {
            weights.release();
        }
    }

    std::vector<int> emptyKernelShape;
    Mat weights, bias;
    MatShape wshape0, prevInpshape;
    ConvState cs;
};

Ptr<ConvTranspose2Layer> ConvTranspose2Layer::create(const LayerParams& params)
{
    return Ptr<ConvTranspose2Layer>(new ConvTranspose2LayerImpl(params));
}

}}
