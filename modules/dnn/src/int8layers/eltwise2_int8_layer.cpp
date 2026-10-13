// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "int8_eltwise_kernels.hpp"

namespace cv {
namespace dnn {

static Mat broadcastExpand(const Mat& src, const MatShape& dstShape)
{
    MatShape srcShape = src.shape();
    const int ndst = (int)dstShape.size();
    while ((int)srcShape.size() < ndst)
        srcShape.insert(srcShape.begin(), 1);
    if (srcShape == dstShape)
        return src;

    MatShape outShape = dstShape;
    Mat out(outShape, src.type());
    const int total = (int)out.total();

    std::vector<int> dstStrides(ndst);
    { int s = 1; for (int d = ndst-1; d >= 0; d--) { dstStrides[d] = s; s *= dstShape[d]; } }

    std::vector<int> srcStrides(ndst, 0);
    { int s = 1; for (int d = ndst-1; d >= 0; d--) {
        srcStrides[d] = (srcShape[d] > 1) ? s : 0; s *= srcShape[d]; } }

    const uint8_t* sp = src.ptr<uint8_t>();
    uint8_t* dp = out.ptr<uint8_t>();
    for (int i = 0; i < total; i++) {
        int rem = i, srcIdx = 0;
        for (int d = 0; d < ndst; d++) {
            int coord = rem / dstStrides[d]; rem %= dstStrides[d];
            srcIdx += coord * srcStrides[d];
        }
        dp[i] = sp[srcIdx];
    }
    return out;
}

class Eltwise2Int8LayerImpl CV_FINAL : public Eltwise2Int8Layer
{
public:
    enum class Op { ADD, MUL };
    Op op;

    std::vector<float> coeffs;
    float offset;
    float mulCoeff;
    bool withRelu;
    //! Zero point the sum is requantized to before the LUT; output_zp later becomes the activation's.
    int requantZp = 0;

    Mat activationLUT;
    Ptr<ActivationLayerInt8> activ;

    Eltwise2Int8LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        output_zp = params.get<int>("zeropoints", 0);
        output_sc = params.get<float>("scales", 1.0f);
        withRelu = params.get<bool>("with_relu", false);

        std::string opStr = params.get<String>("operation", "add");
        op = (opStr == "mul" || opStr == "prod") ? Op::MUL : Op::ADD;

        if (params.has("input_scales"))
        {
            DictValue sc = params.get("input_scales");
            int n = sc.size();
            scales.resize(n);
            for (int i = 0; i < n; i++)
                scales[i] = sc.get<float>(i);
        }

        if (params.has("input_zeropoints"))
        {
            DictValue zp = params.get("input_zeropoints");
            int n = zp.size();
            zeropoints.resize(n);
            for (int i = 0; i < n; i++)
                zeropoints[i] = zp.get<int>(i);
        }

        offset = 0.f;
        mulCoeff = 0.f;
    }

    Eltwise2Int8LayerImpl(const Eltwise2Int8Params& p)
    {
        name = p.name;
        type = "Eltwise2Int8";
        scales = p.input_scales;
        zeropoints = p.input_zeropoints;
        output_sc = p.output_sc;
        output_zp = p.output_zp;
        withRelu = p.with_relu;
        op = (p.operation == "mul" || p.operation == "prod") ? Op::MUL : Op::ADD;
        offset = 0.f;
        mulCoeff = 0.f;
    }

    void ensureCoeffs()
    {
        if (!coeffs.empty() || scales.empty())
            return;

        CV_CheckEQ(scales.size(), zeropoints.size(),
                   "Eltwise2Int8: scales and zeropoints sizes must match");
        CV_Assert(output_sc > 0.f);

        requantZp = output_zp;
        coeffs.resize(scales.size());
        if (op == Op::MUL) {
            CV_Assert(scales.size() == 2);
            mulCoeff = (scales[0] * scales[1]) / output_sc;
            coeffs[0] = (float)zeropoints[0];
            coeffs[1] = (float)zeropoints[1];
            offset = (float)output_zp;
        } else {
            offset = (float)output_zp;
            for (size_t i = 0; i < scales.size(); i++)
            {
                coeffs[i] = scales[i] / output_sc;
                offset -= coeffs[i] * zeropoints[i];
            }
        }
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);
        int ndims = 0;
        DataLayout outLayout = DATA_LAYOUT_UNKNOWN;
        int outC = 0;
        for (const auto& s : inputs) {
            ndims = std::max(ndims, (int)s.size());
            if (s.layout != DATA_LAYOUT_UNKNOWN && outLayout == DATA_LAYOUT_UNKNOWN)
                outLayout = s.layout;
            if (s.C > outC) outC = s.C;
        }
        MatShape outShape(ndims, 1);
        for (const auto& s : inputs) {
            int off = ndims - (int)s.size();
            for (int d = 0; d < (int)s.size(); d++)
                outShape[d + off] = std::max(outShape[d + off], s[d]);
        }
        outShape.layout = outLayout;
        outShape.C = outC;
        outputs.assign(1, outShape);
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        outputs.assign(requiredOutputs, inputs[0]);
        internals.clear();
    }

    int getLayouts(const std::vector<DataLayout>& actualInputs,
                   std::vector<DataLayout>& desiredInputs,
                   const int requiredOutputs,
                   std::vector<DataLayout>& outputs) const CV_OVERRIDE
    {
        auto* netimpl_ = getNetImpl(this);
        DataLayout defaultLayout = netimpl_->originalLayout;
        size_t ninputs = actualInputs.size(), nblockInputs = 0;
        CV_Assert(ninputs >= 1u);
        for (size_t i = 0; i < ninputs; i++)
            nblockInputs += actualInputs[i] == DATA_LAYOUT_BLOCK;

        desiredInputs = actualInputs;
        if (nblockInputs == ninputs) {
            outputs.assign(requiredOutputs, DATA_LAYOUT_BLOCK);
        } else {
            if (nblockInputs < ninputs) {
                for (size_t i = 0; i < ninputs; i++) {
                    DataLayout layout = actualInputs[i];
                    desiredInputs[i] = layout == DATA_LAYOUT_BLOCK ? defaultLayout : layout;
                }
            }
            outputs.assign(requiredOutputs, DATA_LAYOUT_UNKNOWN);
        }
        return outputs[0] == DATA_LAYOUT_BLOCK ? netimpl_->defaultC0 : 0;
    }

    void forward(InputArrayOfArrays input_arrs,
                 OutputArrayOfArrays output_arrs,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        ensureCoeffs();

        int ninputs = (int)input_arrs.total();
        CV_Assert(ninputs >= 2 && coeffs.size() == (size_t)ninputs);

        std::vector<Mat> inputs(ninputs);
        DataLayout outLayout = DATA_LAYOUT_UNKNOWN;
        int outC = 0;
        {
            int ndims = 0;
            for (int k = 0; k < ninputs; k++) {
                inputs[k] = input_arrs.getMat(k);
                const MatShape& sh = inputs[k].shape();
                ndims = std::max(ndims, (int)sh.size());
                if (sh.layout != DATA_LAYOUT_UNKNOWN && outLayout == DATA_LAYOUT_UNKNOWN)
                    outLayout = sh.layout;
                if (sh.C > outC) outC = sh.C;
            }
            MatShape broadShape(ndims, 1);
            for (int k = 0; k < ninputs; k++) {
                const MatShape& sh = inputs[k].shape();
                int off = ndims - (int)sh.size();
                for (int d = 0; d < (int)sh.size(); d++)
                    broadShape[d + off] = std::max(broadShape[d + off], sh[d]);
            }
            broadShape.layout = outLayout;
            broadShape.C = outC;
            for (int k = 0; k < ninputs; k++)
                if (inputs[k].shape() != broadShape)
                    inputs[k] = broadcastExpand(inputs[k], broadShape);
        }

        const Mat& inp0 = inputs[0];
        MatShape outshape = inp0.shape();
        if (outshape.layout == DATA_LAYOUT_UNKNOWN && outLayout != DATA_LAYOUT_UNKNOWN) {
            outshape.layout = outLayout;
            outshape.C = outC;
        }
        int outtype = inp0.type();
        CV_Assert(outtype == CV_8SC1 || outtype == CV_8UC1);
        const bool isU8 = (outtype == CV_8UC1);

        int outkind = output_arrs.kind();
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

        const int total_elems = (int)inp0.total();
        const float* cptr = coeffs.data();
        const uchar* lutptr = !activationLUT.empty() ? activationLUT.ptr<uchar>() : nullptr;
        const bool isMul = (op == Op::MUL);

        // For MUL, ensureCoeffs() left the input zero points in coeffs and the output one in offset.
        const int out_min = withRelu ? requantZp : (isU8 ? 0 : -128);
        const int out_max = isU8 ? 255 : 127;

        std::vector<const uchar*> inptrs(ninputs);
        for (int k = 0; k < ninputs; k++)
            inptrs[k] = inputs[k].ptr<uchar>();
        uchar* outptr = out.ptr<uchar>();

        const double nstripes = (double)std::max(1, getNumThreads() * 4);
        parallel_for_(Range(0, total_elems), [&](const Range& r) {
            const int len = r.end - r.start;
            if (ninputs == 2) {
                const uchar* p0 = inptrs[0] + r.start;
                const uchar* p1 = inptrs[1] + r.start;
                if (isMul)
                    mulInt8(p0, p1, outptr + r.start, len, isU8, cptr[0], cptr[1],
                            mulCoeff, offset, out_min, out_max, lutptr);
                else
                    addInt8(p0, p1, outptr + r.start, len, isU8, cptr[0], cptr[1],
                            offset, out_min, out_max, lutptr);
                return;
            }
            CV_Assert(!isMul);
            for (int i = r.start; i < r.end; i++) {
                float val = offset;
                for (int k = 0; k < ninputs; k++)
                    val += cptr[k] * (isU8 ? (float)inptrs[k][i] : (float)(schar)inptrs[k][i]);
                const int ival = std::max(out_min, std::min(out_max, cvRound(val)));
                outptr[i] = isU8 ? (uchar)ival : (uchar)(schar)ival;
            }
            if (lutptr)
                lutInt8(outptr + r.start, outptr + r.start, len, isU8, lutptr);
        }, nstripes);

        if (uouts) {
            out.copyTo(uouts->at(0));
        }
    }

    bool setActivation(const Ptr<ActivationLayer>& layer) CV_OVERRIDE
    {
        Ptr<ActivationLayerInt8> activ_int8 = layer.dynamicCast<ActivationLayerInt8>();
        if (activ_int8.empty() || !activationLUT.empty())
            return false;
        // A layer built from ActivationInt8Params has its table in activationLUT, not in blobs.
        Mat lut = !activ_int8->activationLUT.empty() ? activ_int8->activationLUT
                : (!activ_int8->blobs.empty() ? activ_int8->blobs[0] : Mat());
        if (lut.total() != 256)
            return false;
        // Freeze the coefficients: the caller next overwrites output_sc/zp with the activation's.
        ensureCoeffs();
        activ = activ_int8;
        activationLUT = lut;
        return true;
    }
};

Ptr<Eltwise2Int8Layer> Eltwise2Int8Layer::create(const LayerParams& params)
{
    return Ptr<Eltwise2Int8Layer>(new Eltwise2Int8LayerImpl(params));
}

Ptr<Eltwise2Int8Layer> Eltwise2Int8Layer::create(const Eltwise2Int8Params& params)
{
    return Ptr<Eltwise2Int8Layer>(new Eltwise2Int8LayerImpl(params));
}

} // namespace dnn
} // namespace cv
