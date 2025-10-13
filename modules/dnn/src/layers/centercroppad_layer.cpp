// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

namespace cv {
namespace dnn {

// ONNX CenterCropPad operator
// Spec: https://onnx.ai/onnx/operators/onnx__CenterCropPad.html
// Supported opsets: 18

class CenterCropPadLayerImpl CV_FINAL : public CenterCropPadLayer
{
public:
    std::vector<int> axes_attr;

    CenterCropPadLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        if (params.has("axes"))
        {
            DictValue dv = params.get("axes");
            axes_attr.resize(dv.size());
            for (int i = 0; i < dv.size(); ++i) axes_attr[i] = dv.get<int>(i);
        }
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        Net::Impl* netimpl_ = getNetImpl(this);
        CV_Assert(netimpl_);
        const size_t ninputs = this->inputs.size();
        CV_Assert(ninputs >= 2);
        bool shapeDynamic = !netimpl_->isConstArg(this->inputs[1]);
        bool axesDynamic = (ninputs >= 3) && !netimpl_->isConstArg(this->inputs[2]);
        return shapeDynamic || axesDynamic;
    }

    void buildTargetShape(const MatShape& inputShape, const Mat& shapeTensor, const Mat& axesTensor,
                          std::vector<int>& targetShape, std::vector<int>& usedAxes) const
    {
        const int rank = inputShape.dims;
        targetShape.assign(inputShape.begin(), inputShape.end());
        usedAxes.clear();

        CV_Assert(shapeTensor.dims == 1);
        CV_Assert(shapeTensor.type() == CV_32S || shapeTensor.type() == CV_64S);

        std::vector<int> axes;
        if (!axesTensor.empty())
        {
            CV_Assert(axesTensor.dims == 1);
            int naxes = (int)axesTensor.total();
            axes.resize(naxes);
            if (axesTensor.type() == CV_32S)
            {
                const int32_t* a32 = (const int32_t*)axesTensor.data;
                for (int i = 0; i < naxes; ++i) axes[i] = normalize_axis((int)a32[i], rank);
            }
            else
            {
                const int64_t* a64 = (const int64_t*)axesTensor.data;
                for (int i = 0; i < naxes; ++i) axes[i] = normalize_axis((int)a64[i], rank);
            }
        }
        else if (!axes_attr.empty())
        {
            axes.resize((int)axes_attr.size());
            for (size_t i = 0; i < axes_attr.size(); ++i) axes[i] = normalize_axis(axes_attr[i], rank);
        }
        else
        {
            axes.resize(rank);
            for (int i = 0; i < rank; ++i) axes[i] = i;
        }

        CV_Assert((int)shapeTensor.total() == (int)axes.size());

        if (shapeTensor.type() == CV_32S)
        {
            const int32_t* shp = (const int32_t*)shapeTensor.data;
            applyShapeFromPtr(shp, (int)axes.size(), axes, targetShape, usedAxes);
        }
        else if (shapeTensor.type() == CV_64S)
        {
            const int64_t* shp = (const int64_t*)shapeTensor.data;
            applyShapeFromPtr(shp, (int)axes.size(), axes, targetShape, usedAxes);
        }
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int /*requiredOutputs*/,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() >= 2);

        const MatShape& inShape = inputs[0];
        Net::Impl* netimpl_ = getNetImpl(this);
        CV_Assert(netimpl_);

        Mat shapeTensor = netimpl_->argTensor(this->inputs[1]);
        Mat axesTensor;
        if (this->inputs.size() >= 3)
            axesTensor = netimpl_->argTensor(this->inputs[2]);

        std::vector<int> targetShape, usedAxes;
        buildTargetShape(inShape, shapeTensor, axesTensor, targetShape, usedAxes);

        outputs.assign(1, MatShape(targetShape.begin(), targetShape.end()));
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
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size size = inputs_arr.size();
        int ninputs = size.area();
        CV_Assert(ninputs >= 2);

        Mat X = inputs_arr.getMat(0);
        Mat shapeTensor = inputs_arr.getMat(1);
        Mat axesTensor;
        if (ninputs >= 3)
            axesTensor = inputs_arr.getMat(2);

        const int rank = X.dims;
        std::vector<int> outShape(rank);
        std::vector<int> usedAxes;

        MatShape inShape = X.shape();
        std::vector<int> tgt;
        buildTargetShape(inShape, shapeTensor, axesTensor, tgt, usedAxes);
        outShape.assign(tgt.begin(), tgt.end());

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT)
        {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(MatShape(outShape.begin(), outShape.end()), X.type());
            centerCropPad(X, outs[0], usedAxes);
        }
        else if (kind == _InputArray::STD_VECTOR_UMAT)
        {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(MatShape(outShape.begin(), outShape.end()), X.type());
            Mat temp(MatShape(outShape.begin(), outShape.end()), X.type());
            centerCropPad(X, temp, usedAxes);
            temp.copyTo(outs[0]);
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "");
        }
    }

private:
    template<typename T>
    static void applyShapeFromPtr(const T* shp, int count, const std::vector<int>& axes,
                                  std::vector<int>& targetShape, std::vector<int>& usedAxes)
    {
        for (int i = 0; i < count; ++i)
        {
            int a = axes[i];
            int desired = (int)shp[i];
            CV_Assert(desired >= 0);
            targetShape[a] = desired;
            usedAxes.push_back(a);
        }
    }

    void centerCropPad(const Mat& src, Mat& dst, const std::vector<int>& axes) const
    {
        CV_Assert(src.dims == dst.dims);
        const int rank = src.dims;

        std::vector<Range> srcRanges(rank), dstRanges(rank);
        for (int i = 0; i < rank; ++i)
        {
            int s = src.size[i];
            int d = dst.size[i];
            if (std::find(axes.begin(), axes.end(), i) == axes.end())
            {
                CV_Assert(s == d);
                srcRanges[i] = Range(0, s);
                dstRanges[i] = Range(0, d);
                continue;
            }

            if (d <= s)
            {
                int diff = s - d;
                int start = diff / 2;
                int end = start + d;
                srcRanges[i] = Range(start, end);
                dstRanges[i] = Range(0, d);
            }
            else
            {
                int diff = d - s;
                int start = diff / 2;
                int end = start + s;
                srcRanges[i] = Range(0, s);
                dstRanges[i] = Range(start, end);
            }
        }

        dst.setTo(Scalar(0));
        src(srcRanges.data()).copyTo(dst(dstRanges.data()));
    }
};

Ptr<CenterCropPadLayer> CenterCropPadLayer::create(const LayerParams& params)
{
    return Ptr<CenterCropPadLayer>(new CenterCropPadLayerImpl(params));
}

} // namespace dnn
} // namespace cv
