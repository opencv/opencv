// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

class CompressLayerImpl CV_FINAL : public CompressLayer
{
public:
    CompressLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        hasAxis_ = params.has("axis");
        axis_ = hasAxis_ ? params.get<int>("axis") : 0;
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool dynamicOutputShapes() const CV_OVERRIDE
    {
        return true;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        outputs.resize(requiredOutputs);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int /*requiredInternals*/,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        outputs.assign(requiredOutputs, inputs[0]);
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        CV_Assert(inputs.size() == 2);

        Mat data = inputs[0];
        Mat condition = inputs[1];

        if (!data.isContinuous())
            data = data.clone();
        if (!condition.isContinuous())
            condition = condition.clone();

        MatShape inShape = shape(data);
        int dimAxis;
        size_t outer = 1, inner = 1;

        if (!hasAxis_)
        {
            dimAxis = (int)data.total();
        }
        else
        {
            CV_Assert(data.dims >= 1);
            CV_Check(axis_, axis_ >= -data.dims && axis_ < data.dims, "Compress: axis is out of range");
            int ax = normalize_axis(axis_, data.dims);
            dimAxis = inShape[ax];
            for (int i = 0; i < ax; ++i)
                outer *= inShape[i];
            for (int i = ax + 1; i < data.dims; ++i)
                inner *= inShape[i];
        }

        size_t condTotal = condition.total();
        size_t numEval = std::min((size_t)dimAxis, condTotal);

        std::vector<int> selectedIndices;
        selectedIndices.reserve(numEval);
        for (size_t i = 0; i < numEval; ++i)
        {
            if (isConditionTrue(condition, i))
                selectedIndices.push_back((int)i);
        }

        int K = (int)selectedIndices.size();

        MatShape outShape;
        if (!hasAxis_)
        {
            outShape = shape(K);
        }
        else
        {
            outShape = inShape;
            int ax = normalize_axis(axis_, data.dims);
            outShape[ax] = K;
        }

        auto kind = outputs_arr.kind();
        std::vector<Mat> tmpOuts;
        std::vector<Mat>* pOuts = nullptr;

        if (kind == _InputArray::STD_VECTOR_MAT)
        {
            pOuts = &outputs_arr.getMatVecRef();
            CV_Assert(!pOuts->empty());
            (*pOuts)[0].fit(outShape, data.type());
        }
        else if (kind == _InputArray::STD_VECTOR_UMAT)
        {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            CV_Assert(!uouts.empty());
            tmpOuts.resize(uouts.size());
            tmpOuts[0].fit(outShape, data.type());
            pOuts = &tmpOuts;
        }
        else
        {
            CV_Error(Error::StsBadArg, cv::format("Unsupported output array kind: %d", kind));
        }

        Mat& out = (*pOuts)[0];

        if (K > 0)
        {
            size_t elemSize = data.elemSize();
            size_t sliceSize = inner * elemSize;
            size_t srcStep = dimAxis * sliceSize;
            size_t dstStep = (size_t)K * sliceSize;

            for (size_t ob = 0; ob < outer; ++ob)
            {
                const uchar* srcSliceBase = data.data + ob * srcStep;
                uchar* dstSliceBase = out.data + ob * dstStep;

                for (int k = 0; k < K; ++k)
                {
                    int srcIndex = selectedIndices[k];
                    memcpy(dstSliceBase + (size_t)k * sliceSize,
                           srcSliceBase + (size_t)srcIndex * sliceSize,
                           sliceSize);
                }
            }
        }

        if (kind == _InputArray::STD_VECTOR_UMAT)
        {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            uouts[0].fit(outShape, data.type());
            tmpOuts[0].copyTo(uouts[0]);
        }
    }

private:
    static inline bool isConditionTrue(const Mat& cond, size_t i)
    {
        switch (cond.depth())
        {
        case CV_Bool:
        case CV_8U:  return cond.ptr<const uint8_t>()[i] != 0;
        case CV_8S:  return cond.ptr<const int8_t>()[i] != 0;
        case CV_16U: return cond.ptr<const uint16_t>()[i] != 0;
        case CV_16S: return cond.ptr<const int16_t>()[i] != 0;
        case CV_32S: return cond.ptr<const int32_t>()[i] != 0;
        case CV_64S: return cond.ptr<const int64_t>()[i] != 0;
        case CV_32F: return cond.ptr<const float>()[i] != 0.0f;
        case CV_64F: return cond.ptr<const double>()[i] != 0.0;
        default:     return cond.ptr<const uint8_t>()[i] != 0;
        }
    }

    bool hasAxis_;
    int axis_;
};

Ptr<CompressLayer> CompressLayer::create(const LayerParams& params)
{
    return makePtr<CompressLayerImpl>(params);
}

}} // namespace cv::dnn
