// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

// ONNX Compress operator
// Spec: https://onnx.ai/onnx/operators/onnx__Compress.html
// Supported opsets: 9-24

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
                         const int /*requiredOutputs*/,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        outputs.assign(1, shape(0));
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int /*requiredInternals*/,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& /*internals*/) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        CV_CheckTypeEQ(inputs[1], CV_Bool, "Compress: condition must be a bool tensor");
        outputs.assign(std::max(1, requiredOutputs), inputs[0]);
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

        const Mat& data = inputs[0];
        const Mat& condition = inputs[1];

        CV_Assert(data.isContinuous());
        CV_Assert(condition.isContinuous());
        CV_CheckTypeEQ(condition.type(), CV_Bool, "Compress: condition must be a bool tensor");

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
        const uchar* cond = condition.ptr<uchar>();

        std::vector<int> selectedIndices;
        selectedIndices.reserve(numEval);
        for (size_t i = 0; i < numEval; ++i)
        {
            if (cond[i])
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

        if (K == 0)
        {
            if (kind == _InputArray::STD_VECTOR_UMAT)
            {
                std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
                uouts[0].fit(outShape, data.type());
                tmpOuts[0].copyTo(uouts[0]);
            }
            return;
        }

        Mat& out = (*pOuts)[0];

        size_t totalSlices = outer * (size_t)K;
        size_t elemSize = data.elemSize();
        size_t sliceSize = inner * elemSize;

        parallel_for_(Range(0, (int)totalSlices), [&](const Range& r) {
            for (int i = r.start; i < r.end; ++i)
            {
                size_t ob = (size_t)i / (size_t)K;
                int k = i % K;
                int srcIndex = selectedIndices[k];
                const uchar* src = data.data + (ob * (size_t)dimAxis + (size_t)srcIndex) * sliceSize;
                uchar* dst = out.data + (size_t)i * sliceSize;
                memcpy(dst, src, sliceSize);
            }
        });

        if (kind == _InputArray::STD_VECTOR_UMAT)
        {
            std::vector<UMat>& uouts = outputs_arr.getUMatVecRef();
            uouts[0].fit(outShape, data.type());
            tmpOuts[0].copyTo(uouts[0]);
        }
    }

private:
    bool hasAxis_;
    int axis_;
};

Ptr<CompressLayer> CompressLayer::create(const LayerParams& params)
{
    return makePtr<CompressLayerImpl>(params);
}

}} // namespace cv::dnn
