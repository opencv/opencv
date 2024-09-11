// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
//#include "../op_cuda.hpp"
//#include "../op_inf_engine.hpp"
//#include "../ie_ngraph.hpp"
//#include "../op_webnn.hpp"
//#include "../op_timvx.hpp"
//#include "../op_cann.hpp"

//#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

/*
    Gather layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Gather.html

    Opset's 1 to 13 are covered.
*/

// out must be pre-allocated
static void gather(const Mat& data, const Mat& ind, Mat& out, int axis)
{
    CV_Assert_N(data.isContinuous(), ind.isContinuous(), out.isContinuous());
    int indType = ind.type();
    CV_Assert(indType == CV_32S || indType == CV_64S);

    MatShape dataShape = data.shape();
    MatShape indShape = ind.shape();
    MatShape outShape = out.shape();
    int dataDims = dataShape.dims;
    int indDims = indShape.dims;
    int outDims = outShape.dims;

    CV_Assert(outDims == dataDims + indDims - 1);
    size_t indTotal = indShape.total(), nslices = 1;
    size_t elemSize = data.elemSize();
    size_t sliceSize = elemSize;

    for(int j = 0; j < dataDims; j++) {
        int szj = dataShape[j];
        if (j < axis)
            nslices *= szj;
        else if (j > axis)
            sliceSize *= szj;
    }
    size_t dataStep = sliceSize * dataShape[axis];
    size_t outStep = sliceSize * indTotal;
    volatile bool globOutOfRangeIdx = false;

    parallel_for_(Range(0, (int)indTotal), [&](const Range& r) {
        int shape_a = dataShape[axis];
        const uchar* dataptr0 = data.data;
        uchar* outptr0 = out.data;
        const int32_t* ind32 = indType == CV_32S ? ind.ptr<int32_t>() : nullptr;
        const int64_t* ind64 = indType == CV_64S ? ind.ptr<int64_t>() : nullptr;
        bool outOfRangeIdx = globOutOfRangeIdx;
        for (int j = r.start; j < r.end && !outOfRangeIdx; j++) {
            int k = ind32 ? (int)ind32[j] : (int)ind64[j];
            uchar* outptr = outptr0 + j*sliceSize;
            const uchar* dataptr = dataptr0;
            for (size_t i = 0; i < nslices; i++, dataptr += dataStep, outptr += outStep) {
                k += k < 0 ? shape_a : 0;
                if (k < 0 || k >= shape_a) {
                    outOfRangeIdx = true;
                    break;
                }
                memcpy(outptr, dataptr + k*sliceSize, sliceSize);
            }
        }
        if (outOfRangeIdx)
            globOutOfRangeIdx = true;
    }, std::min((double)indTotal, (double)sliceSize*nslices*indTotal/1e6));

    if (globOutOfRangeIdx) {
        CV_Error(Error::StsOutOfRange, "some of indices are outside of range");
    }
}

class Gather2LayerImpl CV_FINAL : public Gather2Layer
{
public:
    Gather2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 0);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    MatShape getOutShape(const MatShape& dataShape, const MatShape& indShape) const
    {
        int dataDims = dataShape.dims;
        int indDims = indShape.dims;

        int axis_ = normalize_axis(axis, dataDims);
        CV_Assert(0 <= axis_ && axis_ < dataDims);
        MatShape outShape(dataDims + indDims - 1);

        for (int i = 0; i < outShape.dims; i++) {
            if (i < indDims) {
                outShape[i] = indShape[i];
            } else {
                int j = i - indDims;
                outShape[i] = dataShape[j < axis ? j : j+1];
            }
        }
        return outShape;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 2);
        outputs.assign(1, getOutShape(inputs[0], inputs[1]));
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        size_t ninputs = inputs.size();
        CV_Assert(ninputs == 2);
        int dataType = inputs[0];
        int indType = inputs[1];
        CV_Assert(indType == CV_32S || indType == CV_64S);
        outputs.assign(requiredOutputs, dataType);
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size size = inputs_arr.size();
        int ninputs = size.area();

        CV_Assert(ninputs == 2);

        MatShape dataShape = inputs_arr.shape(0);
        MatShape indShape = inputs_arr.shape(1);
        int dataType = inputs_arr.type(0);
        int indType = inputs_arr.type(1);
        CV_Assert(indType == CV_32S || indType == CV_64S);

        MatShape outShape = getOutShape(dataShape, indShape);
        int outKind = outputs_arr.kind();
        int axis_ = normalize_axis(axis, dataShape.dims);

        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat data = inputs_arr.getMat(0);
            Mat ind = inputs_arr.getMat(1);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, dataType);
            runOp(data, ind, outs[0], axis_);
        } else {
            // [TODO] more efficient OpenCL implementation
            Mat data = inputs_arr.getMat(0);
            Mat ind = inputs_arr.getMat(1);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, dataType);
            Mat temp(outShape, dataType);
            runOp(data, ind, temp, axis_);
            temp.copyTo(outs[0]);
        }
    }

    void runOp(const Mat& data, const Mat& ind, Mat& out, int axis_)
    {
        gather(data, ind, out, axis_);
    }
};

Ptr<Gather2Layer> Gather2Layer::create(const LayerParams& params)
{
    return Ptr<Gather2Layer>(new Gather2LayerImpl(params));
}

}
}
