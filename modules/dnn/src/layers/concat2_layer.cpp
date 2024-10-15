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
    Concat layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Concat.html

    Opset's 1 to 13 are covered.
*/

// out must be pre-allocated
static void concat(const std::vector<Mat>& inps, Mat& out, int axis)
{
    CV_Assert(out.isContinuous());

    MatShape outShape = out.shape();
    int ndims = outShape.dims, nslices = 1;
    size_t esz = out.elemSize();
    size_t sliceSize = esz;
    size_t totalSize = 0;
    size_t outStep = 0;
    int ninputs = (int)inps.size();
    for (int i = ndims-1; i > axis; i--)
        sliceSize *= outShape[i];
    outStep = sliceSize*outShape[axis];
    for (int i = 0; i < axis; i++)
        nslices *= outShape[i];
    for (int i = 0; i < ninputs; i++) {
        CV_Assert(inps[i].isContinuous());
        totalSize += inps[i].total()*esz;
    }

    parallel_for_(Range(0, ninputs), [&](const Range& r) {
        for (int k = r.start; k < r.end; k++) {
            const Mat& inp_k = inps[k];
            uchar* outptr = out.data;
            const uchar* inptr_k = inp_k.data;
            int sz_a;
            for (int i = 0; i < k; i++) {
                sz_a = inps[i].size[axis];
                outptr += sliceSize*sz_a;
            }
            sz_a = inp_k.size[axis];
            size_t sliceSize_k = sliceSize*sz_a;
            for (int i = 0; i < nslices; i++)
                memcpy(outptr + i*outStep, inptr_k + i*sliceSize_k, sliceSize_k);
        }
    }, (totalSize > 1000000 ? ninputs : 1));
}

class Concat2LayerImpl CV_FINAL : public Concat2Layer
{
public:
    Concat2LayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        axis = params.get<int>("axis", 1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    MatShape getOutShape(const std::vector<MatShape>& inpShapes) const
    {
        size_t ninputs = inpShapes.size();
        CV_Assert(ninputs == inputs.size());

        const MatShape& inpShape0 = inpShapes[0];
        int inpDims = inpShape0.dims;
        int axis_ = normalize_axis(axis, inpDims);
        CV_Assert(0 <= axis_ && axis_ < inpDims);
        MatShape outShape = inpShape0;
        outShape[axis_] = 0;

        for (size_t i = 0; i < ninputs; i++) {
            const MatShape& inpShape_i = inpShapes[i];
            CV_Assert(inpShape_i.dims == inpDims);
            for (int j = 0; j < inpDims; j++) {
                if (j == axis_) {
                    outShape[j] += inpShape_i[j];
                    continue;
                }
                CV_Assert(inpShape0[j] == inpShape_i[j]);
            }
        }

        return outShape;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        outputs.assign(1, getOutShape(inputs));
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
        CV_Assert(ninputs > 0);
        for (size_t i = 1; i < ninputs; i++) {
            CV_Assert(inputs[i] == inputs[0]);
        }
        outputs.assign(requiredOutputs, inputs[0]);
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

        CV_Assert(ninputs > 0);

        std::vector<MatShape> inpShapes(ninputs);
        int inpType = inputs_arr.type(0);

        for (int i = 0; i < ninputs; i++) {
            inpShapes[i] = inputs_arr.shape(i);
            CV_Assert(inputs_arr.type(i) == inpType);
        }

        MatShape outShape = getOutShape(inpShapes);
        int outKind = outputs_arr.kind();
        int axis_ = normalize_axis(axis, inpShapes[0].dims);

        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat> inps;
            inputs_arr.getMatVector(inps);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            runOp(inps, outs[0], axis_);
        } else {
            // [TODO] more efficient OpenCL implementation
            std::vector<Mat> inps;
            inputs_arr.getMatVector(inps);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            Mat temp(outShape, inpType);
            runOp(inps, temp, axis_);
            temp.copyTo(outs[0]);
        }
    }

    void runOp(const std::vector<Mat>& inps, Mat& out, int axis_)
    {
        concat(inps, out, axis_);
    }
};

Ptr<Concat2Layer> Concat2Layer::create(const LayerParams& params)
{
    return Ptr<Concat2Layer>(new Concat2LayerImpl(params));
}

}
}
