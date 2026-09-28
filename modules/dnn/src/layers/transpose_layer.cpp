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
    Transpose layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__Transpose.html

    Opset's 1 to 23 are covered.
*/

class TransposeLayerImpl CV_FINAL : public TransposeLayer
{
public:
    TransposeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        perm = params.getVector<int>("perm");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    MatShape getOutShape(const MatShape& inpShape) const
    {
        MatShape outShape(inpShape.dims);
        CV_Assert(perm.empty() || perm.size() == (size_t)inpShape.dims);

        for (int i = 0; i < inpShape.dims; i++) {
            int j = perm.empty() ? inpShape.dims - i - 1 : perm[i];
            CV_Assert(0 <= j && j < inpShape.dims);
            outShape[i] = inpShape[j];
        }

        return outShape;
    }

    bool isDataShuffling() const CV_OVERRIDE { return true; }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        outputs.assign(1, getOutShape(inputs[0]));
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
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
        CV_Assert(ninputs == 1);

        MatShape inpShape = inputs_arr.shape(0);
        MatShape outShape = getOutShape(inpShape);
        int inpType = inputs_arr.type(0);
        int outKind = outputs_arr.kind();

        CV_Assert(outKind == _InputArray::STD_VECTOR_MAT ||
                  outKind == _InputArray::STD_VECTOR_UMAT);

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            Mat inp = inputs_arr.getMat(0);
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            runOp(inp, outs[0]);
        } else {
            // [TODO] more efficient OpenCL implementation
            Mat inp = inputs_arr.getMat(0);
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, inpType);
            Mat temp(outShape, inpType);
            runOp(inp, temp);
            temp.copyTo(outs[0]);
        }
    }

    void runOp(const Mat& inp, Mat& out)
    {
        int ndims = inp.dims;
        std::vector<int> order(ndims);
        for (int i = 0; i < ndims; i++) {
            int j = perm.empty() ? ndims - i - 1 : perm[i];
            order[i] = j < 0 ? j + ndims : j;
        }
        transposeND(inp, order, out);
    }
};

Ptr<TransposeLayer> TransposeLayer::create(const LayerParams& params)
{
    return Ptr<TransposeLayer>(new TransposeLayerImpl(params));
}

}
}
