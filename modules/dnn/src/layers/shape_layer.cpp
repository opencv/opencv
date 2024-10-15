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

class ShapeLayerImpl CV_FINAL : public ShapeLayer
{
public:
    typedef int64_t shape_type_t;
    int shapeType;

    ShapeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        start = params.get<int>("start", 0);
        end = params.get<int>("end", INT_MAX);
        shapeType = DataType<shape_type_t>::type;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    Range getShapeRange(const MatShape& inpShape) const
    {
        int outDims = inpShape.dims;
        int start_ = start < 0 ? start + outDims : start;
        int end_ = end >= outDims ? outDims : end < 0 ? end + outDims : end;

        CV_Assert(0 <= start_);
        CV_Assert(start_ <= end_);
        CV_Assert(end_ <= outDims);

        return Range(start_, end_);
    }

    MatShape getOutShape(const MatShape& inpShape) const
    {
        MatShape outShape;
        outShape.dims = 1;

        Range r = getShapeRange(inpShape);

        outShape[0] = r.end - r.start;
        return outShape;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
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
        outputs.assign(requiredOutputs, shapeType);
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
        Range r = getShapeRange(inpShape);

        shape_type_t shapeData[CV_MAX_DIM];
        for (int i = r.start; i < r.end; i++)
            shapeData[i] = (shape_type_t)inpShape[i];

        Mat shape({r.end - r.start}, shapeType, shapeData);

        int outKind = outputs_arr.kind();

        if (outKind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& out = outputs_arr.getMatVecRef();
            CV_Assert(out.size() == 1);
            shape.copyTo(out[0]);
        } else if (outKind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& out = outputs_arr.getUMatVecRef();
            CV_Assert(out.size() == 1);
            shape.copyTo(out[0]);
        } else {
            CV_Error_(Error::StsBadArg, ("invalid/unsupported outputs_arr kind: %d", outKind));
        }
    }
};

Ptr<ShapeLayer> ShapeLayer::create(const LayerParams& params)
{
    return Ptr<ShapeLayer>(new ShapeLayerImpl(params));
}

}
}
