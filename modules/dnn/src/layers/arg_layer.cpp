// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"


namespace cv { namespace dnn {

class ArgLayerImpl CV_FINAL : public ArgLayer
{
public:
    enum class ArgOp
    {
        MIN = 0,
        MAX = 1,
    };

    ArgLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        axis = params.get<int>("axis", 0);
        keepdims = (params.get<int>("keepdims", 1) == 1);
        select_last_index = (params.get<int>("select_last_index", 0) == 1);

        const std::string& argOp = params.get<std::string>("op");

        if (argOp == "max")
        {
            op = ArgOp::MAX;
        }
        else if (argOp == "min")
        {
            op = ArgOp::MIN;
        }
        else
        {
            CV_Error(Error::StsBadArg, "Unsupported operation");
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    void handleKeepDims(MatShape& shape, const int axis_) const
    {
        if (keepdims)
        {
            shape[axis_] = 1;
        }
        else
        {
            shape.erase(shape.begin() + axis_);
        }
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        MatShape inpShape = inputs[0];
        // no axis for scalar
        if (inpShape.empty()){
            CV_Assert(axis == 0);
        }

        const int axis_ = normalize_axis(axis, inpShape);
        // handle dims = 0 situation
        if (!inpShape.empty())
            handleKeepDims(inpShape, axis_);
        outputs.assign(1, inpShape);

        return false;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
        const int requiredOutputs,
        const int requiredInternals,
        std::vector<MatType>& outputs,
        std::vector<MatType>& internals) const CV_OVERRIDE
    {
        outputs.assign(1, CV_64S);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert_N(inputs.size() == 1, outputs.size() == 1);
        MatShape outShape = shape(outputs[0]);
        Mat output(outShape, CV_32SC1);

        switch (op)
        {
        case ArgOp::MIN:
            cv::reduceArgMin(inputs[0], output, axis, select_last_index);
            break;
        case ArgOp::MAX:
            cv::reduceArgMax(inputs[0], output, axis, select_last_index);
            break;
        default:
            CV_Error(Error::StsBadArg, "Unsupported operation.");
        }

        output = output.reshape(1, outShape);
        output.convertTo(outputs[0], CV_64SC1);
    }

private:
    // The axis in which to compute the arg indices. Accepted range is [-r, r-1] where r = rank(data).
    int axis;
    // Keep the reduced dimension or not
    bool keepdims;
    // Whether to select the first or the last index or Max/Min.
    bool select_last_index;
    // Operation to be performed
    ArgOp op;
};

Ptr<ArgLayer> ArgLayer::create(const LayerParams& params)
{
    return Ptr<ArgLayer>(new ArgLayerImpl(params));
}

}}  // namespace cv::dnn
