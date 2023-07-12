// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

class GatherElementsImpl CV_FINAL : public GatherElementsLayer
{
public:
    GatherElementsImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        m_axis = params.get<int>("axis", 0);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), 2ull, "");
        CV_CheckEQ(inputs[0].size(), inputs[1].size(), "ONNX/GatherElements: input and indices have to be of same rank.");

        MatShape inpShape = inputs[0];
        const int axis = normalize_axis(m_axis, inpShape);

        MatShape indicesShape = inputs[1];
        outputs.assign(1, indicesShape[0]); // shape of output is same as indices
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& data = inputs[0];
        const Mat& indices = inputs[1];
        Mat& out = outputs[0];
        const int axis = normalize_axis(m_axis, shape(data));

        //Todo: finish implementation after finding algorithm 
        


    }

private:
    int m_axis;
};

Ptr<GatherElementsLayer> GatherElements::create(const LayerParams& params)
{
    return makePtr<GatherLayerImpl>(params);
}

}} // namespace cv::dnn