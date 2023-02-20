// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

class TopKLayerImpl CV_FINAL : public TopKLayer
{
public:
    TopKLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        axis = params.get<int>("axis", -1);
        largest = static_cast<bool>(params.get<int>("largest", 1));
        sorted = static_cast<bool>(params.get<int>("sorted", 1));

        K = params.get<int>("K", 1);
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
        auto out_shape = inputs[0];
        out_shape[axis] = K;

        // two outputs: values, indices
        outputs.assign(1, out_shape);
        outputs.assign(2, out_shape);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        ;
    }
};

Ptr<TopKLayer> TopKLayer::create(const LayerParams& params)
{
    return makePtr<TopKLayerImpl>(params);
}

}} // namespace cv::dnn
