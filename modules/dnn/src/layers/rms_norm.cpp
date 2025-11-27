// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "cpu_kernels/fast_norm.hpp"

namespace cv { namespace dnn {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LayerNormalization
class RMSNormLayerImpl CV_FINAL : public RMSNormLayer
{
#ifdef HAVE_OPENCL
    UMat weight_umat, bias_umat;
#endif

public:
    int axis0;

    RMSNormLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        // standard attr
        axis = params.get<int>("axis", -1);
        epsilon = params.get<float>("epsilon", 1e-5);
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
        const int n_inputs = inputs.size();
        CV_Check(n_inputs, n_inputs == 2, "RMSNorm: require two (x, scale) inputs");

        auto x_shape = inputs[0];
        auto scale_shape = inputs[1];
        const int normalized_axis = normalize_axis(axis, static_cast<int>(x_shape.size()));
        const int x_ndims = static_cast<int>(x_shape.size());

        for (int i = 0; i + normalized_axis < x_ndims; ++i)
        {
            CV_CheckTrue(
                x_shape[i + normalized_axis] == scale_shape[i] || scale_shape[i] == 1,
                "RMSNorm: scale should be broadcastable to input shape");
        }

        outputs.assign(1, inputs[0]);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const auto &input = inputs[0];
        const auto &scale = inputs[1];
        auto &output = outputs[0];

        axis = normalize_axis(axis, input.dims);
        fastNormBroadcast(input, scale, output, epsilon, static_cast<size_t>(axis));
    }
private:
    int axis;
    float epsilon;
};

Ptr<RMSNormLayer> RMSNormLayer::create(const LayerParams& params)
{
    return makePtr<RMSNormLayerImpl>(params);
}

}} // cv::dnn
