// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "cpu_kernels/fast_norm.hpp"

// CANN backend
#include "../op_cann.hpp"

namespace cv { namespace dnn {

class LayerNormLayerImpl CV_FINAL : public LayerNormLayer
{
public:
    LayerNormLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        // standard attr
        axis = params.get<int>("axis", -1);
        epsilon = params.get<float>("epsilon", 1e-5);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               (backendId == DNN_BACKEND_CANN && axis != -1); // axis=-1 not supported due to 1d mat shape problem
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        // check shapes of weight and bias if existed
        // inputs >= 2 (X and Weight are requested, bias is optional)
        CV_Check(inputs.size(), inputs.size() >= 2 && inputs.size() <= 3, "LayerNorm: require two (x, weight) or three (x, weight, bias) inputs");

        auto x_shape = inputs[0];
        int x_ndims = static_cast<int>(x_shape.size());

        auto w_shape = inputs[1];
        // if axis == last_dim, scale and b are both 1d tensor (represented as 2d mat nx1)
        int w_ndims = static_cast<int>(w_shape.size());
        w_ndims = (axis == x_ndims - 1 && w_ndims == 2) ? w_ndims - 1 : w_ndims;
        CV_CheckEQ(x_ndims - axis, w_ndims, "LayerNorm: shape of weight does not match with given axis and shape of input");
        for (int i = 0; i < w_ndims; ++i)
            CV_CheckEQ(x_shape[axis+i], w_shape[i], "LayerNorm: weight dimensions does not match with input dimensions");
        if (inputs.size() == static_cast<int>(3))
        {
            auto b_shape = inputs[2];
            CV_CheckEQ(w_shape.size(), b_shape.size(), "LayerNorm: shape of weight does not match with shape of bias");
            for (size_t i = 0; i < w_shape.size(); ++i)
                CV_CheckEQ(w_shape[i], b_shape[i], "LayerNorm: bias dimensions does not match with weight dimensions");
        }

        outputs.assign(1, inputs[0]);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        const auto input_shape = shape(inputs[0]);
        axis = normalize_axis(axis, static_cast<int>(input_shape.size()));
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16S)
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

        if (inputs.size() == 3) {
            const auto &bias = inputs[2];
            fastNorm(input, scale, bias, output, epsilon, static_cast<size_t>(axis));
        } else {
            fastNorm(input, scale, output, epsilon, static_cast<size_t>(axis));
        }
    }

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(3), "LayerNorm/CANN: requires three input wrappers");
        CV_CheckEQ(nodes.size(), static_cast<size_t>(3), "LayerNorm/CANN: requires three input nodes");

        auto input_tensor_wrapper = inputs[0].dynamicCast<CannBackendWrapper>();
        auto input_tensor_desc = input_tensor_wrapper->getTensorDesc();

        CV_CheckNE(axis, static_cast<int>(input_tensor_desc->GetShape().GetDimNum() - 1), "LayerNorm: CANN does not support axis set as last axis due to 1D mat compatibility issue");

        auto scale_tensor_wrapper = inputs[1].dynamicCast<CannBackendWrapper>();
        auto scale_tensor_desc = scale_tensor_wrapper->getTensorDesc();

        auto bias_tensor_wrapper = inputs[2].dynamicCast<CannBackendWrapper>();
        auto bias_tensor_desc = bias_tensor_wrapper->getTensorDesc();

        auto last_node = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto scale_node = nodes[1].dynamicCast<CannBackendNode>()->getOp();
        auto bias_node = nodes[2].dynamicCast<CannBackendNode>()->getOp();

        auto op = std::make_shared<ge::op::LayerNorm>(name);

        // set attrs
        op->set_attr_begin_norm_axis(axis);
        op->set_attr_begin_params_axis(axis);
        op->set_attr_epsilon(epsilon);

        // set inputs
        // set inputs : x
        op->set_input_x_by_name(*last_node, input_tensor_wrapper->name.c_str());
        op->update_input_desc_x(*input_tensor_desc);
        // set inputs : gamma
        op->set_input_gamma_by_name(*scale_node, scale_tensor_wrapper->name.c_str());
        op->update_input_desc_gamma(*scale_tensor_desc);
        // set inputs : beta
        op->set_input_beta_by_name(*bias_node, bias_tensor_wrapper->name.c_str());
        op->update_input_desc_beta(*bias_tensor_desc);

        // set outputs
        auto output_desc_y = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc_y);
        auto output_desc_mean = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_mean(*output_desc_mean);
        auto output_desc_var = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_variance(*output_desc_var);

        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

};

Ptr<LayerNormLayer> LayerNormLayer::create(const LayerParams& params)
{
    return makePtr<LayerNormLayerImpl>(params);
}

}} // cv::dnn
