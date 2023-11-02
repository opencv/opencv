// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "./cpu_kernels/fast_norm.hpp"

// OpenVINO backend
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
// CUDA backend
#include "../op_cuda.hpp"
#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/mvn.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv { namespace dnn {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization
class InstanceNormLayerImpl CV_FINAL : public InstanceNormLayer {
public:
    InstanceNormLayerImpl(const LayerParams &params) {
        setParamsFrom(params);

        epsilon = params.get<float>("epsilon", 1e-5);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE {
        const auto &input = inputs[0];
        const auto &scale = inputs[1];
        const auto &bias = inputs[2];
        CV_CheckGE(input.size(), static_cast<size_t>(3), "DNN/InstanceNorm: input dimension >= 3 is required");

        int C = input[1];
        int scale_dim = std::accumulate(scale.begin(), scale.end(), 1, std::multiplies<int>());
        CV_CheckEQ(scale_dim, C, "DNN/InstanceNorm: scale must be a 1d tensor and match the channel of input");
        int bias_dim = std::accumulate(bias.begin(), bias.end(), 1, std::multiplies<int>());
        CV_CheckEQ(bias_dim, C, "DNN/InstanceNorm: bias must be a 1d tensor and match the channel of input");

        outputs.assign(1, inputs[0]);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
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
        const auto &bias = inputs[2];

        fastNormChannel(input, scale, bias, outputs[0], epsilon);
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE {
        // onnx to openvino convertion: https://github.com/openvinotoolkit/openvino/blob/2023.1.0/src/frontends/onnx/frontend/src/op/instance_norm.cpp

        auto ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        const auto &input_shape = ieInpNode.get_shape();
        std::shared_ptr<ngraph::Node> mvn, result;

        // mvn
#if INF_ENGINE_VER_MAJOR_LE(INF_ENGINE_RELEASE_2021_2)
        // https://docs.openvino.ai/2021.4/api/ngraph_python_api/_autosummary/ngraph.opset3.mvn.html?highlight=mvn#ngraph.opset3.mvn
        bool across_channels = false;
        bool normalize_variance = true;
        mvn = std::make_shared<ngraph::op::MVN>(ieInpNode, across_channels, normalize_variance, epsilon);
#else
        // https://docs.openvino.ai/2023.1/openvino_docs_ops_normalization_MVN_6.html
        std::vector<int64_t> axes_v(input_shape.size() - 2);
        std::iota(axes_v.begin(), axes_v.end(), 2); // {2, 3, ...} for nd input tensor, n>=3
        auto axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{axes_v.size()}, axes_v.data());
        bool normalize_variance = true;
        mvn = std::make_shared<ngraph::op::v6::MVN>(ieInpNode, axes, normalize_variance, epsilon, ngraph::op::MVNEpsMode::INSIDE_SQRT);
#endif

        // instance norm = scale * mvn + bias
        auto scale = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
        std::vector<int64_t> shared_shape_v(input_shape.size(), 1);
        shared_shape_v[1] = -1;
        auto shared_shape = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{shared_shape_v.size()}, shared_shape_v.data());
        scale  = std::make_shared<ngraph::op::v1::Reshape>(scale, shared_shape, true);
        result = std::make_shared<ngraph::op::v1::Multiply>(mvn, scale);
        auto bias = nodes[2].dynamicCast<InfEngineNgraphNode>()->node;
        bias  = std::make_shared<ngraph::op::v1::Reshape>(bias, shared_shape, true);
        result = std::make_shared<ngraph::op::v1::Add>(result, bias);

        return Ptr<BackendNode>(new InfEngineNgraphNode(result));
    }
#endif // HAVE_DNN_NGRAPH

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto input_shape = input_wrapper->getShape();
        size_t num_workers = static_cast<size_t>(total(input_shape, 2));

        return make_cuda_node<cuda4dnn::InstanceNormOp>(preferableTarget, std::move(context->stream), epsilon, num_workers);
    }
#endif

};

Ptr<InstanceNormLayer> InstanceNormLayer::create(const LayerParams &params) {
    return Ptr<InstanceNormLayer>(new InstanceNormLayerImpl(params));
}

}} // cv::dnn
