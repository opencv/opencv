// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "./cpu_kernels/fast_norm.hpp"

// CUDA backend
#include "../op_cuda.hpp"
#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/group_norm.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv {
namespace dnn {

// Group Normalization Layer
class GroupNormLayerImpl CV_FINAL : public GroupNormLayer {
public:
    GroupNormLayerImpl(const LayerParams &params) {
        setParamsFrom(params);

        epsilon = params.get<float>("epsilon", 1e-5);
        num_groups = params.get<int>("num_groups");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
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
        CV_CheckGE(input.size(), static_cast<size_t>(3), "DNN/GroupNorm: input dimension >= 3 is required");

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

        if (inputs_arr.depth() == CV_16S) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const auto& input = inputs[0];
        const auto& scale = inputs[1];
        const auto& bias = inputs[2];

        fastNormGroup(input, scale, bias, outputs[0], epsilon, num_groups);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void *context_,
                          const std::vector<Ptr<BackendWrapper>>& inputs,
                          const std::vector<Ptr<BackendWrapper>>& outputs) override {
    auto context = reinterpret_cast<csl::CSLContext*>(context_);

    auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
    auto input_shape = input_wrapper->getShape();
    size_t N = input_shape[0];
    size_t num_groups = this->num_groups;
    size_t loops = N * num_groups;

    return make_cuda_node<cuda4dnn::GroupNormOp>(preferableTarget, std::move(context->stream), epsilon, loops, num_groups);
}
#endif // HAVE_CUDA

private:
    float epsilon;
    size_t num_groups;
};

Ptr<GroupNormLayer> GroupNormLayer::create(const LayerParams &params) {
    return Ptr<GroupNormLayer>(new GroupNormLayerImpl(params));
}

}} // cv::dnn