// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

// OpenCL backend
#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

// OpenVINO backend
#ifdef HAVE_DNN_NGRAPH
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"
#endif

// CUDA backend
#ifdef HAVE_CUDA
#include "../op_cuda.hpp"
#include "../cuda4dnn/primitives/depth_space_ops.hpp"
#endif

// CANN backend
#ifdef HAVE_CANN
#include "../op_cann.hpp"
#endif

// TIM-VX backend
#ifdef HAVE_TIMVX
#include "../op_timvx.hpp"
#endif

namespace cv { namespace dnn {

struct DepthSpaceOps {
    MatShape internal_shape;
    MatShape transposed_internal_shape;
    std::vector<int> permutation;

#ifdef HAVE_OPENCL
    UMat umat_permutation;
    UMat umat_internal_strides;
    UMat umat_transposed_internal_strides;
#endif

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) {
        transposed_internal_shape = MatShape(internal_shape.size());
        for (size_t i = 0; i < permutation.size(); i++) {
            transposed_internal_shape[i] = internal_shape[permutation[i]];
        }

#ifdef HAVE_OPENCL
        umat_permutation.release();
        umat_internal_strides.release();
        umat_transposed_internal_strides.release();
#endif
    }

    void cpuCompute(const Mat &input, Mat &output) {
        const auto output_shape = shape(output);
        Mat tmp;
        cv::transposeND(input.reshape(1, internal_shape), permutation, tmp);
        tmp.reshape(1, output_shape).copyTo(output);
    }

#ifdef HAVE_OPENCL
    bool oclCompute(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) {
        std::vector<UMat> inputs, outputs;

        inputs_arr.getUMatVector(inputs);
        outputs_arr.getUMatVector(outputs);

        if (umat_permutation.empty() || umat_internal_strides.empty() || umat_transposed_internal_strides.empty()) {
            Mat mat_permutation(1, permutation.size(), CV_32S, permutation.data());
            mat_permutation.copyTo(umat_permutation);

            std::vector<int> internal_strides(permutation.size(), 1), transposed_internal_stides(permutation.size(), 1);
            for (int i = static_cast<int>(permutation.size()) - 2; i >= 0; i--) {
                internal_strides[i] = internal_strides[i + 1] * internal_shape[i + 1];
                transposed_internal_stides[i] = transposed_internal_stides[i + 1] * transposed_internal_shape[i + 1];
            }
            Mat mat_internal_strides(1, internal_strides.size(), CV_32S, internal_strides.data());
            mat_internal_strides.copyTo(umat_internal_strides);
            Mat mat_transposed_internal_strides(1, transposed_internal_stides.size(), CV_32S, transposed_internal_stides.data());
            mat_transposed_internal_strides.copyTo(umat_transposed_internal_strides);
        }

        const auto output_shape = shape(outputs.front());
        UMat tmp = inputs.front().reshape(1, static_cast<int>(internal_shape.size()), internal_shape.data());

        bool use_half = (inputs_arr.depth() == CV_16F);
        std::string permute_options = cv::format("-DDtype=%s", use_half ? "half" : "float");
        ocl::Kernel permute_kernel("permute", ocl::dnn::permute_oclsrc, permute_options);
        if (permute_kernel.empty()) {
            return false;
        }
        UMat transposed_tmp(static_cast<int>(transposed_internal_shape.size()), transposed_internal_shape.data(), inputs_arr.depth());
        size_t num_element = static_cast<size_t>(std::accumulate(internal_shape.begin(), internal_shape.end(), 1, std::multiplies<int>()));
        permute_kernel.set(0, static_cast<int>(num_element));
        permute_kernel.set(1, ocl::KernelArg::PtrReadOnly(tmp));
        permute_kernel.set(2, ocl::KernelArg::PtrReadOnly(umat_permutation));
        permute_kernel.set(3, ocl::KernelArg::PtrReadOnly(umat_internal_strides));
        permute_kernel.set(4, ocl::KernelArg::PtrReadOnly(umat_transposed_internal_strides));
        permute_kernel.set(5, static_cast<int>(permutation.size()));
        permute_kernel.set(6, ocl::KernelArg::PtrWriteOnly(transposed_tmp));
        if (!permute_kernel.run(1, &num_element, NULL, false)) {
            return false;
        }

        transposed_tmp.reshape(1, static_cast<int>(output_shape.size()), output_shape.data()).copyTo(outputs.front());
        return true;
    }
#endif // HAVE_OPENCL
};

class DepthToSpaceLayerImpl CV_FINAL : public DepthToSpaceLayer, public DepthSpaceOps {
public:
    DepthToSpaceLayerImpl(const LayerParams &params) {
        setParamsFrom(params);

        CV_CheckTrue(params.has("blocksize"), "DepthSpaceLayer: blocksize is required");
        blocksize = params.get<int>("blocksize");

        auto mode = params.get<std::string>("mode", "DCR");
        if (mode == "CRD") {
            is_crd = true;
            permutation = {0, 1, 4, 2, 5, 3};
        } else if (mode == "DCR") {
            is_crd = false;
            permutation = {0, 3, 4, 1, 5, 2};
        } else {
            CV_Error(Error::StsBadArg, cv::format("DepthToSpace: unsupported mode %s\n", mode.c_str()));
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
               backendId == DNN_BACKEND_CUDA   ||
               backendId == DNN_BACKEND_CANN   ||
               (backendId == DNN_BACKEND_TIMVX && is_crd);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(1), "DepthSpaceLayer: accepts only one input");
        const auto &input = inputs.front();
        CV_CheckEQ(input.size(), static_cast<size_t>(4), "DepthSpaceLayer: input needs to be 4-dimensional [N, C, H, W]");
        int batch = input[0], input_depth = input[1], input_height = input[2], input_width = input[3];
        int output_depth = -1, output_height = -1, output_width = -1;

        CV_CheckEQ(input_depth % (blocksize * blocksize), 0,
                   "DepthSpaceLayer: requires input depth to be a multiple of (blocksize * blocksize)");
        output_depth = input_depth / blocksize / blocksize;
        output_height = input_height * blocksize;
        output_width = input_width * blocksize;

        outputs.assign(1, MatShape{batch, output_depth, output_height, output_width});
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        auto input_shape = shape(inputs.front());
        int batch = input_shape[0], input_depth = input_shape[1], input_height = input_shape[2], input_width = input_shape[3];
        if (is_crd) {
            internal_shape = MatShape{batch, input_depth / (blocksize * blocksize), blocksize, blocksize, input_height, input_width};
        } else {
            internal_shape = MatShape{batch, blocksize, blocksize, input_depth / (blocksize * blocksize), input_height, input_width};
        }

        DepthSpaceOps::finalize(inputs_arr, outputs_arr);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        // TODO: support 8-bit int in permute kernel
        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) && inputs_arr.depth() != CV_8S,
                   DepthSpaceOps::oclCompute(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16F) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        DepthSpaceOps::cpuCompute(inputs.front(), outputs.front());
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper>> &inputs,
                                        const std::vector<Ptr<BackendNode>> &nodes) CV_OVERRIDE {
        using namespace ov::op;
        auto input_node = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::shared_ptr<ov::Node> output_node;
        if (is_crd) {
            output_node = std::make_shared<v0::DepthToSpace>(input_node, v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, static_cast<size_t>(blocksize));
        } else {
            output_node = std::make_shared<v0::DepthToSpace>(input_node, v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, static_cast<size_t>(blocksize));
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(output_node));
    }
#endif // HAVE_DNN_NGRAPH

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void *context_,
                              const std::vector<Ptr<BackendWrapper>>& inputs,
                              const std::vector<Ptr<BackendWrapper>>& outputs) override {
        using namespace cv::dnn::cuda4dnn;
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        std::vector<size_t> perm(permutation.begin(), permutation.end());
        return make_cuda_node<cuda4dnn::DepthSpaceOps>(preferableTarget, std::move(context->stream), internal_shape, perm);
    }
#endif // HAVE_CUDA

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper>> &inputs,
                                      const std::vector<Ptr<BackendWrapper>> &outputs,
                                      const std::vector<Ptr<BackendNode>> &nodes) CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(1), "DepthToSpace/CANN: only accepts one input wrapper");
        CV_CheckEQ(nodes.size(), static_cast<size_t>(1), "DepthToSpace/CANN: only accepts one input node");

        auto input_tensor_wrapper = inputs.front().dynamicCast<CannBackendWrapper>();
        auto input_tensor_desc = input_tensor_wrapper->getTensorDesc();
        auto input_node = nodes.front().dynamicCast<CannBackendNode>()->getOp();

        auto node = std::make_shared<ge::op::DepthToSpace>(name);

        node->set_attr_block_size(blocksize);
        if (is_crd) {
            node->set_attr_mode("CRD");
        } else {
            node->set_attr_mode("DCR");
        }
        node->set_attr_data_format("NCHW");

        node->set_input_x_by_name(*input_node, input_tensor_wrapper->name.c_str());
        node->update_input_desc_x(*input_tensor_desc);

        auto output_tensor_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        node->update_output_desc_y(*output_tensor_desc);

        return Ptr<BackendNode>(new CannBackendNode(node));
    }
#endif

#ifdef HAVE_TIMVX
    virtual Ptr<BackendNode> initTimVX(void* timvx_info_,
                                       const std::vector<Ptr<BackendWrapper>> &inputs,
                                       const std::vector<Ptr<BackendWrapper>> &outputs,
                                       bool isLast) CV_OVERRIDE {
        auto info = reinterpret_cast<TimVXInfo*>(timvx_info_);
        CV_Assert(info);
        auto timvx_graph = info->getGraph();
        CV_Assert(timvx_graph);
        auto graph = timvx_graph->graph;

        auto input_wrapper = inputs.front().dynamicCast<TimVXBackendWrapper>();
        int input_wrapper_index = -1;
        if (input_wrapper->isTensor()) {
            input_wrapper_index = timvx_graph->getTensorIndex(input_wrapper->getTensor());
            if (input_wrapper_index == -1) {
                auto tmp = input_wrapper->getMat();
                input_wrapper = std::make_shared<TimVXBackendWrapper>(tmp);
            }
        }
        if (!input_wrapper->isTensor() || input_wrapper_index == 1) {
            auto input_node_quant = Ptr<tim::vx::Quantization>(new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, 1.0f, 0));
            input_wrapper->createTensor(graph, tim::vx::TensorAttribute::INPUT, input_node_quant);
            input_wrapper_index = timvx_graph->addWrapper(input_wrapper);
        }

        auto output_wrapper = outputs.front().dynamicCast<TimVXBackendWrapper>();
        auto output_node_quant = input_wrapper->getTensorQuantization();
        if (isLast) {
            auto shape_type = getShapeTypeFromMat(output_wrapper->getMat());
            output_wrapper->setTensorShape(shape_type);
            output_wrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT, output_node_quant);
        } else {
            output_wrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT, output_node_quant);
        }
        int output_wrapper_index = timvx_graph->addWrapper(output_wrapper);

        std::shared_ptr<tim::vx::Operation> timvx_node;
        timvx_node = graph->CreateOperation<tim::vx::ops::DepthToSpace>(blocksize);
        std::vector<int> input_wrapper_indices{input_wrapper_index}, output_wrapper_indices{output_wrapper_index};
        return Ptr<BackendNode>(new TimVXBackendNode(timvx_graph, timvx_node, input_wrapper_indices, output_wrapper_indices));
    }
#endif

private:
    int blocksize;

    bool is_crd;
};

Ptr<DepthToSpaceLayer> DepthToSpaceLayer::create(const LayerParams &params) {
    return makePtr<DepthToSpaceLayerImpl>(params);
}

class SpaceToDepthLayerImpl CV_FINAL : public SpaceToDepthLayer, public DepthSpaceOps {
public:
    SpaceToDepthLayerImpl(const LayerParams &params) {
        setParamsFrom(params);

        CV_CheckTrue(params.has("blocksize"), "SpaceToDepthLayer: blocksize is required");
        blocksize = params.get<int>("blocksize");

        permutation = {0, 3, 5, 1, 2, 4};
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
               backendId == DNN_BACKEND_CUDA   ||
               backendId == DNN_BACKEND_CANN   ||
               (backendId == DNN_BACKEND_TIMVX);
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(1), "SpaceToDepthLayer: accepts only one input");
        const auto &input = inputs.front();
        CV_CheckEQ(input.size(), static_cast<size_t>(4), "SpaceToDepthLayer: input needs to be 4-dimensional [N, C, H, W]");
        int batch = input[0], input_depth = input[1], input_height = input[2], input_width = input[3];
        int output_depth = -1, output_height = -1, output_width = -1;

        CV_CheckEQ(input_height % blocksize, 0, "SpaceToDepthLayer: requires input height to be a multiple of blocksize");
        CV_CheckEQ(input_width % blocksize, 0, "SpaceToDepthLayer: requires input width to be a multiple of blocksize");
        output_depth = input_depth * blocksize * blocksize;
        output_height = input_height / blocksize;
        output_width = input_width / blocksize;

        outputs.assign(1, MatShape{batch, output_depth, output_height, output_width});
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        auto input_shape = shape(inputs.front());
        int batch = input_shape[0], input_depth = input_shape[1], input_height = input_shape[2], input_width = input_shape[3];
        internal_shape = MatShape{batch, input_depth, input_height / blocksize, blocksize, input_width / blocksize, blocksize};

        DepthSpaceOps::finalize(inputs_arr, outputs_arr);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        // TODO: support 8-bit int in permute kernel
        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) && inputs_arr.depth() != CV_8S,
                   DepthSpaceOps::oclCompute(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16F) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        DepthSpaceOps::cpuCompute(inputs.front(), outputs.front());
    }

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper>> &inputs,
                                        const std::vector<Ptr<BackendNode>> &nodes) CV_OVERRIDE {
        using namespace ov::op;
        auto input_node = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::shared_ptr<ov::Node> output_node;
        output_node = std::make_shared<v0::SpaceToDepth>(input_node, v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, static_cast<size_t>(blocksize));
        return Ptr<BackendNode>(new InfEngineNgraphNode(output_node));
    }
#endif // HAVE_DNN_NGRAPH

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void *context_,
                              const std::vector<Ptr<BackendWrapper>> &inputs,
                              const std::vector<Ptr<BackendWrapper>> &outputs) override {
        using namespace cv::dnn::cuda4dnn;
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        std::vector<size_t> perm(permutation.begin(), permutation.end());
        return make_cuda_node<cuda4dnn::DepthSpaceOps>(preferableTarget, std::move(context->stream), internal_shape, perm);
    }
#endif // HAVE_CUDA

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper>> &inputs,
                                      const std::vector<Ptr<BackendWrapper>> &outputs,
                                      const std::vector<Ptr<BackendNode>> &nodes) CV_OVERRIDE {
        CV_CheckEQ(inputs.size(), static_cast<size_t>(1), "DepthToSpace/CANN: only accepts one input wrapper");
        CV_CheckEQ(nodes.size(), static_cast<size_t>(1), "DepthToSpace/CANN: only accepts one input node");

        auto input_tensor_wrapper = inputs.front().dynamicCast<CannBackendWrapper>();
        auto input_tensor_desc = input_tensor_wrapper->getTensorDesc();
        auto input_node = nodes.front().dynamicCast<CannBackendNode>()->getOp();

        auto node = std::make_shared<ge::op::SpaceToDepth>(name);

        node->set_attr_block_size(blocksize);
        node->set_attr_data_format("NCHW");

        node->set_input_x_by_name(*input_node, input_tensor_wrapper->name.c_str());
        node->update_input_desc_x(*input_tensor_desc);

        auto output_tensor_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        node->update_output_desc_y(*output_tensor_desc);

        return Ptr<BackendNode>(new CannBackendNode(node));
    }
#endif

#ifdef HAVE_TIMVX
    virtual Ptr<BackendNode> initTimVX(void* timvx_info_,
                                       const std::vector<Ptr<BackendWrapper>> &inputs,
                                       const std::vector<Ptr<BackendWrapper>> &outputs,
                                       bool isLast) CV_OVERRIDE {
        auto info = reinterpret_cast<TimVXInfo*>(timvx_info_);
        CV_Assert(info);
        auto timvx_graph = info->getGraph();
        CV_Assert(timvx_graph);
        auto graph = timvx_graph->graph;

        auto input_wrapper = inputs.front().dynamicCast<TimVXBackendWrapper>();
        int input_wrapper_index = -1;
        if (input_wrapper->isTensor()) {
            input_wrapper_index = timvx_graph->getTensorIndex(input_wrapper->getTensor());
            if (input_wrapper_index == -1) {
                auto tmp = input_wrapper->getMat();
                input_wrapper = std::make_shared<TimVXBackendWrapper>(tmp);
            }
        }
        if (!input_wrapper->isTensor() || input_wrapper_index == 1) {
            auto input_node_quant = Ptr<tim::vx::Quantization>(new tim::vx::Quantization(tim::vx::QuantType::ASYMMETRIC, 1.0f, 0));
            input_wrapper->createTensor(graph, tim::vx::TensorAttribute::INPUT, input_node_quant);
            input_wrapper_index = timvx_graph->addWrapper(input_wrapper);
        }

        auto output_wrapper = outputs.front().dynamicCast<TimVXBackendWrapper>();
        auto output_node_quant = input_wrapper->getTensorQuantization();
        if (isLast) {
            auto shape_type = getShapeTypeFromMat(output_wrapper->getMat());
            output_wrapper->setTensorShape(shape_type);
            output_wrapper->createTensor(graph, tim::vx::TensorAttribute::OUTPUT, output_node_quant);
        } else {
            output_wrapper->createTensor(graph, tim::vx::TensorAttribute::TRANSIENT, output_node_quant);
        }
        int output_wrapper_index = timvx_graph->addWrapper(output_wrapper);

        std::shared_ptr<tim::vx::Operation> timvx_node;
        timvx_node = graph->CreateOperation<tim::vx::ops::SpaceToDepth>(std::vector<int>{blocksize, blocksize});
        std::vector<int> input_wrapper_indices{input_wrapper_index}, output_wrapper_indices{output_wrapper_index};
        return Ptr<BackendNode>(new TimVXBackendNode(timvx_graph, timvx_node, input_wrapper_indices, output_wrapper_indices));
    }
#endif

private:
    int blocksize;
};

Ptr<SpaceToDepthLayer> SpaceToDepthLayer::create(const LayerParams &params) {
    return makePtr<SpaceToDepthLayerImpl>(params);
}

}} // namespace cv::dnn
