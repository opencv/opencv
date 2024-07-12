// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "./cpu_kernels/fast_norm.hpp"

// CANN backend
#include "../op_cann.hpp"

// OpenVINO backend
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

// CUDA backend
#include "../op_cuda.hpp"
#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/instance_norm.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

// OpenCL backend
#ifdef HAVE_OPENCL
#include "../ocl4dnn/include/math_functions.hpp"
#include "opencl_kernels_dnn.hpp"
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
            //    backendId == DNN_BACKEND_CANN; // not supported due to 1d mat shape issue
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

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

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
        const auto &bias = inputs[2];

        fastNormChannel(input, scale, bias, outputs[0], epsilon);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_) {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        const auto &input = inputs[0], &scale = inputs[1], &bias = inputs[2];
        auto &output = outputs[0];

        const auto input_shape = shape(input);
        size_t N = input_shape[0], C = input_shape[1],
               loops = N * C, norm_size = static_cast<size_t>(total(input_shape, 2));
        float inv_norm_size = 1.f / norm_size;

        // no fp16 support
        if (input.depth() == CV_16F) {
            return false;
        }

        String base_opts = format(" -DT=float -DT4=float4 -Dconvert_T=convert_float4");

        // Calculate mean
        UMat one = UMat::ones(norm_size, 1, CV_32F);
        UMat mean = UMat(loops, 1, CV_32F);
        UMat mean_square = UMat(loops, 1, CV_32F);
        UMat tmp = UMat(loops, norm_size, CV_32F);
        bool ret = ocl4dnn::ocl4dnnGEMV<float>(ocl4dnn::CblasNoTrans, loops, norm_size, inv_norm_size,
                                               input, 0, one, 0, 0.f, mean, 0);
        if (!ret) {
            return false;
        }
        // Calculate mean_square
        int num_vector = (norm_size % 8 == 0) ? 8 : ((norm_size % 4 == 0) ? 4 : 1);
        size_t global[] = {loops, static_cast<size_t>(norm_size / num_vector)};
        String build_opt = format(" -DNUM=%d", num_vector) + base_opts;
        String mean_square_kernel_name = format("calc_mean%d", num_vector);
        ocl::Kernel mean_square_kernel(mean_square_kernel_name.c_str(), ocl::dnn::mvn_oclsrc, build_opt + " -DKERNEL_MEAN");
        if (mean_square_kernel.empty()) {
            return false;
        }
        mean_square_kernel.set(0, ocl::KernelArg::PtrReadOnly(input));
        mean_square_kernel.set(1, (int)loops);
        mean_square_kernel.set(2, (int)norm_size);
        mean_square_kernel.set(3, ocl::KernelArg::PtrReadOnly(mean));
        mean_square_kernel.set(4, ocl::KernelArg::PtrWriteOnly(tmp));
        ret = mean_square_kernel.run(2, global, NULL, false);
        if (!ret) {
            return false;
        }
        ret = ocl4dnn::ocl4dnnGEMV<float>(ocl4dnn::CblasNoTrans, loops, norm_size, inv_norm_size,
                                          tmp, 0, one, 0, 0.f, mean_square, 0);
        if (!ret) {
            return false;
        }
        // Calculate instance norm: output = scale * (x - mean) / sqrt(var + eps) + bias
        String mvn_kernel_name = format("mvn%d", num_vector);
        build_opt += " -DNORM_VARIANCE -DFUSE_BATCH_NORM -DKERNEL_MVN";
        ocl::Kernel mvn_kernel(mvn_kernel_name.c_str(), ocl::dnn::mvn_oclsrc, build_opt);
        if (mvn_kernel.empty()) {
            return false;
        }
        mvn_kernel.set(0, ocl::KernelArg::PtrReadOnly(input));
        mvn_kernel.set(1, (int)loops);
        mvn_kernel.set(2, (int)norm_size);
        mvn_kernel.set(3, (float)epsilon);
        mvn_kernel.set(4, ocl::KernelArg::PtrReadOnly(mean));
        mvn_kernel.set(5, ocl::KernelArg::PtrReadOnly(mean_square));
        mvn_kernel.set(6, ocl::KernelArg::PtrReadOnly(scale));
        mvn_kernel.set(7, ocl::KernelArg::PtrReadOnly(bias));
        mvn_kernel.set(8, (int)C);
        mvn_kernel.set(9, (float)0.f);
        mvn_kernel.set(10, ocl::KernelArg::PtrWriteOnly(output));
        ret = mvn_kernel.run(2, global, NULL, false);
        if (!ret) {
            return false;
        }

        return true;
    }
#endif

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE {
        auto input_tensor_wrapper = inputs[0].dynamicCast<CannBackendWrapper>();
        auto input_tensor_desc = input_tensor_wrapper->getTensorDesc();

        auto scale_tensor_wrapper = inputs[1].dynamicCast<CannBackendWrapper>();
        auto scale_tensor_desc = scale_tensor_wrapper->getTensorDesc();

        auto bias_tensor_wrapper = inputs[2].dynamicCast<CannBackendWrapper>();
        auto bias_tensor_desc = bias_tensor_wrapper->getTensorDesc();

        auto last_node = nodes[0].dynamicCast<CannBackendNode>()->getOp();
        auto scale_node = nodes[1].dynamicCast<CannBackendNode>()->getOp();
        auto bias_node = nodes[2].dynamicCast<CannBackendNode>()->getOp();

        auto op = std::make_shared<ge::op::InstanceNorm>(name);

        // set attrs
        op->set_attr_epsilon(epsilon);

        // set inputs
        // set inputs : x
        op->set_input_x_by_name(*last_node, input_tensor_wrapper->name.c_str());
        op->update_input_desc_x(*input_tensor_desc);
        // set inputs : gamma
        op->set_input_gamma_by_name((*scale_node), scale_tensor_wrapper->name.c_str());
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

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE {
        // onnx to openvino convertion: https://github.com/openvinotoolkit/openvino/blob/2023.1.0/src/frontends/onnx/frontend/src/op/instance_norm.cpp

        auto ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        const auto &input_shape = ieInpNode.get_shape();
        std::shared_ptr<ov::Node> mvn, result;

        // mvn
        // https://docs.openvino.ai/2023.1/openvino_docs_ops_normalization_MVN_6.html
        std::vector<int64_t> axes_v(input_shape.size() - 2);
        std::iota(axes_v.begin(), axes_v.end(), 2); // {2, 3, ...} for nd input tensor, n>=3
        auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes_v.size()}, axes_v.data());
        bool normalize_variance = true;
        mvn = std::make_shared<ov::op::v6::MVN>(ieInpNode, axes, normalize_variance, epsilon, ov::op::MVNEpsMode::INSIDE_SQRT);

        // instance norm = scale * mvn + bias
        auto scale = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
        std::vector<int64_t> shared_shape_v(input_shape.size(), 1);
        shared_shape_v[1] = -1;
        auto shared_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{shared_shape_v.size()}, shared_shape_v.data());
        scale  = std::make_shared<ov::op::v1::Reshape>(scale, shared_shape, true);
        result = std::make_shared<ov::op::v1::Multiply>(mvn, scale);
        auto bias = nodes[2].dynamicCast<InfEngineNgraphNode>()->node;
        bias  = std::make_shared<ov::op::v1::Reshape>(bias, shared_shape, true);
        result = std::make_shared<ov::op::v1::Add>(result, bias);

        return Ptr<BackendNode>(new InfEngineNgraphNode(result));
    }
#endif // HAVE_DNN_NGRAPH

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void *context_,
                              const std::vector<Ptr<BackendWrapper>>& inputs,
                              const std::vector<Ptr<BackendWrapper>>& outputs) override {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        auto input_wrapper = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto input_shape = input_wrapper->getShape();
        size_t loops = static_cast<size_t>(total(input_shape, 0, 2));

        return make_cuda_node<cuda4dnn::InstanceNormOp>(preferableTarget, std::move(context->stream), epsilon, loops);
    }
#endif // HAVE_CUDA

};

Ptr<InstanceNormLayer> InstanceNormLayer::create(const LayerParams &params) {
    return Ptr<InstanceNormLayer>(new InstanceNormLayerImpl(params));
}

}} // cv::dnn
