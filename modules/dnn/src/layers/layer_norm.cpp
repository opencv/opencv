// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "cpu_kernels/fast_norm.hpp"

// CANN backend
#include "../op_cann.hpp"

// OpenVINO backend
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

// CUDA backend
#include "../op_cuda.hpp"
#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/layer_norm.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

// OpenCL backend
#ifdef HAVE_OPENCL
#include "../ocl4dnn/include/math_functions.hpp"
#include "opencl_kernels_dnn.hpp"
#endif

namespace cv { namespace dnn {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#LayerNormalization
class LayerNormLayerImpl CV_FINAL : public LayerNormLayer
{
#ifdef HAVE_OPENCL
    UMat weight_umat, bias_umat;
#endif

public:
    int axis0;

    LayerNormLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        // standard attr
        axis = axis0 = params.get<int>("axis", -1);
        epsilon = params.get<float>("epsilon", 1e-5);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
            return true;
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA   ||
               (backendId == DNN_BACKEND_CANN && axis != -1); // axis=-1 not supported due to 1d mat shape problem
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        // check shapes of weight and bias if existed
        // inputs >= 2 (X and Weight are required, bias is optional)
        int num_inputs = inputs.size() + blobs.size();
        CV_Check(num_inputs, num_inputs >= 2 && num_inputs <= 3, "LayerNorm: require two (x, weight) or three (x, weight, bias) inputs");

        auto x_shape = inputs[0];
        int x_ndims = static_cast<int>(x_shape.size());

        int axis_ = normalize_axis(axis0, x_shape.dims);

        // Weight and bias are either constants or variable
        auto w_shape = blobs.empty() ? inputs[1] : shape(blobs.front());
        // if axis == last_dim, scale and b are both 1d tensor (represented as 2d mat nx1)
        int w_ndims = static_cast<int>(w_shape.size());
        w_ndims = (axis_ == x_ndims - 1 && w_ndims == 2) ? w_ndims - 1 : w_ndims;
        CV_CheckEQ(x_ndims - axis_, w_ndims, "LayerNorm: shape of weight does not match with given axis and shape of input");
        for (int i = 0; i < w_ndims; ++i)
            CV_CheckEQ(x_shape[axis_+i], w_shape[i], "LayerNorm: weight dimensions does not match with input dimensions");
        if (num_inputs >= 3)
        {
            auto b_shape = blobs.empty() ? inputs[2] : shape(blobs.back());
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
        axis = normalize_axis(axis0, static_cast<int>(input_shape.size()));

#ifdef HAVE_OPENCL
        weight_umat.release();
        bias_umat.release();
#endif
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
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
        const auto &scale = blobs.empty() ? inputs[1] : blobs.front();
        auto &output = outputs[0];

        axis = normalize_axis(axis0, input.dims);

        if ((inputs.size() + blobs.size()) >= 3) {
            const auto &bias = blobs.empty() ? inputs[2] : blobs.back();
            fastNorm(input, scale, bias, output, epsilon, static_cast<size_t>(axis));
        } else {
            fastNorm(input, scale, output, epsilon, static_cast<size_t>(axis));
        }
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_) {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        const auto &input = inputs[0];

        // no fp16 support
        if (input.depth() == CV_16F) {
            return false;
        }

        auto &output = outputs[0];

        const auto input_shape = shape(input);
        axis = normalize_axis(axis0, input_shape.dims);
        size_t loops = static_cast<size_t>(total(input_shape, 0, axis)),
               norm_size = static_cast<size_t>(total(input_shape, axis));
        float inv_norm_size = 1.f / norm_size;

        if (weight_umat.empty()) {
            if (blobs.empty()) {
                weight_umat = inputs[1];
            } else {
                blobs.front().copyTo(weight_umat);
            }
        }
        if (bias_umat.empty()) {
            if ((inputs.size() + blobs.size()) == 3) {
                if (blobs.empty()) {
                    bias_umat = inputs[2];
                } else {
                    blobs.back().copyTo(bias_umat);
                }
            } else {
                bias_umat = UMat::zeros(norm_size, 1, CV_32F);
            }
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
        // Calculate instance norm: output = weight * (x - mean) / sqrt(var + eps) + bias
        String mvn_kernel_name = format("mvn%d", num_vector);
        build_opt += " -DNORM_VARIANCE -DLAYER_NORM -DKERNEL_MVN";
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
        mvn_kernel.set(6, ocl::KernelArg::PtrReadOnly(weight_umat));
        mvn_kernel.set(7, ocl::KernelArg::PtrReadOnly(bias_umat));
        mvn_kernel.set(8, (int)1);
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
        CV_CheckEQ(inputs.size(), static_cast<size_t>(3), "LayerNorm/CANN: requires three input wrappers");
        CV_CheckEQ(nodes.size(), static_cast<size_t>(3), "LayerNorm/CANN: requires three input nodes");

        auto input_tensor_wrapper = inputs[0].dynamicCast<CannBackendWrapper>();
        auto input_tensor_desc = input_tensor_wrapper->getTensorDesc();

        CV_CheckNE(axis, static_cast<int>(input_tensor_desc->GetShape().GetDimNum() - 1), "LayerNorm: CANN does not support axis set as last axis due to 1D mat compatibility issue");

        auto last_node = nodes[0].dynamicCast<CannBackendNode>()->getOp();

        auto op = std::make_shared<ge::op::LayerNorm>(name);

        // set attrs
        op->set_attr_begin_norm_axis(axis);
        op->set_attr_begin_params_axis(axis);
        op->set_attr_epsilon(epsilon);

        // set inputs
        // set inputs : x
        op->set_input_x_by_name(*last_node, input_tensor_wrapper->name.c_str());
        op->update_input_desc_x(*input_tensor_desc);
        // set inputs : gamma & beta
        if (blobs.empty()) {
            auto scale_tensor_wrapper = inputs[1].dynamicCast<CannBackendWrapper>();
            auto scale_tensor_desc = scale_tensor_wrapper->getTensorDesc();
            auto scale_node = nodes[1].dynamicCast<CannBackendNode>()->getOp();
            op->set_input_gamma_by_name(*scale_node, scale_tensor_wrapper->name.c_str());
            op->update_input_desc_gamma(*scale_tensor_desc);

            if (inputs.size() == 3) {
                auto bias_tensor_wrapper = inputs[2].dynamicCast<CannBackendWrapper>();
                auto bias_tensor_desc = bias_tensor_wrapper->getTensorDesc();
                auto bias_node = nodes[2].dynamicCast<CannBackendNode>()->getOp();
                op->set_input_beta_by_name(*bias_node, bias_tensor_wrapper->name.c_str());
                op->update_input_desc_beta(*bias_tensor_desc);
            }
        } else {
            const auto &scale_mat = blobs.front();
            const auto op_const_scale = std::make_shared<CannConstOp>(scale_mat.data, scale_mat.type(), shape(scale_mat), cv::format("%s_w", name.c_str()));
            op->set_input_gamma(*(op_const_scale->getOp()));
            op->update_input_desc_gamma(*(op_const_scale->getTensorDesc()));

            if ((inputs.size() + blobs.size()) >= 3) {
                const auto &bias_mat = blobs.back();
                const auto op_const_bias = std::make_shared<CannConstOp>(bias_mat.data, bias_mat.type(), shape(bias_mat), cv::format("%s_b", name.c_str()));
                op->set_input_beta(*(op_const_bias->getOp()));
                op->update_input_desc_beta(*(op_const_bias->getTensorDesc()));
            }
        }

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
        auto ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        const auto &input_shape = ieInpNode.get_shape();
        std::shared_ptr<ov::Node> mvn, result;
        ov::Output<ov::Node> scale, bias;

        // mvn
        // https://docs.openvino.ai/2023.1/openvino_docs_ops_normalization_MVN_6.html
        std::vector<int64_t> axes_v(input_shape.size() - axis);
        std::iota(axes_v.begin(), axes_v.end(), axis);
        auto axes = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{axes_v.size()}, axes_v.data());
        bool normalize_variance = true;
        mvn = std::make_shared<ov::op::v6::MVN>(ieInpNode, axes, normalize_variance, epsilon, ov::op::MVNEpsMode::INSIDE_SQRT);

        // layer norm = scale * mvn + bias
        if (blobs.empty()) {
            scale = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
            if (nodes.size() == 3) {
                bias = nodes[2].dynamicCast<InfEngineNgraphNode>()->node;
            }
        } else {
            auto scale_mat = blobs.front();
            const auto scale_shape = shape(scale_mat);
            scale = std::make_shared<ov::op::v0::Constant>(ov::element::f32, std::vector<size_t>(scale_shape.begin(), scale_shape.end()), scale_mat.data);
            if ((nodes.size() + blobs.size()) == 3) {
                auto bias_mat = blobs.back();
                const auto bias_shape = shape(bias_mat);
                bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, std::vector<size_t>(bias_shape.begin(), bias_shape.end()), bias_mat.data);
            }
        }
        if (axis == -1 || axis == input_shape.size() - 1) { // special case for 1D tensor (2D mat)
            std::vector<int64_t> shared_shape_v(input_shape.size(), 1);
            shared_shape_v.back() = -1;
            auto shared_shape = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{shared_shape_v.size()}, shared_shape_v.data());
            scale = std::make_shared<ov::op::v1::Reshape>(scale, shared_shape, true);
            if ((nodes.size() + blobs.size()) == 3) {
                bias = std::make_shared<ov::op::v1::Reshape>(bias, shared_shape, true);
            }
        }

        result = std::make_shared<ov::op::v1::Multiply>(mvn, scale);
        if ((nodes.size() + blobs.size()) == 3) {
            result = std::make_shared<ov::op::v1::Add>(result, bias);
        }

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
        size_t loops = static_cast<size_t>(total(input_shape, 0, axis));

        const auto scale = blobs.empty() ? Mat() : blobs.front(),
                   bias = blobs.empty() ? Mat() : blobs.back();

        return make_cuda_node<cuda4dnn::LayerNormOp>(preferableTarget, std::move(context->stream), scale, bias, axis, epsilon, loops);
    }
#endif // HAVE_CUDA
};

Ptr<LayerNormLayer> LayerNormLayer::create(const LayerParams& params)
{
    return makePtr<LayerNormLayerImpl>(params);
}

}} // cv::dnn
