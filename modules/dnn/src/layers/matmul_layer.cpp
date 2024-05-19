// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include "cpu_kernels/fast_gemm.hpp"

// OpenVINO backend
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

// Vulkan backend
#include "../op_vkcom.hpp"

// CUDA backend
#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/matmul_broadcast.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

// CANN backend
#include "../op_cann.hpp"

namespace cv { namespace dnn {

class MatMulLayerImpl CV_FINAL : public MatMulLayer {
#ifdef HAVE_OPENCL
    UMat weight_umat, bias_umat;
#endif

 public:
    MatMulLayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        trans_a = params.get<bool>("transA", false);
        trans_b = params.get<bool>("transB", false);
        alpha = params.get<float>("alpha", 1.f);
        beta = params.get<float>("beta", 1.f);

        real_ndims_C = params.get<int>("real_ndims_C", -1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
               (backendId == DNN_BACKEND_VKCOM && haveVulkan() && !trans_a && !trans_b) ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_CANN;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        int num_inputs = inputs.size() + blobs.size();
        CV_CheckGE(num_inputs, 2, "DNN/MatMul: two inputs at least");
        CV_CheckLE(num_inputs, 3, "DNN/MatMul: three inputs at most");

        const auto shape_A = inputs[0], shape_B = blobs.empty() ? inputs[1] : shape(blobs[0]);
        CV_CheckGE(shape_A.size(), static_cast<size_t>(2), "DNN/MatMul: invalid shape of input A");
        CV_CheckGE(shape_B.size(), static_cast<size_t>(2), "DNN/MatMul: invalid shape of input B");

        // Check legal matrix multiplication
        int mA = shape_A[shape_A.size() - 2], nA = shape_A.back();
        int mB = shape_B[shape_B.size() - 2], nB = shape_B.back();
        int M = trans_a ? nA : mA;
        int N = trans_b ? mB : nB;
        int K_A = trans_a ? mA : nA;
        int K_B = trans_b ? nB : mB;
        CV_CheckEQ(K_A, K_B, "DNN/MatMul: invalid dimension K");

        // Check if inputs are broadcastable.
        MatShape common_shape;
        if (shape_A.size() != 2 || shape_B.size() != 2) {
            const auto &shape_more_dims = shape_A.size() > shape_B.size() ? shape_A : shape_B;
            const auto &shape_less_dims = shape_A.size() > shape_B.size() ? shape_B : shape_A;
            size_t diff_dims = shape_more_dims.size() - shape_less_dims.size();
            common_shape = shape_more_dims;
            for (size_t i = 0; i < shape_less_dims.size() - 2; i++) {
                const auto dl = shape_less_dims[i], dm = shape_more_dims[i + diff_dims];
                if (dl != 1 && dm != 1 && dl != dm) {
                    CV_Error(Error::StsBadSize, format("DNN/MatMul: invalid shape for broadcasting, shape_A[%zu]=%d, shape_B[%zu]=%d\n", i, shape_less_dims[i], i, shape_more_dims[i + diff_dims]));
                }

                if (dm == 1) {
                    common_shape[i + diff_dims] = dl;
                }
            }
            common_shape[common_shape.size() - 2] = M;
            common_shape[common_shape.size() - 1] = N;
        } else {
            common_shape.resize(2);
            common_shape[0] = M;
            common_shape[1] = N;
        }

        // Check if bias is broadcastable
        if (num_inputs == 3) {
            const auto shape_C = blobs.empty() ? inputs.back() : shape(blobs.back());
            if (real_ndims_C == 1) { // (1) or (N)
                CV_Check(shape_C[0], shape_C[0] == 1 || shape_C[0] == N, "DNN/MatMul: invalid dimension of C");
            } else if (real_ndims_C >= 2) {
                const auto &shape_large = common_shape.size() > shape_C.size() ? common_shape : shape_C;
                const auto &shape_small = common_shape.size() > shape_C.size() ? shape_C : common_shape;
                size_t diff_dims = shape_large.size() - shape_small.size();
                for (size_t i = 0; i < shape_small.size(); i++) {
                    const auto dl = shape_small[i], dm = shape_large[i + diff_dims];
                    if (dl != 1 && dm != 1 && dl != dm) {
                        CV_Error(Error::StsBadSize, "DNN/MatMul: invalid shape of C");
                    }
                }
            }
        }

        outputs.assign(1, common_shape);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        opt.init();

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const auto A_shape = shape(inputs[0]),
                   B_shape = blobs.empty() ? shape(inputs[1]) : shape(blobs[0]),
                   C_shape = shape(outputs[0]);
        helper.compute(trans_a, trans_b, A_shape, B_shape, C_shape);

        if (!blobs.empty()) {
            fastGemmPackB(blobs[0], packed_input_B, trans_b, opt);
            helper.updatePackedBOffsets(packed_input_B.size());
        }

        // broadcast bias if needed
        if ((inputs.size() + blobs.size()) >= 3 && blobs.size() >= 2) {
            const auto bias_mat = blobs.back();
            const auto bias_shape = shape(bias_mat);
            bool is_broadcast_needed = real_ndims_C == 0 || real_ndims_C == 1 || (total(bias_shape) != total(C_shape) || bias_shape.size() != C_shape.size());

            if (is_broadcast_needed) {
                broadcast_bias = Mat(C_shape, CV_32F);
                auto *broadcast_bias_ptr = broadcast_bias.ptr<float>();

                const auto *bias = bias_mat.ptr<const float>();
                if (bias_mat.total() == 1) { // [], [1], [1, ...]
                    float b = (*bias) * beta;
                    for (size_t i = 0; i < broadcast_bias.total(); i++) {
                        broadcast_bias_ptr[i] = b;
                    }
                } else if (real_ndims_C == 1) { // [n]
                    size_t inner_size = C_shape.back(),
                        loops = total(C_shape) / inner_size;
                    for (size_t i = 0; i < loops; i++) {
                        size_t step = i * inner_size;
                        for (size_t j = 0; j < inner_size; j++) {
                            broadcast_bias_ptr[step + j] = beta * bias[j];
                        }
                    }
                } else {
                    broadcast(bias_mat, C_shape, broadcast_bias);
                }
            } else {
                broadcast_bias = blobs.back();
            }
        }

#ifdef HAVE_OPENCL
        weight_umat.release();
        bias_umat.release();
#endif
    }

    // works like Y = numpy.matmul(A, B)
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

        const auto &A = inputs[0];
        auto &Y = outputs[0];

        const auto *a = A.ptr<const float>();
        auto *y = Y.ptr<float>();
        // add bias if existed
        if ((inputs.size() + blobs.size()) >= 3) {
            const auto &shape_Y = shape(Y);
            if (blobs.empty()) { // bias from input
                const auto &bias_mat = inputs.back();
                const auto *bias = bias_mat.ptr<const float>();
                if (bias_mat.total() == 1) { // [], [1], [1, ...]
                    float b = (*bias) * beta;
                    for (size_t i = 0; i < Y.total(); i++) {
                        y[i] = b;
                    }
                } else if (real_ndims_C == 1) { // [n]
                    const size_t inner_size = shape_Y.back(),
                                 batches = total(Y) / inner_size;
                    parallel_for_(Range(0, batches), [&] (const Range &r) {
                        for (int i = r.start; i < r.end; i++) {
                            const size_t output_offset = i * inner_size;
                            for (size_t j = 0; j < inner_size; j++) {
                                y[output_offset + j] = beta * bias[j];
                            }
                        }
                    }, double(batches * inner_size * (1 / 1024.0)));
                } else {
                    broadcast(bias_mat, shape_Y, Y);
                }
            } else { // bias from constant
                const auto *bias = broadcast_bias.ptr<const float>();
                std::memcpy(y, bias, total(shape_Y) * sizeof(float));
            }
        } else {
            std::memset(y, 0, Y.total() * sizeof(float));
        }

        if (blobs.empty()) {
            const auto &B = inputs[1];
            const auto *b = B.ptr<const float>();
            fastGemmBatch(helper.batch, helper.A_offsets.data(), helper.B_offsets.data(), helper.C_offsets.data(),
                          helper.M, helper.N, helper.K, alpha, a, helper.lda0, helper.lda1,
                          b, helper.ldb0, helper.ldb1, beta, y, helper.ldc, opt);
        } else {
            fastGemmBatch(helper.batch, helper.A_offsets.data(), helper.packed_B_offsets.data(), helper.C_offsets.data(),
                          helper.M, helper.N, helper.K, alpha, a, helper.lda0, helper.lda1,
                          packed_input_B.data(), beta, y, helper.ldc, opt);
        }
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, InputArrayOfArrays internals) {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        bool use_half = (inputs_arr.depth() == CV_16F);
        inputs_arr.getUMatVector(inputs);
        outputs_arr.getUMatVector(outputs);

        // does not support bias as input
        if (inputs.size() >= 3) {
            return false;
        }

        const auto &input_A = inputs[0];
        auto &output = outputs[0];
        const auto output_shape = shape(output);

        if (blobs.empty()) {
            weight_umat = inputs[1];
            if ((inputs.size() + blobs.size() >= 3)) {
                bias_umat = UMat::zeros(output_shape.size(), output_shape.data(), CV_32F);
            }
        } else {
            if (weight_umat.empty()) {
                blobs.front().copyTo(weight_umat);
            }
            if ((inputs.size() + blobs.size() >= 3)) {
                if (bias_umat.empty()) {
                    broadcast_bias.copyTo(bias_umat);
                }
            } else {
                if (bias_umat.empty()) {
                    bias_umat = UMat::zeros(output_shape.size(), output_shape.data(), CV_32F);
                }
            }
        }

        auto &input_B = weight_umat;

        int M = static_cast<int>(helper.M),
            N = static_cast<int>(helper.N),
            K = static_cast<int>(helper.K),
            batch = static_cast<int>(helper.batch);
        int batch_A = total(shape(input_A)) / (M * K),
            batch_B = total(shape(input_B)) / (N * K);
        MatShape new_shape_A{batch_A, M * K}, new_shape_B{batch_B, N * K}, new_shape_output{batch, M * N};

        const auto input_A_2d = input_A.reshape(1, new_shape_A.size(), &new_shape_A[0]),
                   input_B_2d = input_B.reshape(1, new_shape_B.size(), &new_shape_B[0]);
        auto output_2d = output.reshape(1, new_shape_output.size(), &new_shape_output[0]);
        UMat A, B, C, A_fp32, B_fp32, C_fp32;
        for (int i = 0; i < batch; i++) {
            A = input_A_2d.row(helper.A_rows[i]).reshape(1, trans_a ? K : M);
            B = input_B_2d.row(helper.B_rows[i]).reshape(1, trans_b ? N : K);
            C = output_2d.row(helper.C_rows[i]).reshape(1, M);

            if (trans_a) {
                A = A.t();
            }
            if (trans_b) {
                B = B.t();
            }

            if (use_half) {
                A.convertTo(A_fp32, CV_32F);
                B.convertTo(B_fp32, CV_32F);
                C.convertTo(C_fp32, CV_32F);
            } else {
                A_fp32 = A;
                B_fp32 = B;
                C_fp32 = C;
            }
            cv::gemm(A_fp32, B_fp32, 1.f, noArray(), 0.f, C_fp32);
            if (use_half) {
                A_fp32.convertTo(A, CV_16F);
                B_fp32.convertTo(B, CV_16F);
                C_fp32.convertTo(C, CV_16F);
            }
        }

        // add bias
        if (!bias_umat.empty()) {
            cv::add(output, bias_umat, output);
        }

        return true;
    }
#endif // HAVE_OPENCL

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE {
        auto& input_A_node = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::shared_ptr<ov::Node> result;
        ov::Output<ov::Node> bias;

        if (blobs.empty()) {
            auto &input_B_node = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
            result = std::make_shared<ov::op::v0::MatMul>(input_A_node, input_B_node, trans_a, trans_b);
            if (nodes.size() >= 3) {
                bias = nodes[2].dynamicCast<InfEngineNgraphNode>()->node;
                result = std::make_shared<ov::op::v1::Add>(result, bias);
            }
        } else {
            auto input_B_shape = getShape<size_t>(blobs[0]);
            auto input_B_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, input_B_shape, blobs[0].data);
            result = std::make_shared<ov::op::v0::MatMul>(input_A_node, input_B_node, trans_a, trans_b);
            if ((nodes.size() + blobs.size()) >= 3) {
                const auto bias_shape = shape(broadcast_bias);
                bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, std::vector<size_t>(bias_shape.begin(), bias_shape.end()), broadcast_bias.data);
                result = std::make_shared<ov::op::v1::Add>(result, bias);
            }
        }

        return Ptr<BackendNode>(new InfEngineNgraphNode(result));
    }
#endif // HAVE_DNN_NGRAPH

#ifdef HAVE_VULKAN
    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs,
                                       std::vector<Ptr<BackendWrapper> > &outputs) CV_OVERRIDE {
        auto input_A_wrapper = inputs[0].dynamicCast<VkComBackendWrapper>();
        auto output_wrapper = outputs[0].dynamicCast<VkComBackendWrapper>();

        const auto input_A_shape = shape(*input_A_wrapper->getMat());
        const auto output_shape = shape(*output_wrapper->getMat());
        if ((inputs.size() + blobs.size()) >= 3 || output_shape.size() != 2) {
            return Ptr<BackendNode>();
        }

        std::vector<Mat> constants;

        if (!blobs.empty()) {
            constants.push_back(blobs[0]);
        }

        Ptr<vkcom::OpBase> op = new vkcom::OpMatMul(constants, input_A_shape[0], input_A_shape[1], output_shape[1]);
        return Ptr<BackendNode>(new VkComBackendNode(inputs, op, outputs));
    }
#endif

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void *context_,
                              const std::vector<Ptr<BackendWrapper>>& inputs,
                              const std::vector<Ptr<BackendWrapper>>& outputs) override {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        auto input_B = Mat(), bias = Mat();
        if (!blobs.empty()) {
            input_B = blobs.front();
            if (blobs.size() >= 2) {
                bias = broadcast_bias;
            }
        }

        CV_CheckFalse(helper.empty(), "DNN/MatMul/CUDA: MatMulHelper is not initialized");

        return make_cuda_node<cuda4dnn::MatMulBroadcastOp>(preferableTarget, std::move(context->stream), std::move(context->cublas_handle), input_B, bias, trans_a, trans_b, helper.A_offsets, helper.B_offsets, helper.C_offsets, helper.batch);
    }
#endif // HAVE_CUDA

#ifdef HAVE_CANN
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE {
        auto input_A_wrapper = inputs[0].dynamicCast<CannBackendWrapper>();
        auto input_A_desc = input_A_wrapper->getTensorDesc();
        auto input_A_node = nodes[0].dynamicCast<CannBackendNode>()->getOp();

        auto op = std::make_shared<ge::op::BatchMatMulV2>(name);

        // set attributes
        op->set_attr_adj_x1(trans_a);
        op->set_attr_adj_x2(trans_b);

        // set inputs
        // set inputs : x1
        op->set_input_x1_by_name(*input_A_node, input_A_wrapper->name.c_str());
        op->update_input_desc_x1(*input_A_desc);
        // set inputs : x2
        if (blobs.empty()) { // varaible input B
            auto input_B_wrapper = inputs[1].dynamicCast<CannBackendWrapper>();
            auto input_B_desc = input_B_wrapper->getTensorDesc();
            auto input_B_node = nodes[1].dynamicCast<CannBackendNode>()->getOp();
            op->set_input_x2_by_name(*input_B_node, "y");
            op->update_input_desc_x2(*input_B_desc);
            if (inputs.size() >= 3) {
                auto input_bias_wrapper = inputs[2].dynamicCast<CannBackendWrapper>();
                auto input_bias_desc = input_bias_wrapper->getTensorDesc();
                auto input_bias_node = nodes[2].dynamicCast<CannBackendNode>()->getOp();
                op->set_input_bias_by_name(*input_bias_node, "y");
                op->update_input_desc_bias(*input_bias_desc);
            }
        } else { // constant input B
            auto B = blobs[0];
            auto const_B_node = std::make_shared<CannConstOp>(B.data, B.type(), shape(B), cv::format("%s_B", name.c_str()));
            op->set_input_x2_by_name(*(const_B_node->getOp()), "y");
            op->update_input_desc_x2(*(const_B_node->getTensorDesc()));
            if ((inputs.size() + blobs.size()) >= 3) { // does not support broadcast bias
                auto bias_mat = blobs.back();
                auto bias_shape = shape(bias_mat);

                // reshape if 1d
                if (real_ndims_C == 1 && bias_shape.front() != 1) {
                    bias_shape = std::vector<int>{bias_shape.front()};
                }

                auto const_bias_node = std::make_shared<CannConstOp>(bias_mat.data, bias_mat.type(), bias_shape, cv::format("%s_bias", name.c_str()));
                op->set_input_bias_by_name(*(const_bias_node->getOp()), "y");
                op->update_input_desc_bias(*(const_bias_node->getTensorDesc()));
            }
        }

        // set outputs
        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);
        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

 private:
    bool trans_a;
    bool trans_b;
    float alpha;
    float beta;

    int real_ndims_C;

    std::vector<float> packed_input_B;
    Mat broadcast_bias;

    FastGemmOpt opt;
    MatMulHelper helper;
};

Ptr<MatMulLayer> MatMulLayer::create(const LayerParams& params)
{
    return makePtr<MatMulLayerImpl>(params);
}

}} // cv::dnn
