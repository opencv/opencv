// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
// backends
#include "../op_cuda.hpp"
#ifdef HAVE_CUDA
// #include "../cuda4dnn/primitives/matmul.hpp"
#include "../cuda4dnn/primitives/inner_product.hpp"
using namespace cv::dnn::cuda4dnn;
#endif
#include "../op_cann.hpp"
#include "../ie_ngraph.hpp"
#include "../op_vkcom.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include "cpu_kernels/fast_gemm.hpp"

namespace cv { namespace dnn {

class GemmLayerImpl CV_FINAL : public GemmLayer {
public:
    GemmLayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        trans_a = params.get<bool>("transA", false);
        trans_b = params.get<bool>("transB", false);
        alpha = params.get<float>("alpha", 1.0f);
        beta = params.get<float>("beta", 1.0f);

        const_B = params.get<bool>("constB", false); // true means blobs[0] is B
        const_C = params.get<bool>("constC", false); // true means blobs.back() is C
        have_bias = params.get<bool>("have_bias", false); // NOTE: have_bias being true does not mean bias is constant

        real_ndims_C = params.get<int>("real_ndims_C", -1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV ||
               (backendId == DNN_BACKEND_CUDA && const_B && !trans_a) ||
               backendId == DNN_BACKEND_CANN ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
               (backendId == DNN_BACKEND_VKCOM && haveVulkan() && !have_bias && !trans_a);
    }

    // TODO: replace with cv::Mat.broadcast() once https://github.com/opencv/opencv/pull/23965 is merged.
    void prepare_broadcast_C(int M, int N, const Mat& C) {
        broadcast_C.clear();
        broadcast_C.resize(M * N);
        const auto shape_C = shape(C);
        float *ptr_bc = broadcast_C.data();
        const float *ptr_c = C.ptr<const float>();
        if (real_ndims_C == 0 || (real_ndims_C == 1 && shape_C[0] == 1) ||
            (real_ndims_C == 2 && shape_C[0] == 1 && shape_C[1] == 1)) {
            // (), (1,), (1, 1)
            float c = *ptr_c;
            int total = M * N;
            for (int i = 0; i < total; ++i) {
                ptr_bc[i] = c;
            }
        } else if ((real_ndims_C == 1 && shape_C[0] != 1) ||
                   (real_ndims_C == 2 && shape_C[0] == 1)) {
            // (N,), (1, N)
            for (int i = 0; i < M; ++i) {
                std::memcpy(ptr_bc + i * N, ptr_c, N * sizeof(float));
            }
        } else if (real_ndims_C == 2 && shape_C[1] == 1) {
            // (M, 1)
            for (int i = 0; i < M; ++i) {
                int step = i * M;
                for (int j = 0; j < N; ++j) {
                    ptr_bc[step + j] = ptr_c[i];
                }
            }
        } else {
            // (M, N)
            std::memcpy(ptr_bc, ptr_c, M * N * sizeof(float));
        }
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        opt.init();

        // pack B if it is const
        if (const_B) {
            /*
                blobs:
                    b,
                    b, bias
            */
            fast_gemm_packB(blobs[0], packed_B, trans_b, opt);
        }

        // also pre-broadcast bias
        if (const_C) {
            const auto &C = blobs.back();

            std::vector<Mat> outputs;
            outputs_arr.getMatVector(outputs);
            const auto &Y = outputs[0];
            const auto shape_Y = shape(Y);
            int M = shape_Y[0], N = shape_Y[1];

            // broadcast
            prepare_broadcast_C(M, N, C);
        }
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        int num_inputs = static_cast<int>(inputs.size() + blobs.size());
        CV_CheckGE(num_inputs, 2, "DNN/Gemm: Gemm takes at least two inputs");
        CV_CheckLE(num_inputs, 3, "DNN/Gemm: Gemm takes at most three inputs");

        // Check whether A and B are two dimensional
        const auto shape_A = inputs[0];
        const auto shape_B = const_B ? shape(blobs[0]) : inputs[1];
        CV_CheckEQ(shape_A.size(), static_cast<size_t>(2), "DNN/Gemm: Tensor A must be two dimensional");
        CV_CheckEQ(shape_B.size(), static_cast<size_t>(2), "DNN/Gemm: Tensor B must be two dimensional");

        // Check legal matrix multiplication
        int ma = shape_A[0], na = shape_A[1];
        int mb = shape_B[0], nb = shape_B[1];
        int M = trans_a ? na : ma;
        int N = trans_b ? mb : nb;
        int K_a = trans_a ? ma : na;
        int K_b = trans_b ? nb : mb;
        CV_CheckEQ(K_a, K_b, "DNN/Gemm: Invalid dimension of dim K");

        // Check whether C can be unidirectional broadcast to (M, N). Handle carefully with 1D Mat.
        if (have_bias) {
            const auto shape_C = const_C ? shape(blobs.back()) : inputs.back();

            auto ndims_C = shape_C.size();
            CV_CheckLE(ndims_C, static_cast<size_t>(2), "DNN/Gemm: C can only be 0d (scalar) / 1d / 2d tensor");

            if (real_ndims_C == 1) { // (1,) or (N,)
                CV_Check(shape_C[0], shape_C[0] == 1 || shape_C[0] == N, "DNN/Gemm: C must be either of shape (1,) or (N,)");
            } else if (real_ndims_C == 2) { // (1, 1) or (1, N) or (M, 1) or (M, N)
                // printf("shape_C=[%d, %d]\n", shape_C[0], shape_C[1]);
                CV_Check(shape_C[0], (shape_C[0] == 1 && shape_C[1] == 1) ||
                                     (shape_C[0] == 1 && shape_C[1] == N) ||
                                     (shape_C[0] == M && shape_C[1] == 1) ||
                                     (shape_C[0] == M && shape_C[1] == N),
                                     "DNN/Gemm: C must be of shape (1, 1) or (1, N) or (M, 1) or (M, N)");
            }
        }

        MatShape shape_y{M, N};
        outputs.assign(1, shape_y);
        return false;
    }

    // Y = A * B + C, note that C is unidirectionaly broadcastable to (A * B).
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

        const auto &A = inputs[0];
        auto &Y = outputs[0];

        const auto shape_A = shape(A), shape_Y = shape(Y);
        int ma = shape_A[0], na = shape_A[1];
        int M = shape_Y[0], N = shape_Y[1];
        int K = trans_a ? ma : na;

        // broadcast C and copy C to output
        if (have_bias) {
            if (!const_C) {
                prepare_broadcast_C(M, N, inputs.back());
            }
            CV_CheckEQ(broadcast_C.size(), static_cast<size_t>(M * N), "DNN/Gemm: broadcast_C is not prepared");
            float *ptr_y = Y.ptr<float>();
            std::memcpy(ptr_y, broadcast_C.data(), M * N * sizeof(float));
        } else { // initialization
            float *ptr_y = Y.ptr<float>();
            std::memset(ptr_y, 0, M * N * sizeof(float));
        }

        if (const_B) {
            CV_CheckGT(packed_B.size(), static_cast<size_t>(0), "DNN/Gemm: constant B is not pre-packed");
            fast_gemm(trans_a, M, N, K, alpha, A.ptr<const float>(), na, packed_B.data(), beta, Y.ptr<float>(), N, opt);
        } else {
            fast_gemm(trans_a, trans_b, alpha, A, inputs[1], beta, Y, opt);
        }
    }

#ifdef HAVE_CUDA
    // Y = A * B + C. A and B should be guaranteed as two dimensional.
    Ptr<BackendNode> initCUDA(void *context_,
                              const std::vector<Ptr<BackendWrapper>>& inputs,
                              const std::vector<Ptr<BackendWrapper>>& outputs) CV_OVERRIDE {
        CV_CheckFalse(trans_a, "DNN/Gemm/Cuda: does not support transA");
        CV_CheckTrue(const_B, "DNN/Gemm/Cuda: input B (weight) is required to be constant");
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        auto B = blobs[0];
        auto C = have_bias && const_C ? blobs[1] : Mat();

        if (!trans_b)
            cv::transpose(B, B);
        return make_cuda_node<cuda4dnn::InnerProductOp>(preferableTarget, std::move(context->stream), std::move(context->cublas_handle), 1, B, C);
    }
#endif // HAVE_CUDA

#ifdef HAVE_CANN
    // Y = A * B + C.
    virtual Ptr<BackendNode> initCann(const std::vector<Ptr<BackendWrapper> > &inputs,
                                      const std::vector<Ptr<BackendWrapper> > &outputs,
                                      const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE {
        auto x1 = inputs[0].dynamicCast<CannBackendWrapper>();
        auto desc_x1 = x1->getTensorDesc();
        auto op_x1 = nodes[0].dynamicCast<CannBackendNode>()->getOp();

        auto op = std::make_shared<ge::op::MatMulV2>(name);

        // set attributes
        op->set_attr_transpose_x1(trans_a);
        op->set_attr_transpose_x2(trans_b);

        // set inputs
        // set inputs : x1
        op->set_input_x1_by_name(*op_x1, x1->name.c_str());
        op->update_input_desc_x1(*desc_x1);
        // set inputs : x2
        if (const_B) {
            auto B = blobs[0];
            auto op_const_B = std::make_shared<CannConstOp>(B.data, B.type(), shape(B), cv::format("%s_w", name.c_str()));
            op->set_input_x2_by_name(*(op_const_B->getOp()), "y");
            op->update_input_desc_x2(*(op_const_B->getTensorDesc()));
        } else {
            CV_CheckGE(inputs.size(), static_cast<size_t>(2), "DNN/Gemm/CANN: input B is required since it is not constant");
            CV_CheckGE(nodes.size(), static_cast<size_t>(2), "DNN/Gemm/CANN: input B is required since it is not constant");
            auto op_x2 = nodes[1].dynamicCast<CannBackendNode>()->getOp();
            auto desc_x2 = inputs[1].dynamicCast<CannBackendWrapper>()->getTensorDesc();
            op->set_input_x2_by_name(*op_x2, "y");
            op->update_input_desc_x2(*desc_x2);
        }
        // set inputs : bias
        auto mat_C = have_bias && const_C ? blobs.back() : Mat::zeros(1, 1, CV_32F);
        auto op_const_C = std::make_shared<CannConstOp>(mat_C.data, mat_C.type(), shape(mat_C), cv::format("%s_b", name.c_str()));
        op->set_input_bias(*(op_const_C->getOp()));
        op->update_input_desc_bias(*(op_const_C->getTensorDesc()));

        // set outputs
        op->update_output_desc_y(*output_desc);
        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::shared_ptr<ngraph::Node> matmul;
        int axis = -2;

        if (nodes.size() == 2)
        {
            auto& inp2 = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
            matmul = std::make_shared<ngraph::op::MatMul>(ieInpNode, inp2, transA, transB);
        }
        else
        {
            std::vector<int> shape(1 + normalize_axis(axis, ieInpNode->get_shape().size()), 0);
            shape[shape.size() - 1] = -1;
            auto inp = std::make_shared<ngraph::op::v1::Reshape>(
                ieInpNode,
                std::make_shared<ngraph::op::Constant>(ngraph::element::i32, ngraph::Shape{shape.size()}, shape.data()),
                true
            );

            std::vector<size_t> weight_shape{(size_t)blobs[0].size[0], (size_t)blobs[0].size[1]};
            auto ieWeights = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, weight_shape, blobs[0].data);
            matmul = std::make_shared<ngraph::op::MatMul>(inp, ieWeights, transA, transB);
        }

        if (have_bias && const_C) {
            auto bias_node = std::make_shared<ngraph::op::Constant>(ngraph::element::f32,
                                              ngraph::Shape{(size_t)blobs.back().size[1]}, blobs.back().data);
            matmul = std::make_shared<ngraph::op::v1::Add>(matmul, bias_node, ngraph::op::AutoBroadcastType::NUMPY);
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(matmul));
    }
#endif // HAVE_DNN_NGRAPH

#ifdef HAVE_VULKAN
    // Y = A * B + C. Currently support 2d matrix multiplication without bias.
    virtual Ptr<BackendNode> initVkCom(const std::vector<Ptr<BackendWrapper> > &inputs,
                                       std::vector<Ptr<BackendWrapper> > &outputs) CV_OVERRIDE
    {
        // does not support with bias; only 2d matmul
        auto wrapper_Y = outputs[0].dynamicCast<VkComBackendWrapper>();
        auto shape_Y = shape(*(wrapper_Y->getMat()));
        if (have_bias || shape_Y.size() > static_cast<size_t>(2)) {
            return Ptr<BackendNode>();
        }

        std::vector<Mat> vkBlobs;
        if (const_B) {
            vkBlobs.push_back(blobs[0]);
        }

        auto wrapper_A = inputs[0].dynamicCast<VkComBackendWrapper>();
        auto shape_A = shape(*wrapper_A->getMat());
        Ptr<vkcom::OpBase> op = (new vkcom::OpMatMul(vkBlobs, shape_A[0], shape_A[1], shape_Y[1]));
        return Ptr<BackendNode>(new VkComBackendNode(inputs, op, outputs));
    }
#endif

private:
    bool const_B;
    bool const_C;
    bool have_bias;
    std::vector<float> packed_B;
    std::vector<float> broadcast_C;
    int real_ndims_C;
    FastGemmOpt opt;
};

Ptr<GemmLayer> GemmLayer::create(const LayerParams& params) {
    return makePtr<GemmLayerImpl>(params);
}

}} // namespace cv::dnn
