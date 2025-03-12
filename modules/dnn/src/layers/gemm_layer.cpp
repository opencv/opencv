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

enum class LayerGemmOpMode {
    blobB,
    blobBC,
    blobC,
    noblob
};

bool constB(LayerGemmOpMode mode){
    switch (mode) {
        case LayerGemmOpMode::blobB:
        case LayerGemmOpMode::blobBC:
            return true;
        default:
            return false;
    }
}

bool constC(LayerGemmOpMode mode){
    switch (mode) {
        case LayerGemmOpMode::blobC:
        case LayerGemmOpMode::blobBC:
            return true;
        default:
            return false;
    }
}


// Y = alpha * A’ * B’ + beta * C
class GemmLayerImpl CV_FINAL : public GemmLayer {
public:
    GemmLayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        trans_a = params.get<bool>("transA", false);
        trans_b = params.get<bool>("transB", false);
        alpha = params.get<float>("alpha", 1.0f);
        beta = params.get<float>("beta", 1.0f);

        real_ndims_C = params.get<int>("real_ndims_C", -1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV ||
               (backendId == DNN_BACKEND_CUDA && const_B && !trans_a) ||
               backendId == DNN_BACKEND_CANN ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
               (backendId == DNN_BACKEND_VKCOM && haveVulkan() && !have_bias && !trans_a);
    }



    LayerGemmOpMode getOpMode(size_t n_inputs, size_t n_blobs) const {
        if (n_blobs == 0) return LayerGemmOpMode::noblob;
        if (n_inputs == 3) return LayerGemmOpMode::noblob ; // if all inputs are given, then no blobs are used
        if (n_inputs == 2) {
            if (have_bias) {
                // check where the input comes from
                if(const_B) 
                    return LayerGemmOpMode::blobB;
                if(const_C)
                    return LayerGemmOpMode::blobC;
                return LayerGemmOpMode::blobC;
            } else {
                if (n_blobs == 1) 
                    // 2 inputs, no bias => input[1] is B
                    return LayerGemmOpMode::noblob;
                if (n_blobs == 2)
                    return LayerGemmOpMode::blobC;
            }
        }
        if (n_inputs == 1) {
            // only A is given per input
            if (n_blobs == 2) 
                return LayerGemmOpMode::blobBC;
            else
                return LayerGemmOpMode::blobB;
        }
        CV_Error(Error::StsError, "DNN/Gemm: could not derive OP mode"); 
    }


    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        int num_inputs = static_cast<int>(inputs.size() + blobs.size());

        CV_CheckGE(num_inputs, 2, "DNN/Gemm: Gemm takes at least two inputs"); 
        CV_CheckLE(num_inputs, 3, "DNN/Gemm: Gemm takes at most three inputs");
        
        LayerGemmOpMode mode = getOpMode(inputs.size(), blobs.size());

        // Check whether A and B are two dimensional
        const auto shape_A = inputs[0];
        const auto shape_B =  constB(mode) ? shape(blobs[0]) : inputs[1];
        CV_CheckGE(shape_A.size(), static_cast<size_t>(2), "DNN/Gemm: Tensor A must be n-dimensional (n >= 2)");
        CV_CheckEQ(shape_B.size(), static_cast<size_t>(2), "DNN/Gemm: Tensor B must be two dimensional");

        // Check legal matrix multiplication
        size_t dims_A = shape_A.size();
        int ma = shape_A[dims_A - 2], na = shape_A[dims_A - 1];
        int mb = shape_B[0], nb = shape_B[1];
        int M = trans_a ? na : ma;
        int N = trans_b ? mb : nb;
        int K_a = trans_a ? ma : na;
        int K_b = trans_b ? nb : mb;


        CV_CheckEQ(K_a, K_b, "DNN/Gemm: Invalid dimension of dim K");

        // Check whether C can be unidirectional broadcast to (M, N). Handle carefully with 1D Mat.
        if (have_bias) {
            const auto shape_C = constC(mode) ? shape(blobs.back()) : inputs.back();

            auto ndims_C = shape_C.size();
            CV_CheckLE(ndims_C, static_cast<size_t>(2), "DNN/Gemm: C can only be 0d (scalar) / 1d / 2d tensor");

            int real_ndims_C_ = real_ndims_C >= 0 ? real_ndims_C : ndims_C;

            if (real_ndims_C_ == 1) { // (1,) or (N,)
                CV_Check(shape_C[0], shape_C[0] == 1 || shape_C[0] == N, "DNN/Gemm: invalid dimension of C");
            } else if (real_ndims_C_ == 2) { // (1, 1) or (1, N) or (M, 1) or (M, N)
                // printf("shape_C=[%d, %d]\n", shape_C[0], shape_C[1]);
                CV_Check(shape_C[0], (shape_C[0] == 1 && shape_C[1] == 1) ||
                                     (shape_C[0] == 1 && shape_C[1] == N) ||
                                     (shape_C[0] == M && shape_C[1] == 1) ||
                                     (shape_C[0] == M && shape_C[1] == N),
                                     "DNN/Gemm: C must be of shape (1, 1) or (1, N) or (M, 1) or (M, N)");
                if (shape_C[0] == 1) {
                    CV_Check(shape_C[1], shape_C[1] == 1 || shape_C[1] == N, "DNN/Gemm: invalid dimension of C");
                } else if (shape_C[0] == M) {
                    CV_Check(shape_C[1], shape_C[1] == 1 || shape_C[1] == N, "DNN/Gemm: invalid dimension of C");
                } else {
                    CV_Error(Error::StsBadSize, "DNN/Gemm: invalid dimension of C");
                }
            }
        }

        int batches = std::accumulate(shape_A.begin(), shape_A.end() - 2, 1, std::multiplies<int>());
        MatShape shape_y{M * batches, N};
        outputs.assign(1, shape_y);
        return false;
    }

    // TODO: replace with cv::broadcast() once 1d mat is supported
    // FIXME: fix if conditions if 1d mat is supported properly
    void broadcastCWtihBeta(int M, int N, const Mat &C) {
        broadcast_C.clear();
        broadcast_C.resize(M * N, 0.f);
        if (beta != 0 && !C.empty()) {
            int real_ndims_C_ = real_ndims_C >= 0 ? real_ndims_C : C.dims;

            const float *ptr_c = C.ptr<const float>();
            const auto shape_C = shape(C);
            if ((real_ndims_C_ == 0) || (real_ndims_C_ == 1 && shape_C[0] == 1) ||
                (real_ndims_C_ == 2 && shape_C[0] == 1 && shape_C[1] == 1)) {
                // (), (1,), (1, 1)
                float c = *ptr_c;
                int total = M * N;
                for (int i = 0; i < total; ++i) {
                    broadcast_C[i] = beta * c;
                }
            } else if ((real_ndims_C_ == 1 && shape_C[0] == N) ||
                       (real_ndims_C_ == 2 && shape_C[0] == 1 && shape_C[1] == N)) {
                // (N,), (1, N)
                for (int i = 0; i < M; ++i) {
                    int step = i * N;
                    for (int j = 0; j < N; ++j) {
                        broadcast_C[step + j] = beta * ptr_c[j];
                    }
                }
            } else if (real_ndims_C_ == 2 && shape_C[0] == M && shape_C[1] == 1) {
                // (M, 1)
                for (int i = 0; i < M; ++i) {
                    int step = i * N;
                    for (int j = 0; j < N; ++j) {
                        broadcast_C[step + j] = beta * ptr_c[i];
                    }
                }
            } else {
                // (M, N)
                std::transform(ptr_c, ptr_c + M * N, broadcast_C.begin(), [this] (const float &c) {
                    return this->beta * c; });
            }
        }
    }



    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        opt.init();
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        LayerGemmOpMode mode = getOpMode(inputs.size(), blobs.size());

        // pack B if it is const
        if (constB(mode)) {
            fastGemmPackB(blobs[0], packed_B, trans_b, opt);
        }

        // also pre-broadcast bias
        if (constC(mode)) {
            const auto &C = blobs.back();

            std::vector<Mat> outputs;
            outputs_arr.getMatVector(outputs);
            const auto &Y = outputs[0];
            const auto shape_Y = shape(Y);
            size_t dims_Y = shape_Y.size();
            int M = shape_Y[dims_Y - 2], N = shape_Y[dims_Y - 1];

            // broadcast
            broadcastCWtihBeta(M, N, C);
        }
    }

    // Y = A * B + C, note that C is unidirectionaly broadcastable to (A * B).
    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
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

        LayerGemmOpMode mode = getOpMode(inputs.size(), blobs.size());

        const auto &A = inputs[0];
        auto &Y = outputs[0];

        const auto shape_A = shape(A), shape_Y = shape(Y);
        size_t dims_A = shape_A.size();
        int ma = shape_A[dims_A - 2], na = shape_A[dims_A - 1];
        size_t dims_Y = shape_Y.size();
        int M = shape_Y[dims_Y - 2], N = shape_Y[dims_Y - 1];
        int K = trans_a ? ma : na;

        // broadcast C and copy C to output
        if (constC(mode) || inputs.size() >= 3) {
            if (!constC(mode) || broadcast_C.empty()) {
                broadcastCWtihBeta(M, N, (inputs.size() >= 3 ? inputs.back() : blobs.back()));
            }
            int step = M * N;
            CV_CheckEQ(broadcast_C.size(), static_cast<size_t>(step), "DNN/Gemm: C is not broadcast properly");
            float *ptr_y = Y.ptr<float>();
            std::memcpy(ptr_y, broadcast_C.data(), step * sizeof(float));
        } else { // initialization
            float *ptr_y = Y.ptr<float>();
            size_t total = Y.total();
            std::memset(ptr_y, 0, total * sizeof(float));
        }

        if (constB(mode)) {
            CV_CheckGT(packed_B.size(), static_cast<size_t>(0), "DNN/Gemm: constant B is not pre-packed");
            fastGemm(trans_a, M, N, K, alpha, A.ptr<const float>(), na, packed_B.data(), 1.f, Y.ptr<float>(), N, opt);
        } else {
            fastGemmBatch(trans_a, trans_b, alpha, A, inputs[1], 1.f, Y, opt);
        }
    }

#ifdef HAVE_CUDA
    // Y = A * B + C. B should be guaranteed as two dimensional.
    Ptr<BackendNode> initCUDA(void *context_,
                              const std::vector<Ptr<BackendWrapper>>& inputs,
                              const std::vector<Ptr<BackendWrapper>>& outputs) CV_OVERRIDE {
        CV_CheckFalse(trans_a, "DNN/Gemm/Cuda: does not support transA");
        CV_CheckTrue(const_B, "DNN/Gemm/Cuda: input B (weight) is required to be constant");
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        auto wrapper_A = inputs[0].dynamicCast<CUDABackendWrapper>();
        auto B = blobs[0];
        auto C = have_bias && const_C ? blobs[1] : Mat(); // in most cases C is constant

        if (!trans_b)
            cv::transpose(B, B);
        auto flatten_start_axis = normalize_axis(1, wrapper_A->getRank());
        return make_cuda_node<cuda4dnn::InnerProductOp>(preferableTarget, std::move(context->stream), std::move(context->cublas_handle), flatten_start_axis, B, C);
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
        // TODO: clearify if the bias needs to be constant here
        auto mat_C = have_bias && const_C ? blobs.back() : Mat::zeros(1, 1, CV_32F);
        auto shape_C = shape(mat_C);
        if (real_ndims_C == 1) {
            int dim = static_cast<int>(mat_C.total());
            shape_C = std::vector<int>{dim};
        }
        auto op_const_C = std::make_shared<CannConstOp>(mat_C.data, mat_C.type(), shape_C, cv::format("%s_b", name.c_str()));
        op->set_input_bias(*(op_const_C->getOp()));
        op->update_input_desc_bias(*(op_const_C->getTensorDesc()));

        // set outputs
        auto output_desc = std::make_shared<ge::TensorDesc>(ge::Shape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
        op->update_output_desc_y(*output_desc);
        return Ptr<BackendNode>(new CannBackendNode(op));
    }
#endif // HAVE_CANN

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        ov::Output<ov::Node> nodeA = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        ov::Output<ov::Node> nodeB;
        if (const_B)
            nodeB = std::make_shared<ov::op::v0::Constant>(ov::element::f32, getShape(blobs[0]), blobs[0].data);
        else
            nodeB = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;

        int flatten_axis = nodeA.get_shape().size() - nodeB.get_shape().size();
        if (flatten_axis > 0) {
            std::vector<int> shape(1 + flatten_axis, 0);
            shape[shape.size() - 1] = -1;
            nodeA = std::make_shared<ov::op::v1::Reshape>(
                nodeA,
                std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{shape.size()}, shape.data()),
                true);
        }

        std::shared_ptr<ov::Node> nodeAB = std::make_shared<ov::op::v0::MatMul>(nodeA, nodeB, trans_a, trans_b);
        if (alpha != 1.0f)
        {
            nodeAB = std::make_shared<ov::op::v1::Multiply>(
                nodeAB,
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &alpha));
        }

        if (!have_bias)
            return Ptr<BackendNode>(new InfEngineNgraphNode(nodeAB));

        ov::Output<ov::Node> nodeC;
        if (const_C)
        {
            auto shape_C = blobs.back().total() == blobs.back().size[0] ? ov::Shape{blobs.back().total()} : getShape(blobs.back());
            nodeC = std::make_shared<ov::op::v0::Constant>(ov::element::f32, shape_C, blobs.back().data);
        }
        else
        {
            nodeC = nodes.back().dynamicCast<InfEngineNgraphNode>()->node;
        }

        if (beta != 1.0f)
        {
            nodeC = std::make_shared<ov::op::v1::Multiply>(
                nodeC,
                std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, &beta));
        }

        auto nodeGemm = std::make_shared<ov::op::v1::Add>(nodeAB, nodeC, ov::op::AutoBroadcastType::NUMPY);
        return Ptr<BackendNode>(new InfEngineNgraphNode(nodeGemm));
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
