// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../net_impl.hpp"
#include "layers_common.hpp"
#include "cpu_kernels/fusion_apply.hpp"

#include <type_traits>
#include <opencv2/dnn/shape_utils.hpp>
#include "cpu_kernels/fast_gemm.hpp"
#include "cpu_kernels/mlas_gemm.hpp"

// OpenVINO backend
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

// Vulkan backend
#include "../op_vkcom.hpp"

// CUDA backend
#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/matmul_broadcast.hpp"
#include "fusion_cuda.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

// CANN backend
#include "../op_cann.hpp"

namespace cv { namespace dnn {

class MatMulLayerImpl CV_FINAL : public MatMulLayer, public ConstWeightOwner {
#ifdef HAVE_OPENCL
    UMat weight_umat, bias_umat;
#endif
    PreparedFusion fusion;

 public:
    static bool absorbOp(Layer* self, const Ptr<AdjacencyGraph>& expr)
    {
        return static_cast<MatMulLayerImpl*>(self)->fusion.take(expr);
    }

    MatMulLayerImpl(const LayerParams& params) {
        registerFusionOpsOnce<MatMulLayerImpl>({ nullptr, &MatMulLayerImpl::absorbOp });
        setParamsFrom(params);

        trans_a = params.get<bool>("transA", false);
        trans_b = params.get<bool>("transB", false);
        alpha = params.get<float>("alpha", 1.f);
        beta = params.get<float>("beta", 1.f);

        real_ndims_C = params.get<int>("real_ndims_C", -1);

        // Built straight from LayerParams, the constants arrive in blobs rather than as
        // graph inputs; take them over so there is one owner either way.
        if (!blobs.empty()) {
            setWeights(blobs[0], blobs.size() > 1 ? blobs[1] : Mat());
            blobs.clear();
        }
    }

    // constArgs() hands the constant operands over once; the layer owns them from here.
    void setWeights(InputArray weights_arr, InputArray bias_arr) CV_OVERRIDE
    {
        // An empty weights keeps the current one; on the first handover there is none to
        // keep, and a zero-sized dimension is a legal degenerate B.
        Mat w = weights_arr.getMat();
        if (!w.empty() || !ownsWeights) {
            widenToFP32(w);
            weights = w;
            wshape0 = shape(w);
            ownsWeights = true;
            // A recycled address would compare equal and skip the repack.
            last_packed_input_B_data = nullptr;
            // The copy initCUDA() was going to read is now the wrong one.
            origWeights.release();
        }

        Mat b = bias_arr.getMat();
        if (!b.empty()) {
            widenToFP32(b);
            bias = b;
            ownsBias = true;
        }
    }

    const Mat& constWeights() const CV_OVERRIDE { return weights; }
    const Mat& constBias() const CV_OVERRIDE { return bias; }

    // wshape0 and the ownership flags survive, as they do after packing.
    void releaseConstWeights() CV_OVERRIDE
    {
        weights.release();
        origWeights.release();
        bias.release();
    }

    //! Stays correct after finalize() has packed and dropped the weight.
    size_t numConsts() const { return (size_t)ownsWeights + (size_t)ownsBias; }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
#ifdef HAVE_CUDA
        // CUDA runs a fused activation or scale/shift after the MatMul; N is known for a constant B
        const MatShape bshape = ownsWeights ? constBShape() : MatShape();
        const int N = (!ownsWeights || bshape.dims < 2) ? -1
                      : bshape[bshape.dims - (trans_b ? 2 : 1)];
        const bool cudaFusion = backendId == DNN_BACKEND_CUDA && cudaSupportsFusion(fusion, N);
#else
        const bool cudaFusion = false;
#endif
        if (fusion.expr && !cudaFusion)
            return backendId == DNN_BACKEND_OPENCV;
#ifdef HAVE_CUDA
        if (backendId == DNN_BACKEND_CUDA) {
            // classic-engine nets carry no graph args to inspect; keep their previous answer
            if (inputs.empty())
                return true;
            // MatMulBroadcastOp applies neither alpha nor a scaled bias, and adds the bias without broadcasting
            if (alpha != 1.f || beta != 1.f)
                return false;
            const bool constB = inputs.size() == 1 && numConsts() >= 1;
            const bool inputB = !ownsWeights && inputs.size() == 2;
            if (!constB && !inputB)
                return false;
            Net::Impl* netimpl_ = getNetImpl(this);
            if (!netimpl_)
                return false;
            // an intermediate arg has no type until the first forward; refuse only a known non-float one
            for (const Arg& a : inputs) {
                const int t = netimpl_->argData(a).type;
                if (t >= 0 && t != CV_32F)
                    return false;
            }
            if ((!weights.empty() && weights.type() != CV_32F) ||
                (!bias.empty() && bias.type() != CV_32F))
                return false;
            // gemmBatched reads the last two axes of A, B and Y; 1-D operands stay on CPU
            for (const Arg& a : inputs) {
                const int dims = netimpl_->argData(a).shape.dims;
                if (dims >= 0 && dims < 2)
                    return false;
            }
            if (ownsWeights && constBShape().dims < 2)
                return false;
            return true;
        }
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH ||
               (backendId == DNN_BACKEND_VKCOM && haveVulkan() && !trans_a && !trans_b) ||
               backendId == DNN_BACKEND_CANN;
    }

    // finalize() drops B once it is packed, so its shape has to outlive it
    MatShape constBShape() const { return wshape0; }

    virtual std::ostream& dumpAttrs(std::ostream& strm, int indent) const CV_OVERRIDE
    {
        if (!wshape0.empty()) {
            prindent(strm, indent);
            dumpWeightShape(strm, wshape0);
        }
        return strm;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        int num_inputs = inputs.size() + numConsts();
        CV_CheckGE(num_inputs, 2, "DNN/MatMul: two inputs at least");
        CV_CheckLE(num_inputs, 3, "DNN/MatMul: three inputs at most");

        const auto shape_A = inputs[0], shape_B = !ownsWeights ? inputs[1] : constBShape();
        CV_CheckGE(shape_A.size(), static_cast<size_t>(1), "DNN/MatMul: invalid shape of input A");
        CV_CheckGE(shape_B.size(), static_cast<size_t>(1), "DNN/MatMul: invalid shape of input B");

        MatShape shape_Ap, shape_Bp, common_shape, out_shape;
        matmulShapes(shape_A, shape_B, trans_a, trans_b, shape_Ap, shape_Bp, common_shape, out_shape);
        int N = common_shape.back();

        // Check if bias is broadcastable
        if (num_inputs == 3) {
            const auto shape_C = !ownsBias ? inputs.back() : shape(bias);
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

        outputs.assign(1, out_shape);
        return false;
    }

    // Only types forward() actually dispatches; the default gate's CV_8S/CV_8U had no kernel and corrupted memory.
    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size());
        for (auto input : inputs)
            CV_CheckType(input, input == CV_32F || input == CV_64F ||
                                input == CV_32S || input == CV_64S || input == CV_32U || input == CV_64U, "");

        outputs.assign(requiredOutputs, inputs[0]);
        internals.assign(requiredInternals, inputs[0]);
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        // Promote 1-D operands so shape.size()-2 can't underflow on a 1-D input.
        MatShape shape_Ap, shape_Bp, full_shape, out_shape;
        matmulShapes(inputs[0], !ownsWeights ? inputs[1] : constBShape(),
                     trans_a, trans_b, shape_Ap, shape_Bp, full_shape, out_shape);
        int mA = shape_Ap[shape_Ap.size() - 2], nA = shape_Ap.back();
        int M = trans_a ? nA : mA;
        int K = trans_a ? mA : nA;
        int N = full_shape.back();

        int64 batch = 1;
        for (size_t i = 0; i + 2 < full_shape.size(); i++)
            batch *= full_shape[i];

        // 2*M*N*K multiply-adds per batch element, +M*N for bias if present
        int64 flops = batch * (CV_BIG_INT(2) * M * N * K);
        int num_inputs = (int)inputs.size() + (int)numConsts();
        if (num_inputs == 3)
            flops += batch * M * N;
        return flops;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        opt.init();

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        MatShape A_shape, B_shape, C_shape, out_shape;
        matmulShapes(shape(inputs[0]), !ownsWeights ? shape(inputs[1]) : constBShape(),
                     trans_a, trans_b, A_shape, B_shape, C_shape, out_shape);
        helper.compute(trans_a, trans_b, A_shape, B_shape, C_shape);

        // These five types skip the float-only packed-B/MLAS caching below.
        {
            int depth0 = inputs[0].depth();
            if (depth0 == CV_64F || depth0 == CV_32S || depth0 == CV_64S || depth0 == CV_32U || depth0 == CV_64U)
                return;
        }

        // Pack only 2D weight matrices; skip higher-dim tensors (e.g. Q@K^T in attention).
        // An already-released B yields nullptr: it is packed and has nothing left to pack from.
        const Mat* B_mat = ownsWeights ? (weights.empty() ? nullptr : &weights) :
                           (inputs.size() >= 2 && inputs[1].dims == 2 ? &inputs[1] : nullptr);

        // A constant rank-1 weight ([K] in the logical [M, K] @ [K] -> [M] contract) is
        // still physically 1-D here; fastGemmPackB/fastGemmThinPackB read the last two
        // dims, so promote it to [K, 1] first. reshape() shares the blob's buffer, so
        // last_packed_input_B_data below still tracks the original data pointer.
        Mat promoted_B;
        if (B_mat && B_mat->dims == 1) {
            promoted_B = B_mat->reshape(1, std::vector<int>{B_mat->size[0], 1});
            B_mat = &promoted_B;
        }

        if (B_mat && helper.N > 0 && helper.K > 0 && B_mat->data != last_packed_input_B_data) {
            packed_input_B.clear();
            packed_input_B.shrink_to_fit();
            thin_packed_B.clear();

            if (helper.batch == 1 && B_mat->type() == CV_32F &&
                fastGemmThinEligible(helper.M, helper.N, helper.K)) {
                thin_packed_B.resize(fastGemmThinPackBSize(helper.N, helper.K));
                fastGemmThinPackB(helper.N, helper.K,
                                  B_mat->ptr<const float>(),
                                  (size_t)helper.ldb0, (size_t)helper.ldb1,
                                  thin_packed_B.data());
            } else {
                fastGemmPackB(*B_mat, packed_input_B, trans_b, opt);
                helper.updatePackedBOffsets(packed_input_B.size());
            }
            last_packed_input_B_data = B_mat->data;
        }

        // A layer built outside a Net holds the only copy of B, and so does one on the legacy
        // execution path, whose OpenVINO/Vulkan/CANN arms read it after finalize().
        Net::Impl* netimpl = getNetImpl(this);
        const bool ownedByGraph = netimpl && netimpl->mainGraph;

        bool keepForCuda = false;
#ifdef HAVE_CUDA
        // initCUDA() reads B after finalize(), so a net targeting CUDA keeps it.
        keepForCuda = ownedByGraph && netimpl->preferableBackend == DNN_BACKEND_CUDA &&
                      supportBackend(DNN_BACKEND_CUDA);
#endif

        // The packed copy is what the kernels read from here on; the original is dead weight.
        if (ownsWeights && !weights.empty() && ownedByGraph &&
            weights.type() == CV_32F &&
            (!packed_input_B.empty() || !thin_packed_B.empty())) {
            // initCUDA() reads the plain matrix after finalize(). Hold it out of reach of
            // getParam() so the contract does not vary by backend.
            if (keepForCuda)
                origWeights = weights;
            else
                netimpl->weightsReleased = true;
            weights.release();
            // A recycled address would compare equal and skip the repack.
            last_packed_input_B_data = nullptr;
        }
        // A net moved off CUDA has no reader left for the plain matrix.
        if (ownedByGraph && !keepForCuda && !origWeights.empty()) {
            origWeights.release();
            netimpl->weightsReleased = true;
        }

        // broadcast bias if needed
        if ((inputs.size() + numConsts()) >= 3 && ownsBias) {
            const auto bias_mat = bias;
            const auto bias_shape = shape(bias_mat);
            bool is_broadcast_needed = real_ndims_C == 0 || real_ndims_C == 1 || (total(bias_shape) != total(C_shape) || bias_shape.size() != C_shape.size());

            if (is_broadcast_needed) {
                broadcast_bias = Mat(C_shape, CV_32F);
                auto *broadcast_bias_ptr = broadcast_bias.ptr<float>();

                const auto *bias = bias_mat.ptr<const float>();
                // Stored unscaled: forward() applies beta once (gemm beta or explicitly), and the
                // OpenCL/nGraph/CUDA paths add scaledBroadcastBias() instead.
                if (bias_mat.total() == 1) { // [], [1], [1, ...]
                    float b = *bias;
                    for (size_t i = 0; i < broadcast_bias.total(); i++) {
                        broadcast_bias_ptr[i] = b;
                    }
                } else if (real_ndims_C == 1) { // [n]
                    size_t inner_size = C_shape.back(),
                        loops = total(C_shape) / inner_size;
                    for (size_t i = 0; i < loops; i++) {
                        size_t step = i * inner_size;
                        for (size_t j = 0; j < inner_size; j++) {
                            broadcast_bias_ptr[step + j] = bias[j];
                        }
                    }
                } else {
                    broadcast(bias_mat, C_shape, broadcast_bias);
                }
            } else {
                broadcast_bias = bias;
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
            if (fusion.expr) {
                std::vector<Mat> outs;
                outputs_arr.getMatVector(outs);
                if (!outs.empty()) {
                    Mat y32;
                    outs[0].convertTo(y32, CV_32F);
                    fusion.run(y32);
                    y32.convertTo(outs[0], outs[0].type());
                }
            }
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        switch (inputs[0].depth()) {
            case CV_64F: forwardDouble(inputs, outputs); return;
            case CV_32S: forwardInt<int32_t>(inputs, outputs); return;
            case CV_64S: forwardInt<int64_t>(inputs, outputs); return;
            case CV_32U: forwardInt<uint32_t>(inputs, outputs); return;
            case CV_64U: forwardInt<uint64_t>(inputs, outputs); return;
            default: break;
        }

        // Promote 1-D operands (numpy MatMul semantics) so leading dims match helper.
        Mat A = inputs[0];
        { MatShape sa = shape(A); if (sa.size() == 1) A = A.reshape(1, std::vector<int>{1, sa[0]}); }
        if (!ownsWeights) {
            MatShape sb = shape(inputs[1]);
            if (sb.size() == 1) inputs[1] = inputs[1].reshape(1, std::vector<int>{sb[0], 1});
        }
        auto &Y = outputs[0];

        const auto *a = A.ptr<const float>();
        auto *y = Y.ptr<float>();
        // Without a bias the gemm overwrites Y (beta = 0), so Y needs no zero-fill.
        const bool has_bias = (inputs.size() + numConsts()) >= 3;
        const float gemm_beta = has_bias ? beta : 0.f;
        // add bias if existed
        if (has_bias) {
            const auto &shape_Y = shape(Y);
            // Stored unscaled here; beta is applied exactly once below, either by the gemm's
            // own beta (M, N, K all > 0) or explicitly when the gemm is skipped.
            if (!ownsWeights) { // bias from input
                const auto &bias_mat = inputs.back();
                const auto *bias = bias_mat.ptr<const float>();
                if (bias_mat.total() == 1) { // [], [1], [1, ...]
                    float b = *bias;
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
                                y[output_offset + j] = bias[j];
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
        }
        // Any of M, N, K == 0 means the product contributes nothing, and finalize()
        // skips packing B whenever N or K is 0 - never reach the gemm dispatch (and its own
        // beta) below, so apply beta to the bias prefill here instead.
        if (helper.M == 0 || helper.N == 0 || helper.K == 0) {
            if (!has_bias)
                std::memset(y, 0, Y.total() * sizeof(float));
            else if (beta != 1.f)
                for (size_t i = 0; i < Y.total(); i++)
                    y[i] *= beta;
            fusion.run(Y);
            return;
        }

        if (!ownsWeights) {
            const auto &B = inputs[1];
            const auto *b = B.ptr<const float>();
            bool done = false;
            if (mlasAvailable() && !opt.use_rvv && helper.M > 0 && helper.N > 0 && helper.K > 0) {
                const auto A_shape = shape(A);
                const auto B_shape = shape(B);
                const int lda_mem = A_shape.back();
                const int ldb_mem = B_shape.back();
                done = mlasSgemmBatch(helper.batch,
                                      helper.A_offsets.data(), helper.B_offsets.data(), helper.C_offsets.data(),
                                      trans_a, trans_b, helper.M, helper.N, helper.K,
                                      alpha, a, lda_mem, b, ldb_mem, gemm_beta, y, helper.ldc);
            }
            if (!done) {
                fastGemmBatch(helper.batch, helper.A_offsets.data(), helper.B_offsets.data(), helper.C_offsets.data(),
                              helper.M, helper.N, helper.K, alpha, a, helper.lda0, helper.lda1,
                              b, helper.ldb0, helper.ldb1, gemm_beta, y, helper.ldc, opt);
            }
        } else if (!thin_packed_B.empty()) {
            fastGemmThin(helper.M, helper.N, helper.K, alpha,
                         a, helper.lda0, helper.lda1,
                         thin_packed_B.data(), gemm_beta,
                         y, helper.ldc, opt.multi_thread);
        } else if (!packed_input_B.empty()) {
            fastGemmBatch(helper.batch, helper.A_offsets.data(), helper.packed_B_offsets.data(), helper.C_offsets.data(),
                          helper.M, helper.N, helper.K, alpha, a, helper.lda0, helper.lda1,
                          packed_input_B.data(), gemm_beta, y, helper.ldc, opt);
        } else {
            // truly dynamic B (changes every call — no packing cache available)
            const auto &B = inputs[1];
            const auto *b = B.ptr<const float>();
            fastGemmBatch(helper.batch, helper.A_offsets.data(), helper.B_offsets.data(), helper.C_offsets.data(),
                          helper.M, helper.N, helper.K, alpha, a, helper.lda0, helper.lda1,
                          b, helper.ldb0, helper.ldb1, gemm_beta, y, helper.ldc, opt);
        }
        fusion.run(Y);
    }

    // CV_64F: one cv::gemm call per batch slice (batches don't collapse like Gemm's).
    void forwardDouble(const std::vector<Mat>& inputs, std::vector<Mat>& outputs)
    {
        const Mat &A = inputs[0];
        Mat &Y = outputs[0];
        const Mat &B = !ownsWeights ? inputs[1] : weights;

        CV_CheckTypeEQ(B.depth(), CV_64F, "DNN/MatMul: B must be CV_64F to match A");
        CV_Assert(A.isContinuous() && B.isContinuous() && Y.isContinuous());

        const auto shape_A = shape(A), shape_B = shape(B), shape_Y = shape(Y);
        const int ma = shape_A[shape_A.size() - 2], na = shape_A.back();
        const int mb = shape_B[shape_B.size() - 2], nb = shape_B.back();
        const int M = helper.M, N = helper.N;

        double* yptr0 = Y.ptr<double>();
        const bool haveBias = (inputs.size() + numConsts()) >= 3;
        if (haveBias) {
            const Mat& bias_mat = ownsBias ? bias : inputs.back();
            CV_CheckTypeEQ(bias_mat.depth(), CV_64F, "DNN/MatMul: bias must be CV_64F to match A");
            if (bias_mat.total() == 1) {
                std::fill_n(yptr0, Y.total(), (double)beta * bias_mat.ptr<double>()[0]);
            } else {
                Mat biasBroadcast(shape_Y, CV_64F);
                cv::broadcast(bias_mat, shape_Y, biasBroadcast);
                const double* bb = biasBroadcast.ptr<double>();
                double b = (double)beta;
                for (size_t i = 0; i < Y.total(); i++)
                    yptr0[i] = b * bb[i];
            }
        } else {
            std::fill_n(yptr0, Y.total(), 0.0);
        }

        const int flags = (trans_a ? GEMM_1_T : 0) | (trans_b ? GEMM_2_T : 0);
        const double* aptr0 = A.ptr<double>();
        const double* bptr0 = B.ptr<double>();

        for (size_t i = 0; i < helper.batch; i++) {
            Mat Aview(ma, na, CV_64F, (void*)(aptr0 + helper.A_offsets[i]));
            Mat Bview(mb, nb, CV_64F, (void*)(bptr0 + helper.B_offsets[i]));
            Mat Yview(M, N, CV_64F, (void*)(yptr0 + helper.C_offsets[i]));
            cv::gemm(Aview, Bview, (double)alpha, Yview, 1.0, Yview, flags);
        }
    }

    // No BLAS routine handles integers; accumulates in 64-bit Acc and wraps on cast.
    template<typename T>
    void forwardInt(const std::vector<Mat>& inputs, std::vector<Mat>& outputs)
    {
        typedef typename std::conditional<std::is_unsigned<T>::value, uint64_t, int64_t>::type Acc;

        const Mat &A = inputs[0];
        Mat &Y = outputs[0];
        const Mat &B = !ownsWeights ? inputs[1] : weights;

        CV_CheckTypeEQ(B.depth(), A.depth(), "DNN/MatMul: B must match A's type");
        CV_Assert(A.isContinuous() && B.isContinuous() && Y.isContinuous());

        const auto shape_Y = shape(Y);
        const int M = helper.M, N = helper.N, K = helper.K;
        const int lda0 = helper.lda0, lda1 = helper.lda1;
        const int ldb0 = helper.ldb0, ldb1 = helper.ldb1;
        const int ldc = helper.ldc;

        const T* aptr0 = A.ptr<T>();
        const T* bptr0 = B.ptr<T>();
        T* yptr0 = Y.ptr<T>();

        // Reject non-integer alpha/beta rather than round: rounding 0.4/0.2 to 0 would silently zero the result.
        const bool haveBias = (inputs.size() + numConsts()) >= 3;
        double alpha_d = (double)alpha, beta_d = haveBias ? (double)beta : 0.0;
        CV_CheckTrue(std::floor(alpha_d) == alpha_d, "DNN/MatMul: alpha must be an integer value for integer types");
        CV_CheckTrue(std::floor(beta_d) == beta_d, "DNN/MatMul: beta must be an integer value for integer types");
        const Acc alphaScale = (Acc)alpha_d;
        const Acc betaScale  = (Acc)beta_d;

        Mat biasBroadcast;
        if (haveBias) {
            const Mat& bias_mat = ownsBias ? bias : inputs.back();
            CV_CheckTypeEQ(bias_mat.depth(), A.depth(), "DNN/MatMul: bias must match A's type");
            if ((size_t)bias_mat.total() != Y.total() || shape(bias_mat).size() != shape_Y.size()) {
                biasBroadcast = Mat(shape_Y, A.depth());
                cv::broadcast(bias_mat, shape_Y, biasBroadcast);
            } else {
                biasBroadcast = bias_mat;
            }
        }
        const T* biasPtr0 = haveBias ? biasBroadcast.ptr<T>() : nullptr;

        parallel_for_(Range(0, (int)(helper.batch * (size_t)M)), [&](const Range& r) {
            for (int idx = r.start; idx < r.end; idx++) {
                int bi = idx / M;
                int m  = idx % M;
                const T* aBase = aptr0 + helper.A_offsets[bi];
                const T* bBase = bptr0 + helper.B_offsets[bi];
                T* yRow = yptr0 + helper.C_offsets[bi] + (size_t)m * ldc;
                const T* biasRow = haveBias ? biasPtr0 + helper.C_offsets[bi] + (size_t)m * ldc : nullptr;

                for (int n = 0; n < N; n++) {
                    Acc sum = 0;
                    for (int k = 0; k < K; k++) {
                        Acc av = (Acc)aBase[(size_t)m * lda0 + (size_t)k * lda1];
                        Acc bv = (Acc)bBase[(size_t)k * ldb0 + (size_t)n * ldb1];
                        sum += av * bv;
                    }
                    Acc result = sum * alphaScale;
                    if (haveBias)
                        result += betaScale * (Acc)biasRow[n];
                    yRow[n] = (T)result;
                }
            }
        }, (double)helper.batch * M * N * (1 / 1024.0));
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
        // batch_A/batch_B below divide by M*K and N*K; the CPU path handles empty dims
        if (helper.M == 0 || helper.N == 0 || helper.K == 0) {
            return false;
        }

        const auto &input_A = inputs[0];
        auto &output = outputs[0];
        const auto output_shape = shape(output);

        if (!ownsWeights) {
            weight_umat = inputs[1];
            if ((inputs.size() + numConsts() >= 3)) {
                bias_umat = UMat::zeros(output_shape.size(), output_shape.data(), CV_32F);
            }
        } else {
            if (weight_umat.empty()) {
                weights.copyTo(weight_umat);
            }
            if ((inputs.size() + numConsts() >= 3)) {
                if (bias_umat.empty()) {
                    scaledBroadcastBias().copyTo(bias_umat);
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

        if (!ownsWeights) {
            auto &input_B_node = nodes[1].dynamicCast<InfEngineNgraphNode>()->node;
            result = std::make_shared<ov::op::v0::MatMul>(input_A_node, input_B_node, trans_a, trans_b);
            if (nodes.size() >= 3) {
                bias = nodes[2].dynamicCast<InfEngineNgraphNode>()->node;
                if (beta != 1.f) {
                    auto coeff = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, &beta);
                    bias = std::make_shared<ov::op::v1::Multiply>(bias, coeff, ov::op::AutoBroadcastType::NUMPY);
                }
                result = std::make_shared<ov::op::v1::Add>(result, bias);
            }
        } else {
            auto input_B_shape = getShape<size_t>(weights);
            auto input_B_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, input_B_shape, weights.data);
            result = std::make_shared<ov::op::v0::MatMul>(input_A_node, input_B_node, trans_a, trans_b);
            if ((nodes.size() + numConsts()) >= 3) {
                const Mat bias_mat = scaledBroadcastBias();
                const auto bias_shape = shape(bias_mat);
                bias = std::make_shared<ov::op::v0::Constant>(ov::element::f32, std::vector<size_t>(bias_shape.begin(), bias_shape.end()), bias_mat.data);
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
        if ((inputs.size() + numConsts()) >= 3 || output_shape.size() != 2) {
            return Ptr<BackendNode>();
        }

        std::vector<Mat> constants;

        if (ownsWeights) {
            constants.push_back(weights);
        }

        Ptr<vkcom::OpBase> op = new vkcom::OpMatMul(constants, input_A_shape[0], input_A_shape[1], output_shape[1]);
        return Ptr<BackendNode>(new VkComBackendNode(inputs, op, outputs));
    }
#endif

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(void *context_,
                              InputArrayOfArrays,
                              InputArrayOfArrays outputs) override {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        auto input_B = Mat(), bias = Mat();
        if (ownsWeights) {
            input_B = origWeights.empty() ? weights : origWeights;
            CV_CheckFalse(input_B.empty(), "DNN/MatMul/CUDA: constant B was released");
            if (ownsBias) {
                bias = scaledBroadcastBias();
            }
        }

        CV_CheckFalse(helper.empty(), "DNN/MatMul/CUDA: MatMulHelper is not initialized");

        // the fused math runs on the same stream, right after the MatMul
        csl::Stream stream = context->stream;
        Ptr<BackendNode> matmulNode = make_cuda_node<cuda4dnn::MatMulBroadcastOp>(preferableTarget, std::move(context->stream), std::move(context->cublas_handle), input_B, bias, beta, trans_a, trans_b, helper.A_offsets, helper.B_offsets, helper.C_offsets, helper.batch);
        return makeFusedCudaNode(fusion, matmulNode, preferableTarget, stream, outputs.shape(0));
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
        if (!ownsWeights) { // variable input B
            auto input_B_wrapper = inputs[1].dynamicCast<CannBackendWrapper>();
            auto input_B_desc = input_B_wrapper->getTensorDesc();
            auto input_B_node = nodes[1].dynamicCast<CannBackendNode>()->getOp();
            op->set_input_x2_by_name(*input_B_node, "y");
            op->update_input_desc_x2(*input_B_desc);
            if (inputs.size() >= 3) {
                auto input_bias_wrapper = inputs[2].dynamicCast<CannBackendWrapper>();
                auto input_bias_desc = input_bias_wrapper->getTensorDesc();
                auto input_bias_node = nodes[2].dynamicCast<CannBackendNode>()->getOp();
                if (beta != 1.f) { // BatchMatMulV2 adds the bias unscaled
                    Mat beta_mat(1, 1, CV_32F, Scalar(beta));
                    auto const_beta_node = std::make_shared<CannConstOp>(
                        beta_mat.data, beta_mat.type(), std::vector<int>{1},
                        cv::format("%s_beta", name.c_str()));
                    auto mul = std::make_shared<ge::op::Mul>(
                        cv::format("%s_bias_beta", name.c_str()));
                    mul->set_input_x1_by_name(*input_bias_node, "y");
                    mul->update_input_desc_x1(*input_bias_desc);
                    mul->set_input_x2_by_name(*(const_beta_node->getOp()), "y");
                    mul->update_input_desc_x2(*(const_beta_node->getTensorDesc()));
                    mul->update_output_desc_y(*input_bias_desc);
                    input_bias_node = mul;
                }
                op->set_input_bias_by_name(*input_bias_node, "y");
                op->update_input_desc_bias(*input_bias_desc);
            }
        } else { // constant input B
            auto B = weights;
            auto const_B_node = std::make_shared<CannConstOp>(B.data, B.type(), shape(B), cv::format("%s_B", name.c_str()));
            op->set_input_x2_by_name(*(const_B_node->getOp()), "y");
            op->update_input_desc_x2(*(const_B_node->getTensorDesc()));
            if ((inputs.size() + numConsts()) >= 3) { // does not support broadcast bias
                Mat bias_mat;
                bias.convertTo(bias_mat, -1, beta);  // BatchMatMulV2 adds the bias unscaled
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
    Mat scaledBroadcastBias() const {
        if (beta == 1.f)
            return broadcast_bias;
        Mat scaled;
        broadcast_bias.convertTo(scaled, -1, beta);
        return scaled;
    }

    int real_ndims_C;

    Mat weights;                 //!< constant B; released once packed, ownsWeights survives
    Mat origWeights;             //!< plain B held for initCUDA(); getParam() must not see it
    Mat bias;                    //!< constant C
    bool ownsWeights = false;
    bool ownsBias = false;
    std::vector<float> packed_input_B;
    std::vector<float> thin_packed_B;
    Mat broadcast_bias;

    const uchar* last_packed_input_B_data = nullptr;
    MatShape wshape0;

    FastGemmOpt opt;
    MatMulHelper helper;
};

Ptr<MatMulLayer> MatMulLayer::create(const LayerParams& params)
{
    return makePtr<MatMulLayerImpl>(params);
}

}} // cv::dnn
