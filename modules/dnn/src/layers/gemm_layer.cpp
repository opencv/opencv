// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "Accelerate/Accelerate.h"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

class GemmLayerImpl CV_FINAL : public GemmLayer {
public:
    GemmLayerImpl(const LayerParams& params) {
        setParamsFrom(params);

        transA = params.get<bool>("transA", false);
        transB = params.get<bool>("transB", false);
        alpha = params.get<float>("alpha", 1.0f);
        beta = params.get<float>("beta", 1.0f);

        is_C_1d = params.get<bool>("is_C_1d", false);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        size_t num_inputs = inputs.size();
        CV_CheckGE(num_inputs, static_cast<size_t>(2), "DNN/Gemm: Gemm takes at least two inputs");
        CV_CheckLE(num_inputs, static_cast<size_t>(3), "DNN/Gemm: Gemm takes at most three inputs");

        // Check whether A and B are two dimensional
        const auto shape_A = inputs[0], shape_B = inputs[1];
        CV_CheckEQ(shape_A.size(), static_cast<size_t>(2), "DNN/Gemm: Input A of Gemm should be two dimensional");
        CV_CheckEQ(shape_B.size(), static_cast<size_t>(2), "DNN/Gemm: Input B of Gemm should be two dimensional");

        // Check legal matrix multiplication
        int M_, N_, K_A, K_B;
        if (transA) {
            K_A = shape_A[0];
            M_ = shape_A[1];
        } else {
            M_ = shape_A[0];
            K_A = shape_A[1];
        }
        if (transB) {
            N_ = shape_B[0];
            K_B = shape_B[1];
        } else {
            K_B = shape_B[0];
            N_ = shape_B[1];
        }
        CV_CheckEQ(K_A, K_B, "DNN/Gemm: Invalid dimension of dim K");

        // TODO: Check whether C can be unidirectional broadcast to (M, N). Handle carefully with 1D Mat.
        if (inputs.size() == static_cast<size_t>(3)) {
            const auto shape_C = inputs[2];
            CV_CheckEQ(shape_C.size(), static_cast<size_t>(2), "DNN/Gemm: C must be two-dimentional");

            if (is_C_1d) {
                CV_Check(shape_C[0], shape_C[0] == 1 || shape_C[0] == M_ || shape_C[0] == N_, "DNN/Gemm: 1d C cannot be broadcast");
            } else {
                CV_Check(shape_C[0], shape_C[0] == 1 || shape_C[0] == M_, "DNN/Gemm: 2d C cannot be broadcast");
                CV_Check(shape_C[1], shape_C[1] == 1 || shape_C[1] == N_, "DNN/Gemm: 2d C cannot be broadcast");
            }
        }

        MatShape shape_y{M_, N_};
        outputs.assign(1, shape_y);
        return false;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const auto& A = inputs[0], B = inputs[1];
        const auto shape_A = shape(A), shape_B = shape(B);
        if (transA) {
            K = shape_A[0];
            M = shape_A[1];
        } else {
            M = shape_A[0];
            K = shape_A[1];
        }
        if (transB) {
            N = shape_B[0];
        } else {
            N = shape_B[1];
        }
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

        const auto& A = inputs[0], B = inputs[1];
        auto& Y = outputs[0];
        if (inputs.size() == 3) { // with bias
            const auto& C = inputs[2];

            // broadcast C to (M, N) if it is of shape (1), (N), (M, 1)
            auto shape_C = shape(C);
            CV_CheckEQ(shape_C.size(), (size_t)2, "DNN/Gemm: C must be two-dimentional");
            auto C_broadcast = C;
            if (shape_C[0] == N && is_C_1d) {
                // (N, 1) -> (1, N)
                cv::transpose(C_broadcast, C_broadcast);
            }
            if (shape_C[0] != M) {
                C_broadcast = cv::repeat(C_broadcast, M, 1);
            }
            if (shape_C[1] != N) {
                C_broadcast = cv::repeat(C_broadcast, 1, N);
            }

            // std::cout << "C_broadcast = " << C_broadcast << std::endl;
            // for (int i = 0; i < C_broadcast.total(); ++i) {
            //     std::cout << C_broadcast.at<float>(i) << " ";
            // }
            // std::cout << std::endl;

            // copy broadcast bias to output
            std::memcpy(Y.ptr<float>(), C_broadcast.ptr<float>(), M * N * sizeof(float));
            // std::cout << "Y = " << Y << std::endl;
        }

        // std::cout << "M = " << M << ", N = " << N << ", K = " << K << ", alpha = " << alpha << ", beta = " << beta << std::endl;
        // cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha, const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc)
        if (transA) {
            if (transB) {
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, M, N, K, alpha, A.ptr<float>(), M, B.ptr<float>(), K, beta, Y.ptr<float>(), N);
            } else {
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A.ptr<float>(), M, B.ptr<float>(), N, beta, Y.ptr<float>(), N);
            }
        } else {
            if (transB) {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A.ptr<float>(), K, B.ptr<float>(), K, beta, Y.ptr<float>(), N);
            } else {
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A.ptr<float>(), K, B.ptr<float>(), N, beta, Y.ptr<float>(), N);
            }
        }

    }

private:
    int M;
    int N;
    int K;

    bool is_C_1d;
};

Ptr<GemmLayer> GemmLayer::create(const LayerParams& params) {
    return makePtr<GemmLayerImpl>(params);
}

}} // namespace cv::dnn
