// cuDNN/cuBLAS-backed Fully Connected (GEMM) for FP32
// y [N x M] = x [N x K] * W^T [K x M] + b [M]

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cstdio>
#include "conv_naive.hpp"

namespace cv { namespace dnn { namespace cuda_naive_conv {

static inline const char* cudnn_err(cudnnStatus_t s) { return cudnnGetErrorString(s); }

void fc_fp32(const float* x, const float* w, const float* b, float* y,
             int N, int K, int M)
{
    // Diagnostics
    std::fprintf(stderr, "DNN(cuDNN): fc_fp32 using cuBLAS GEMM (N=%d K=%d -> M=%d)\n", N, K, M);

    cublasHandle_t blas = nullptr;
    cudnnHandle_t cudnn = nullptr;
    cudnnTensorDescriptor_t yDesc = nullptr, bDesc = nullptr;

    bool ok = true;
    cublasStatus_t cbst = CUBLAS_STATUS_SUCCESS;
    cudnnStatus_t  cdst = CUDNN_STATUS_SUCCESS;

    // cuBLAS for GEMM
    cbst = cublasCreate(&blas);
    if (cbst != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "DNN(cuDNN): fc cublasCreate failed: %d\n", (int)cbst);
        ok = false;
    }

    // Row-major trick with cuBLAS (which is column-major):
    // Compute Y_row(NxM) = X_row(NxK) * W_row^T(KxM)
    // by calling: C_col(MxN) = op(A)=B_row^T(KxN) * op(B)=A_row^T(KxM)
    // cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, ...)
    if (ok) {
        const float alpha = 1.0f, beta = 0.0f;
        cbst = cublasSgemm(blas,
                           CUBLAS_OP_T, CUBLAS_OP_T,
                           /*m*/ M, /*n*/ N, /*k*/ K,
                           &alpha,
                           /*A = B_row (W)*/ w, /*lda = K*/ K,
                           /*B = A_row (X)*/ x, /*ldb = K*/ K,
                           &beta,
                           /*C = Y_row*/ y, /*ldc = M*/ M);
        if (cbst != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "DNN(cuDNN): fc GEMM failed: %d\n", (int)cbst);
            ok = false;
        }
    }

    if (ok && b != nullptr) {
        cdst = cudnnCreate(&cudnn);
        if (cdst != CUDNN_STATUS_SUCCESS) {
            std::fprintf(stderr, "DNN(cuDNN): fc cudnnCreate failed: %s\n", cudnn_err(cdst));
            ok = false;
        }
    }

    if (ok && b != nullptr) {
        cdst = cudnnCreateTensorDescriptor(&yDesc);
        if (cdst != CUDNN_STATUS_SUCCESS) {
            std::fprintf(stderr, "DNN(cuDNN): fc create yDesc failed: %s\n", cudnn_err(cdst));
            ok = false;
        }
    }
    if (ok && b != nullptr) {
        cdst = cudnnCreateTensorDescriptor(&bDesc);
        if (cdst != CUDNN_STATUS_SUCCESS) {
            std::fprintf(stderr, "DNN(cuDNN): fc create bDesc failed: %s\n", cudnn_err(cdst));
            ok = false;
        }
    }

    if (ok && b != nullptr) {
        // NCHW with H=W=1
        const int n = N, c = M, h = 1, wv = 1;
        const int ysN = c * h * wv;
        const int ysC = h * wv;
        const int ysH = wv;
        const int ysW = 1;
        cdst = cudnnSetTensor4dDescriptorEx(yDesc, CUDNN_DATA_FLOAT, n, c, h, wv, ysN, ysC, ysH, ysW);
        if (cdst != CUDNN_STATUS_SUCCESS) {
            std::fprintf(stderr, "DNN(cuDNN): fc set yDesc failed: %s\n", cudnn_err(cdst));
            ok = false;
        }
    }
    if (ok && b != nullptr) {
        // Bias has shape (1, M, 1, 1)
        const int n = 1, c = M, h = 1, wv = 1;
        const int bsN = c * h * wv;
        const int bsC = h * wv;
        const int bsH = wv;
        const int bsW = 1;
        cdst = cudnnSetTensor4dDescriptorEx(bDesc, CUDNN_DATA_FLOAT, n, c, h, wv, bsN, bsC, bsH, bsW);
        if (cdst != CUDNN_STATUS_SUCCESS) {
            std::fprintf(stderr, "DNN(cuDNN): fc set bDesc failed: %s\n", cudnn_err(cdst));
            ok = false;
        }
    }

    if (ok && b != nullptr) {
        const float alpha = 1.0f, beta = 1.0f; // Y = 1*B + 1*Y
        cdst = cudnnAddTensor(cudnn, &alpha, bDesc, b, &beta, yDesc, y);
        if (cdst != CUDNN_STATUS_SUCCESS) {
            std::fprintf(stderr, "DNN(cuDNN): fc add bias failed: %s\n", cudnn_err(cdst));
            ok = false;
        }
    }

    if (yDesc) cudnnDestroyTensorDescriptor(yDesc);
    if (bDesc) cudnnDestroyTensorDescriptor(bDesc);
    if (cudnn) cudnnDestroy(cudnn);
    if (blas)  cublasDestroy(blas);
}

}}} // namespace cv::dnn::cuda_naive_conv
