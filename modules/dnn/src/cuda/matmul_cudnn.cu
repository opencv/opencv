// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include "layer_cudnn.hpp"

namespace cv { namespace dnn { namespace cuda {

static inline const char* cudnn_err(cudnnStatus_t s) { return cudnnGetErrorString(s); }

void matMul(cublasHandle_t blas,
            cudnnHandle_t cudnn,
            cudnnTensorDescriptor_t yDesc,
            cudnnTensorDescriptor_t bDesc,
            const cv::cuda::GpuMat& x,
            const cv::cuda::GpuMat& w,
            cv::cuda::GpuMat& y,
            const cv::cuda::GpuMat& b)
{
    // Shapes: x [N x K], w [M x K] (row-major), y [N x M]
    const int N = x.rows;
    const int K = x.cols;
    const int M = w.rows;

    bool ok = true;
    cublasStatus_t cbst = CUBLAS_STATUS_SUCCESS;
    cudnnStatus_t  cdst = CUDNN_STATUS_SUCCESS;

    // Basic validation
    if (ok) {
        int xtype = x.type(), wtype = w.type(), ytype = y.type();
        if (!b.empty() && b.type() != ytype) {
            std::fprintf(stderr, "DNN(cuDNN): matmul bias/output type mismatch\n"); ok = false;
        }
        if (!((xtype == CV_32F || xtype == CV_16F || xtype == CV_16BF) &&
              (wtype == CV_32F || wtype == CV_16F || wtype == CV_16BF) &&
              (ytype == CV_32F || ytype == CV_16F || ytype == CV_16BF))) {
            std::fprintf(stderr, "DNN(cuDNN): matmul expects CV_32F/CV_16F/CV_16BF mats\n"); ok = false;
        }
    }
    if (ok) {
        if (w.cols != K) {
            std::fprintf(stderr, "DNN(cuDNN): matmul shape mismatch (w.cols=%d, K=%d)\n", w.cols, K);
            ok = false;
        }
        if (y.rows != N || y.cols != M) {
            std::fprintf(stderr, "DNN(cuDNN): matmul output shape mismatch, got (%d x %d), expected (%d x %d)\n",
                         y.rows, y.cols, N, M);
            ok = false;
        }
        if (!x.isContinuous() || !w.isContinuous() || !y.isContinuous() || (!b.empty() && !b.isContinuous())) {
            std::fprintf(stderr, "DNN(cuDNN): matmul expects continuous GpuMat buffers\n");
            ok = false;
        }
    }

    // Row-major trick with cuBLAS (column-major lib)
    if (ok) {
        float alpha = 1.0f, beta = 0.0f;
        const void* wPtr = (const void*)w.ptr();
        const void* xPtr = (const void*)x.ptr();
        void* yPtr = (void*)y.ptr();
        const int lda = static_cast<int>(w.step / (size_t)CV_ELEM_SIZE(w.type()));
        const int ldb = static_cast<int>(x.step / (size_t)CV_ELEM_SIZE(x.type()));
        const int ldc = static_cast<int>(y.step / (size_t)CV_ELEM_SIZE(y.type()));
        cudaDataType wType = (w.type() == CV_32F ? CUDA_R_32F : (w.type() == CV_16F ? CUDA_R_16F : CUDA_R_16BF));
        cudaDataType xType = (x.type() == CV_32F ? CUDA_R_32F : (x.type() == CV_16F ? CUDA_R_16F : CUDA_R_16BF));
        cudaDataType yType = (y.type() == CV_32F ? CUDA_R_32F : (y.type() == CV_16F ? CUDA_R_16F : CUDA_R_16BF));
        cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
        cbst = cublasGemmEx(blas,
                            CUBLAS_OP_T, CUBLAS_OP_T,
                            /*m*/ M, /*n*/ N, /*k*/ K,
                            &alpha,
                            /*A = W*/ wPtr, wType, /*lda*/ lda,
                            /*B = X*/ xPtr, xType, /*ldb*/ ldb,
                            &beta,
                            /*C = Y*/ yPtr, yType, /*ldc*/ ldc,
                            computeType, CUBLAS_GEMM_DEFAULT);
        if (cbst != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "DNN(cuDNN): fc GEMM failed: %d\n", (int)cbst);
            ok = false;
        }
    }

    if (ok && !b.empty() && yDesc && bDesc) {
        // Typed alpha/beta for cudnnAddTensor
        int n, c, h, w, ns, cs, hs, ws; cudnnDataType_t ydt;
        cudnnGetTensor4dDescriptor(yDesc, &ydt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
        switch (ydt) {
            case CUDNN_DATA_FLOAT: {
                const float alpha = 1.0f, beta = 1.0f;
                cdst = cudnnAddTensor(cudnn, &alpha, bDesc, b.ptr<float>(), &beta, yDesc, y.ptr<float>());
                break;
            }
            case CUDNN_DATA_HALF: {
                __half alpha = __float2half(1.0f), beta = __float2half(1.0f);
                cdst = cudnnAddTensor(cudnn, &alpha, bDesc, b.ptr<__half>(), &beta, yDesc, y.ptr<__half>());
                break;
            }
            case CUDNN_DATA_BFLOAT16: {
                __nv_bfloat16 alpha = __float2bfloat16(1.0f), beta = __float2bfloat16(1.0f);
                cdst = cudnnAddTensor(cudnn, &alpha, bDesc, b.ptr<__nv_bfloat16>(), &beta, yDesc, y.ptr<__nv_bfloat16>());
                break;
            }
            default: {
                const float alpha = 1.0f, beta = 1.0f;
                cdst = cudnnAddTensor(cudnn, &alpha, bDesc, b.ptr<float>(), &beta, yDesc, y.ptr<float>());
                break;
            }
        }
        if (cdst != CUDNN_STATUS_SUCCESS) {
            std::fprintf(stderr, "DNN(cuDNN): fc add bias failed: %s\n", cudnn_err(cdst));
            ok = false;
        }
    }
}

}}} // namespace cv::dnn::cuda
