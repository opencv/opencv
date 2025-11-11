#pragma once
#include <cudnn.h>
#include <cublas_v2.h>
#include <opencv2/core/cuda.hpp>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace cv { namespace dnn { namespace cuda {

namespace detail {
struct CudnnScalar {
    cudnnDataType_t dtype;
    float f;
    __half h;
    __nv_bfloat16 bf16;
    double d;
};
inline void setCudnnScalars(cudnnTensorDescriptor_t refDesc, CudnnScalar& a, CudnnScalar& b, float av, float bv)
{
    int n, c, h, w, ns, cs, hs, ws;
    cudnnGetTensor4dDescriptor(refDesc, &a.dtype, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
    b.dtype = a.dtype;
    switch (a.dtype) {
        case CUDNN_DATA_FLOAT:
            a.f = av; b.f = bv; break;
        case CUDNN_DATA_HALF:
            a.h = __float2half(av); b.h = __float2half(bv); break;
        case CUDNN_DATA_BFLOAT16:
            a.bf16 = __float2bfloat16(av); b.bf16 = __float2bfloat16(bv); break;
        case CUDNN_DATA_DOUBLE:
            a.d = (double)av; b.d = (double)bv; break;
        default:
            a.f = av; b.f = bv; a.dtype = CUDNN_DATA_FLOAT; b.dtype = CUDNN_DATA_FLOAT; break;
    }
}
inline const void* scalarPtr(const CudnnScalar& s)
{
    switch (s.dtype) {
        case CUDNN_DATA_FLOAT: return (const void*)&s.f;
        case CUDNN_DATA_HALF: return (const void*)&s.h;
        case CUDNN_DATA_BFLOAT16: return (const void*)&s.bf16;
        case CUDNN_DATA_DOUBLE: return (const void*)&s.d;
        default: return (const void*)&s.f;
    }
}
} // namespace detail

void conv2dNCHW(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc,
    cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t yDesc,
    cudnnTensorDescriptor_t bDesc, // may be nullptr if no bias
    const void* d_input,
    const void* d_weights,
    const void* d_bias,
    void* d_output);

void relu(cudnnHandle_t handle,
          cudnnActivationDescriptor_t actDesc,
          cudnnTensorDescriptor_t xDesc,
          cudnnTensorDescriptor_t yDesc,
          const void* d_input,
          void* d_output);

void add2(cudnnHandle_t handle,
          cudnnOpTensorDescriptor_t opDesc,
          cudnnTensorDescriptor_t aDesc,
          cudnnTensorDescriptor_t bDesc,
          cudnnTensorDescriptor_t yDesc,
          const void* d_a,
          const void* d_b,
          void* d_y);

void addInplace(cudnnHandle_t handle,
                cudnnOpTensorDescriptor_t opDesc,
                cudnnTensorDescriptor_t xDesc,
                cudnnTensorDescriptor_t yDesc,
                const void* d_x,
                void* d_y);

void avgPool2dNCHW(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc,
    cudnnTensorDescriptor_t yDesc,
    cudnnPoolingDescriptor_t pDesc,
    const void* d_input,
    void* d_output);

void maxPool2dNCHW(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc,
    cudnnTensorDescriptor_t yDesc,
    cudnnPoolingDescriptor_t pDesc,
    const void* d_input,
    void* d_output);

void matMul(cublasHandle_t blas,
            cudnnHandle_t cudnn,
            cudnnTensorDescriptor_t yDesc,
            cudnnTensorDescriptor_t bDesc,
            const cv::cuda::GpuMat& x,
            const cv::cuda::GpuMat& w,
            cv::cuda::GpuMat& y,
            const cv::cuda::GpuMat& b);

}}} // namespace cv::dnn::cuda
