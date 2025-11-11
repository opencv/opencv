#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace cv { namespace dnn { namespace cuda {

void add2(
    cudnnHandle_t handle,
    cudnnOpTensorDescriptor_t opDesc,
    cudnnTensorDescriptor_t aDesc,
    cudnnTensorDescriptor_t bDesc,
    cudnnTensorDescriptor_t yDesc,
    const void* d_a,
    const void* d_b,
    void* d_y)
{
    detail::CudnnScalar alpha1{}, alpha2{}, beta{};
    detail::setCudnnScalars(yDesc, alpha1, beta, 1.0f, 0.0f);
    detail::setCudnnScalars(yDesc, alpha2, beta, 1.0f, 0.0f);
    cudnnOpTensor(handle, opDesc,
                  detail::scalarPtr(alpha1), aDesc, d_a,
                  detail::scalarPtr(alpha2), bDesc, d_b,
                  detail::scalarPtr(beta), yDesc, d_y);
}

void addInplace(
    cudnnHandle_t handle,
    cudnnOpTensorDescriptor_t opDesc,
    cudnnTensorDescriptor_t xDesc,
    cudnnTensorDescriptor_t yDesc,
    const void* d_x,
    void* d_y)
{
    detail::CudnnScalar alpha1{}, alpha2{}, beta{};
    detail::setCudnnScalars(yDesc, alpha1, beta, 1.0f, 1.0f);
    detail::setCudnnScalars(yDesc, alpha2, beta, 1.0f, 1.0f);
    cudnnOpTensor(handle, opDesc,
                  detail::scalarPtr(alpha1), yDesc, d_y,
                  detail::scalarPtr(alpha2), xDesc, d_x,
                  detail::scalarPtr(beta), yDesc, d_y);
}

}}} // namespace cv::dnn::cuda
