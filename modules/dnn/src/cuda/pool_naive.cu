#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace cv { namespace dnn { namespace cuda {

void avgPool2dNCHW(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc,
    cudnnTensorDescriptor_t yDesc,
    cudnnPoolingDescriptor_t pDesc,
    const void* d_input,
    void* d_output)
{
    {
        detail::CudnnScalar alpha{}, beta{};
        detail::setCudnnScalars(yDesc, alpha, beta, 1.0f, 0.0f);
        cudnnPoolingForward(handle, pDesc,
                            detail::scalarPtr(alpha), xDesc, d_input,
                            detail::scalarPtr(beta), yDesc, d_output);
        return;
    }
}

void maxPool2dNCHW(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc,
    cudnnTensorDescriptor_t yDesc,
    cudnnPoolingDescriptor_t pDesc,
    const void* d_input,
    void* d_output)
{
    {
        detail::CudnnScalar alpha{}, beta{};
        detail::setCudnnScalars(yDesc, alpha, beta, 1.0f, 0.0f);
        cudnnPoolingForward(handle, pDesc,
                            detail::scalarPtr(alpha), xDesc, d_input,
                            detail::scalarPtr(beta), yDesc, d_output);
        return;
    }
}

}}} // namespace cv::dnn::cuda
