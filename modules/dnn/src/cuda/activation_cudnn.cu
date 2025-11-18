#include <cuda_runtime.h>
#include <float.h>
#include "layer_cudnn.hpp"
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace cv { namespace dnn { namespace cuda {

void relu(cudnnHandle_t handle,
               cudnnActivationDescriptor_t actDesc,
               cudnnTensorDescriptor_t xDesc,
               cudnnTensorDescriptor_t yDesc,
               const void* d_input,
               void* d_output)
{
    detail::CudnnScalar alpha{}, beta{};
    detail::setCudnnScalars(yDesc, alpha, beta, 1.0f, 0.0f);
    cudnnActivationForward(handle, actDesc,
                           detail::scalarPtr(alpha), xDesc, d_input,
                           detail::scalarPtr(beta), yDesc, d_output);
}

}}} // namespace cv::dnn::cuda
