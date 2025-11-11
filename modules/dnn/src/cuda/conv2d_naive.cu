#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"
#include <vector>
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

namespace cv { namespace dnn { namespace cuda {

void conv2dNCHW(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t xDesc,
    cudnnFilterDescriptor_t wDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t yDesc,
    cudnnTensorDescriptor_t bDesc,
    const void* d_input,
    const void* d_weights,
    const void* d_bias,
    void* d_output)
{
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    const void* workspace = nullptr;
    size_t workspace_size = 0;

    detail::CudnnScalar alpha{}, beta0{};
    detail::setCudnnScalars(yDesc, alpha, beta0, 1.0f, 0.0f);
    cudnnStatus_t st = cudnnConvolutionForward(
        handle, detail::scalarPtr(alpha),
        xDesc, d_input,
        wDesc, d_weights,
        convDesc, algo,
        const_cast<void*>(workspace), workspace_size,
        detail::scalarPtr(beta0),
        yDesc, d_output);
    if (st != CUDNN_STATUS_SUCCESS) {
        return;
    }
    if (d_bias && bDesc) {
        detail::CudnnScalar alpha_b{}, beta_b{};
        detail::setCudnnScalars(yDesc, alpha_b, beta_b, 1.0f, 1.0f);
        cudnnAddTensor(handle, detail::scalarPtr(alpha_b), bDesc, d_bias, detail::scalarPtr(beta_b), yDesc, d_output);
    }
}

}}}
