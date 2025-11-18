// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <float.h>
#include "layer_cudnn.hpp"
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
        cudnnAddTensor(handle, detail::scalarPtr(alpha_b), bDesc, d_bias,
                       detail::scalarPtr(beta_b), yDesc, d_output);
    }
}

// 1D convolution in NCHW “disguised” as 2D:
// e.g. input shape: [N, C, 1, L], kernel: [K, C, 1, kW]
void conv1dNCHW(
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
        cudnnAddTensor(handle, detail::scalarPtr(alpha_b), bDesc, d_bias,
                       detail::scalarPtr(beta_b), yDesc, d_output);
    }
}

// 3D convolution, typical layout NCDHW (5D tensor descriptor)
void conv3dNCDHW(
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
        cudnnAddTensor(handle, detail::scalarPtr(alpha_b), bDesc, d_bias,
                       detail::scalarPtr(beta_b), yDesc, d_output);
    }
}

}}}
