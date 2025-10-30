#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"
#include <cudnn.h>
#include <cstdio>

namespace cv { namespace dnn { namespace cuda_naive_conv {

void relu_fp32(const float* d_input, float* d_output, size_t count)
{
    {
        if (count == 0) { return; }
        cudnnHandle_t handle = nullptr;
        cudnnTensorDescriptor_t xDesc = nullptr, yDesc = nullptr;
        cudnnActivationDescriptor_t actDesc = nullptr;
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS; bool ok = true;
        if ((st = cudnnCreate(&handle)) != CUDNN_STATUS_SUCCESS) return;
        cudnnSetStream(handle, 0);
        if (ok && (st = cudnnCreateTensorDescriptor(&xDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreateTensorDescriptor(&yDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreateActivationDescriptor(&actDesc)) != CUDNN_STATUS_SUCCESS) ok = false;

        int n = 1, c = (int)count, h = 1, w = 1;
        int stride_w = 1, stride_h = 1, stride_c = 1, stride_n = c;
        if (ok && (st = cudnnSetTensor4dDescriptorEx(xDesc, CUDNN_DATA_FLOAT, n, c, h, w,
                                          stride_n, stride_c, stride_h, stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnSetTensor4dDescriptorEx(yDesc, CUDNN_DATA_FLOAT, n, c, h, w,
                                          stride_n, stride_c, stride_h, stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0)) != CUDNN_STATUS_SUCCESS) ok = false;

        const float alpha = 1.0f, beta = 0.0f;
        if (ok && (st = cudnnActivationForward(handle, actDesc,
                                   &alpha, xDesc, d_input,
                                   &beta, yDesc, d_output)) != CUDNN_STATUS_SUCCESS) ok = false;

        if (ok) std::fprintf(stderr, "DNN(cuDNN): relu_fp32 using cuDNN (count=%zu)\n", count);
        else std::fprintf(stderr, "DNN(cuDNN): relu_fp32 cuDNN error: %s\n", cudnnGetErrorString(st));

        if (actDesc) cudnnDestroyActivationDescriptor(actDesc);
        if (yDesc) cudnnDestroyTensorDescriptor(yDesc);
        if (xDesc) cudnnDestroyTensorDescriptor(xDesc);
        if (handle) cudnnDestroy(handle);
        return;
    }
}

// removed kernel; cuDNN-only implementation

void relu_fp32_2d(const float* d_input, size_t input_step, float* d_output, size_t output_step, int rows, int cols)
{
    {
        cudnnHandle_t handle = nullptr;
        cudnnTensorDescriptor_t xDesc = nullptr, yDesc = nullptr;
        cudnnActivationDescriptor_t actDesc = nullptr;
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS; bool ok = true;
        if ((st = cudnnCreate(&handle)) != CUDNN_STATUS_SUCCESS) return;
        cudnnSetStream(handle, 0);
        if (ok && (st = cudnnCreateTensorDescriptor(&xDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreateTensorDescriptor(&yDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreateActivationDescriptor(&actDesc)) != CUDNN_STATUS_SUCCESS) ok = false;

        int n = rows, c = cols, h = 1, w = 1;
        int x_stride_w = 1;
        int x_stride_h = 1;
        int x_stride_c = 1;
        int x_stride_n = (int)(input_step / sizeof(float));
        if (ok && (st = cudnnSetTensor4dDescriptorEx(xDesc, CUDNN_DATA_FLOAT, n, c, h, w,
                                          x_stride_n, x_stride_c, x_stride_h, x_stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;

        int y_stride_w = 1;
        int y_stride_h = 1;
        int y_stride_c = 1;
        int y_stride_n = (int)(output_step / sizeof(float));
        if (ok && (st = cudnnSetTensor4dDescriptorEx(yDesc, CUDNN_DATA_FLOAT, n, c, h, w,
                                          y_stride_n, y_stride_c, y_stride_h, y_stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;

        if (ok && (st = cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0)) != CUDNN_STATUS_SUCCESS) ok = false;

        const float alpha = 1.0f, beta = 0.0f;
        if (ok && (st = cudnnActivationForward(handle, actDesc,
                                   &alpha, xDesc, d_input,
                                   &beta, yDesc, d_output)) != CUDNN_STATUS_SUCCESS) ok = false;

        if (ok) std::fprintf(stderr, "DNN(cuDNN): relu_fp32_2d using cuDNN (rows=%d cols=%d)\n", rows, cols);
        else std::fprintf(stderr, "DNN(cuDNN): relu_fp32_2d cuDNN error: %s\n", cudnnGetErrorString(st));

        if (actDesc) cudnnDestroyActivationDescriptor(actDesc);
        if (yDesc) cudnnDestroyTensorDescriptor(yDesc);
        if (xDesc) cudnnDestroyTensorDescriptor(xDesc);
        if (handle) cudnnDestroy(handle);
        return;
    }
}

}}} // namespace cv::dnn::cuda_naive_conv
