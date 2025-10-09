#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"
#include <cudnn.h>
#include <cstdio>

namespace cv { namespace dnn { namespace cuda_naive_conv {

void add2_fp32_2d(
    const float* d_a, size_t a_step,
    const float* d_b, size_t b_step,
    float* d_y, size_t y_step,
    int rows, int cols)
{
    {
        cudnnHandle_t handle = nullptr;
        cudnnTensorDescriptor_t aDesc = nullptr, bDesc = nullptr, yDesc = nullptr;
        cudnnOpTensorDescriptor_t opDesc = nullptr;
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS; bool ok = true;
        if ((st = cudnnCreate(&handle)) != CUDNN_STATUS_SUCCESS) return;
        cudnnSetStream(handle, 0);
        if (ok && (st = cudnnCreateTensorDescriptor(&aDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreateTensorDescriptor(&bDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreateTensorDescriptor(&yDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreateOpTensorDescriptor(&opDesc)) != CUDNN_STATUS_SUCCESS) ok = false;

        int n = rows, c = 1, h = 1, w = cols;
        int a_stride_w = 1;
        int a_stride_h = w;
        int a_stride_c = h * w;
        int a_stride_n = (int)(a_step / sizeof(float));
        int b_stride_w = 1;
        int b_stride_h = w;
        int b_stride_c = h * w;
        int b_stride_n = (int)(b_step / sizeof(float));
        int y_stride_w = 1;
        int y_stride_h = w;
        int y_stride_c = h * w;
        int y_stride_n = (int)(y_step / sizeof(float));
        if (ok && (st = cudnnSetTensor4dDescriptorEx(aDesc, CUDNN_DATA_FLOAT, n, c, h, w,
                                          a_stride_n, a_stride_c, a_stride_h, a_stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnSetTensor4dDescriptorEx(bDesc, CUDNN_DATA_FLOAT, n, c, h, w,
                                          b_stride_n, b_stride_c, b_stride_h, b_stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnSetTensor4dDescriptorEx(yDesc, CUDNN_DATA_FLOAT, n, c, h, w,
                                          y_stride_n, y_stride_c, y_stride_h, y_stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;

        if (ok && (st = cudnnSetOpTensorDescriptor(opDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN)) != CUDNN_STATUS_SUCCESS) ok = false;

        const float alpha1 = 1.0f, alpha2 = 1.0f, beta = 0.0f;
        if (ok && (st = cudnnOpTensor(handle, opDesc,
                          &alpha1, aDesc, d_a,
                          &alpha2, bDesc, d_b,
                          &beta, yDesc, d_y)) != CUDNN_STATUS_SUCCESS) ok = false;

        if (ok) std::fprintf(stderr, "DNN(cuDNN): add2_fp32_2d using cuDNN (rows=%d cols=%d)\n", rows, cols);
        else std::fprintf(stderr, "DNN(cuDNN): add2_fp32_2d cuDNN error: %s\n", cudnnGetErrorString(st));

        if (opDesc) cudnnDestroyOpTensorDescriptor(opDesc);
        if (yDesc) cudnnDestroyTensorDescriptor(yDesc);
        if (bDesc) cudnnDestroyTensorDescriptor(bDesc);
        if (aDesc) cudnnDestroyTensorDescriptor(aDesc);
        if (handle) cudnnDestroy(handle);
        return;
    }
}

static __global__ void add_inplace_fp32_2d_kernel(
    const float* __restrict__ x, size_t x_step,
    float* __restrict__ y, size_t y_step,
    int rows, int cols)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        const float* xr = (const float*)((const unsigned char*)x + r * x_step);
        float* yr = (float*)((unsigned char*)y + r * y_step);
        yr[c] = yr[c] + xr[c];
    }
}

void add_inplace_fp32_2d(
    const float* d_x, size_t x_step,
    float* d_y, size_t y_step,
    int rows, int cols)
{
    {
        cudnnHandle_t handle = nullptr;
        cudnnTensorDescriptor_t xDesc = nullptr, yDesc = nullptr;
        cudnnOpTensorDescriptor_t opDesc = nullptr;
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS; bool ok = true;
        if ((st = cudnnCreate(&handle)) != CUDNN_STATUS_SUCCESS) return;
        cudnnSetStream(handle, 0);
        if (ok && (st = cudnnCreateTensorDescriptor(&xDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreateTensorDescriptor(&yDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreateOpTensorDescriptor(&opDesc)) != CUDNN_STATUS_SUCCESS) ok = false;

        int n = rows, c = 1, h = 1, w = cols;
        int x_stride_w = 1;
        int x_stride_h = w;
        int x_stride_c = h * w;
        int x_stride_n = (int)(x_step / sizeof(float));
        int y_stride_w = 1;
        int y_stride_h = w;
        int y_stride_c = h * w;
        int y_stride_n = (int)(y_step / sizeof(float));
        if (ok && (st = cudnnSetTensor4dDescriptorEx(xDesc, CUDNN_DATA_FLOAT, n, c, h, w,
                                          x_stride_n, x_stride_c, x_stride_h, x_stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnSetTensor4dDescriptorEx(yDesc, CUDNN_DATA_FLOAT, n, c, h, w,
                                          y_stride_n, y_stride_c, y_stride_h, y_stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnSetOpTensorDescriptor(opDesc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN)) != CUDNN_STATUS_SUCCESS) ok = false;

        const float alpha1 = 1.0f, alpha2 = 1.0f, beta = 1.0f;
        if (ok && (st = cudnnOpTensor(handle, opDesc,
                          &alpha1, yDesc, d_y,
                          &alpha2, xDesc, d_x,
                          &beta, yDesc, d_y)) != CUDNN_STATUS_SUCCESS) ok = false;

        if (ok) std::fprintf(stderr, "DNN(cuDNN): add_inplace_fp32_2d using cuDNN (rows=%d cols=%d)\n", rows, cols);
        else std::fprintf(stderr, "DNN(cuDNN): add_inplace_fp32_2d cuDNN error: %s\n", cudnnGetErrorString(st));

        if (opDesc) cudnnDestroyOpTensorDescriptor(opDesc);
        if (yDesc) cudnnDestroyTensorDescriptor(yDesc);
        if (xDesc) cudnnDestroyTensorDescriptor(xDesc);
        if (handle) cudnnDestroy(handle);
        return;
    }
}

}}} // namespace cv::dnn::cuda_naive_conv
