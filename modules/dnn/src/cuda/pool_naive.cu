#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"
#include <cudnn.h>
#include <cstdio>

namespace cv { namespace dnn { namespace cuda_naive_conv {

void global_avgpool2d_nchw_flat_fp32(
    const float* d_input, size_t input_step_bytes,
    float* d_output, size_t output_step_bytes,
    int N, int C, int H, int W)
{
    {
        cudnnHandle_t handle = nullptr;
        cudnnTensorDescriptor_t xDesc = nullptr, yDesc = nullptr;
        cudnnPoolingDescriptor_t pDesc = nullptr;
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        bool ok = true;

        st = cudnnCreate(&handle); if (st != CUDNN_STATUS_SUCCESS) return;
        cudnnSetStream(handle, 0);
        if (ok && (st = cudnnCreateTensorDescriptor(&xDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreateTensorDescriptor(&yDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreatePoolingDescriptor(&pDesc)) != CUDNN_STATUS_SUCCESS) ok = false;

        int x_stride_w = 1;
        int x_stride_h = W;
        int x_stride_c = H * W;
        int x_stride_n = (int)(input_step_bytes / sizeof(float));
        if (ok && (st = cudnnSetTensor4dDescriptorEx(xDesc, CUDNN_DATA_FLOAT,
                                         N, C, H, W,
                                         x_stride_n, x_stride_c, x_stride_h, x_stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;

        // Output is N x C x 1 x 1
        int y_stride_w = 1;
        int y_stride_h = 1;
        int y_stride_c = 1;
        int y_stride_n = (int)(output_step_bytes / sizeof(float));
        if (ok && (st = cudnnSetTensor4dDescriptorEx(yDesc, CUDNN_DATA_FLOAT,
                                         N, C, 1, 1,
                                         y_stride_n, y_stride_c, y_stride_h, y_stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;

        if (ok && (st = cudnnSetPooling2dDescriptor(pDesc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_PROPAGATE_NAN,
                                        H, W, 0, 0, 1, 1)) != CUDNN_STATUS_SUCCESS) ok = false;

        const float alpha = 1.0f, beta = 0.0f;
        if (ok && (st = cudnnPoolingForward(handle, pDesc,
                                &alpha, xDesc, d_input,
                                &beta, yDesc, d_output)) != CUDNN_STATUS_SUCCESS) ok = false;

        if (ok) {
            std::fprintf(stderr, "DNN(cuDNN): global_avgpool2d using cuDNN (N=%d C=%d H=%d W=%d)\n", N, C, H, W);
        } else {
            std::fprintf(stderr, "DNN(cuDNN): global_avgpool2d cuDNN error: %s\n", cudnnGetErrorString(st));
        }

        if (pDesc) cudnnDestroyPoolingDescriptor(pDesc);
        if (yDesc) cudnnDestroyTensorDescriptor(yDesc);
        if (xDesc) cudnnDestroyTensorDescriptor(xDesc);
        if (handle) cudnnDestroy(handle);
        return;
    }
}

static __global__ void maxpool2d_nchw_flatrows_fp32_kernel(
    const float* __restrict__ x, size_t in_step,
    float* __restrict__ y, size_t out_step,
    int N, int C,
    int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW)
{
    int n = blockIdx.z; // batch
    int c = blockIdx.y; // channel
    int hw = blockIdx.x * blockDim.x + threadIdx.x; // flattened H_out*W_out
    if (n >= N || c >= C || hw >= H_out * W_out) return;
    int oh = hw / W_out;
    int ow = hw % W_out;

    float maxv = -FLT_MAX;
    for (int kh = 0; kh < kH; ++kh) {
        int ih = oh * sH - pH + kh;
        if (ih < 0 || ih >= H_in) continue;
        for (int kw = 0; kw < kW; ++kw) {
            int iw = ow * sW - pW + kw;
            if (iw < 0 || iw >= W_in) continue;
            size_t offs = ((size_t)c * H_in + (size_t)ih) * (size_t)W_in + (size_t)iw;
            const float* xn = (const float*)((const unsigned char*)x + (size_t)n * in_step);
            float v = xn[offs];
            maxv = v > maxv ? v : maxv;
        }
    }
    size_t ooffs = ((size_t)c * H_out + (size_t)oh) * (size_t)W_out + (size_t)ow;
    float* yn = (float*)((unsigned char*)y + (size_t)n * out_step);
    yn[ooffs] = maxv;
}

void maxpool2d_nchw_flatrows_fp32(
    const float* d_input, size_t input_step_bytes,
    float* d_output, size_t output_step_bytes,
    int N, int C,
    int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW)
{
    {
        cudnnHandle_t handle = nullptr;
        cudnnTensorDescriptor_t xDesc = nullptr, yDesc = nullptr;
        cudnnPoolingDescriptor_t pDesc = nullptr;
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        bool ok = true;

        st = cudnnCreate(&handle); if (st != CUDNN_STATUS_SUCCESS) return;
        cudnnSetStream(handle, 0);
        if (ok && (st = cudnnCreateTensorDescriptor(&xDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreateTensorDescriptor(&yDesc)) != CUDNN_STATUS_SUCCESS) ok = false;
        if (ok && (st = cudnnCreatePoolingDescriptor(&pDesc)) != CUDNN_STATUS_SUCCESS) ok = false;

        int x_stride_w = 1;
        int x_stride_h = W_in;
        int x_stride_c = H_in * W_in;
        int x_stride_n = (int)(input_step_bytes / sizeof(float));
        if (ok && (st = cudnnSetTensor4dDescriptorEx(xDesc, CUDNN_DATA_FLOAT,
                                         N, C, H_in, W_in,
                                         x_stride_n, x_stride_c, x_stride_h, x_stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;

        int y_stride_w = 1;
        int y_stride_h = W_out;
        int y_stride_c = H_out * W_out;
        int y_stride_n = (int)(output_step_bytes / sizeof(float));
        if (ok && (st = cudnnSetTensor4dDescriptorEx(yDesc, CUDNN_DATA_FLOAT,
                                         N, C, H_out, W_out,
                                         y_stride_n, y_stride_c, y_stride_h, y_stride_w)) != CUDNN_STATUS_SUCCESS) ok = false;

        if (ok && (st = cudnnSetPooling2dDescriptor(pDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
                                        kH, kW, pH, pW, sH, sW)) != CUDNN_STATUS_SUCCESS) ok = false;

        const float alpha = 1.0f, beta = 0.0f;
        if (ok && (st = cudnnPoolingForward(handle, pDesc,
                                &alpha, xDesc, d_input,
                                &beta, yDesc, d_output)) != CUDNN_STATUS_SUCCESS) ok = false;

        if (ok) {
            std::fprintf(stderr, "DNN(cuDNN): maxpool2d using cuDNN (N=%d C=%d Hin=%d Win=%d Hout=%d Wout=%d k=%dx%d s=%dx%d p=%dx%d)\n",
                     N, C, H_in, W_in, H_out, W_out, kH, kW, sH, sW, pH, pW);
        } else {
            std::fprintf(stderr, "DNN(cuDNN): maxpool2d cuDNN error: %s\n", cudnnGetErrorString(st));
        }

        if (pDesc) cudnnDestroyPoolingDescriptor(pDesc);
        if (yDesc) cudnnDestroyTensorDescriptor(yDesc);
        if (xDesc) cudnnDestroyTensorDescriptor(xDesc);
        if (handle) cudnnDestroy(handle);
        return;
    }
}

}}} // namespace cv::dnn::cuda_naive_conv
