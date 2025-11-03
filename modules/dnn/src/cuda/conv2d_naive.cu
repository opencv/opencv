#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"
#include <vector>
#include <cudnn.h>
#include <cstdio>

namespace cv { namespace dnn { namespace cuda_naive_conv {

__global__ void pad_nchw_kernel(const float* in, float* out,
                                int N, int C, int H_in, int W_in,
                                int H_pad, int W_pad,
                                int pad_top, int pad_left)
{
    size_t total = (size_t)N * (size_t)C * (size_t)H_in * (size_t)W_in;
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= total) return;

    int w = (int)(idx % (size_t)W_in);
    size_t tmp = idx / (size_t)W_in;
    int h = (int)(tmp % (size_t)H_in);
    tmp /= (size_t)H_in;
    int c = (int)(tmp % (size_t)C);
    int n = (int)(tmp / (size_t)C);

    size_t dst_idx = (((size_t)n * (size_t)C + (size_t)c) * (size_t)H_pad + (size_t)(h + pad_top)) * (size_t)W_pad + (size_t)(w + pad_left);
    size_t src_idx = (((size_t)n * (size_t)C + (size_t)c) * (size_t)H_in + (size_t)h) * (size_t)W_in + (size_t)w;
    out[dst_idx] = in[src_idx];
}

void pad_nchw_fp32(
    const float* d_input,
    float* d_output,
    int N,
    int C,
    int H_in,
    int W_in,
    int H_pad,
    int W_pad,
    int pad_top,
    int pad_left)
{
    size_t total = (size_t)N * (size_t)C * (size_t)H_in * (size_t)W_in;
    int threads = 256;
    int blocks = (int)((total + (size_t)threads - 1) / (size_t)threads);
    if (blocks > 0) {
        pad_nchw_kernel<<<blocks, threads>>>(d_input, d_output, N, C, H_in, W_in, H_pad, W_pad, pad_top, pad_left);
        cudaDeviceSynchronize();
    }
}

void conv2d_nchw_fp32(
    const float* d_input,
    const float* d_weights,
    const float* d_bias,
    float* d_output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int in_ldw, int out_ldw, int w_ldw, int w_ldh,
    int groups, int C_in_per_group)
{
    int H_out = (H_in + 2 * padH - kH) / strideH + 1;
    int W_out = (W_in + 2 * padW - kW) / strideW + 1;

    {
        cudnnHandle_t handle = nullptr;
        cudnnTensorDescriptor_t xDesc = nullptr, yDesc = nullptr, bDesc = nullptr;
        cudnnFilterDescriptor_t wDesc = nullptr;
        cudnnConvolutionDescriptor_t convDesc = nullptr;
        void* workspace = nullptr;
        size_t workspace_size = 0;

        cudnnStatus_t st;
        st = cudnnCreate(&handle); if (st != CUDNN_STATUS_SUCCESS) { return; }
        cudnnSetStream(handle, 0);

        bool ok = true;
        if (ok) {
            st = cudnnCreateTensorDescriptor(&xDesc);
            if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
        }
        if (ok) {
            st = cudnnCreateTensorDescriptor(&yDesc);
            if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
        }
        if (ok && d_bias) {
            st = cudnnCreateTensorDescriptor(&bDesc);
            if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
        }
        if (ok) {
            st = cudnnCreateFilterDescriptor(&wDesc);
            if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
        }
        if (ok) {
            st = cudnnCreateConvolutionDescriptor(&convDesc);
            if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
        }

        if (ok) {
            int x_n = N, x_c = C_in, x_h = H_in, x_w = W_in;
            int contiguous_in = x_c * x_h * x_w;
            if (in_ldw <= 0 || in_ldw == contiguous_in) {
                st = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, x_n, x_c, x_h, x_w);
            } else {
                int x_stride_w = 1;
                int x_stride_h = x_w;
                int x_stride_c = x_h * x_stride_h;
                int x_stride_n = in_ldw;
                st = cudnnSetTensor4dDescriptorEx(xDesc, CUDNN_DATA_FLOAT, x_n, x_c, x_h, x_w,
                                                  x_stride_n, x_stride_c, x_stride_h, x_stride_w);
            }
            if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
        }
        if (ok) {
            int y_n = N, y_c = C_out, y_h = H_out, y_w = W_out;
            int contiguous_out = y_c * y_h * y_w;
            if (out_ldw <= 0 || out_ldw == contiguous_out) {
                st = cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, y_n, y_c, y_h, y_w);
            } else {
                int y_stride_w = 1;
                int y_stride_h = y_w;
                int y_stride_c = y_h * y_stride_h;
                int y_stride_n = out_ldw;
                st = cudnnSetTensor4dDescriptorEx(yDesc, CUDNN_DATA_FLOAT, y_n, y_c, y_h, y_w,
                                                  y_stride_n, y_stride_c, y_stride_h, y_stride_w);
            }
            if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
        }
        if (ok && d_bias) {
            st = cudnnSetTensor4dDescriptor(bDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1);
            if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
        }
        if (ok) {
            st = cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in_per_group, kH, kW);
            if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
        }
        if (ok) {
            st = cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strideH, strideW, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
            if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
            if (ok && groups > 1) {
                st = cudnnSetConvolutionGroupCount(convDesc, groups);
                if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
            }
            // Reduce numeric drift vs CPU by avoiding TF32/tensor cores
#if CUDNN_MAJOR >= 7
            if (ok) {
                st = cudnnSetConvolutionMathType(convDesc, CUDNN_DEFAULT_MATH);
                if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
            }
#endif
        }

        // Algo and workspace: pick a stable, deterministic algo
        cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        workspace_size = 0;
        if (ok && workspace_size > 0) {
            if (cudaMalloc(&workspace, workspace_size) != cudaSuccess) ok = false;
        }
        if (ok) {
            const float alpha = 1.0f, beta = 0.0f;
            st = cudnnConvolutionForward(handle, &alpha, xDesc, d_input, wDesc, d_weights, convDesc, algo, workspace, workspace_size, &beta, yDesc, d_output);
            if (st != CUDNN_STATUS_SUCCESS) ok = false;
        }
        if (ok && d_bias) {
            const float alpha = 1.0f, beta = 1.0f;
            st = cudnnAddTensor(handle, &alpha, bDesc, d_bias, &beta, yDesc, d_output);
            if (st != CUDNN_STATUS_SUCCESS) ok = false;
        }

        if (workspace) cudaFree(workspace);
        if (convDesc) cudnnDestroyConvolutionDescriptor(convDesc);
        if (wDesc) cudnnDestroyFilterDescriptor(wDesc);
        if (bDesc) cudnnDestroyTensorDescriptor(bDesc);
        if (yDesc) cudnnDestroyTensorDescriptor(yDesc);
        if (xDesc) cudnnDestroyTensorDescriptor(xDesc);
        if (handle) cudnnDestroy(handle);
        return;
    }
}

}}}
