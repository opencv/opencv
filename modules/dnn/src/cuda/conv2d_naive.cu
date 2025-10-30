#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"
#include <vector>
#include <cudnn.h>
#include <cstdio>

namespace cv { namespace dnn { namespace cuda_naive_conv {

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
            int x_stride_w = 1;
            int x_stride_h = x_w;
            int x_stride_c = x_h * x_stride_h;
            // Use provided leading dimension (in elements) for batch stride to account for pitch
            int x_stride_n = in_ldw; // elements per row in the 2D view
            st = cudnnSetTensor4dDescriptorEx(xDesc, CUDNN_DATA_FLOAT, x_n, x_c, x_h, x_w,
                                              x_stride_n, x_stride_c, x_stride_h, x_stride_w);
            if (st != CUDNN_STATUS_SUCCESS) { ok = false; }
        }
        if (ok) {
            int y_n = N, y_c = C_out, y_h = H_out, y_w = W_out;
            int y_stride_w = 1;
            int y_stride_h = y_w;
            int y_stride_c = y_h * y_stride_h;
            // Use provided leading dimension (in elements) for batch stride to account for pitch
            int y_stride_n = out_ldw; // elements per row in the 2D view
            st = cudnnSetTensor4dDescriptorEx(yDesc, CUDNN_DATA_FLOAT, y_n, y_c, y_h, y_w,
                                              y_stride_n, y_stride_c, y_stride_h, y_stride_w);
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
        }

        // Algo and workspace
        cudnnConvolutionFwdAlgo_t algo;
        int maxAlgos = 0, retAlgos = 0;
        if (ok) {
#if CUDNN_MAJOR >= 8
            st = cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &maxAlgos);
            if (st != CUDNN_STATUS_SUCCESS || maxAlgos <= 0) ok = false;
            std::vector<cudnnConvolutionFwdAlgoPerf_t> perf;
            if (ok) {
                perf.resize(maxAlgos);
                st = cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc, maxAlgos, &retAlgos, perf.data());
                if (st != CUDNN_STATUS_SUCCESS || retAlgos <= 0) ok = false; else { algo = perf[0].algo; workspace_size = perf[0].memory; }
            }
#else
            st = cudnnGetConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
            if (st != CUDNN_STATUS_SUCCESS) ok = false;
            if (ok) {
                st = cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, algo, &workspace_size);
                if (st != CUDNN_STATUS_SUCCESS) ok = false;
            }
#endif
        }
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

        if (!ok) {
            std::fprintf(stderr,
                "DNN(cuDNN): conv2d_nchw_fp32 failed: %s (N=%d C_in=%d H_in=%d W_in=%d -> C_out=%d k=%dx%d s=%dx%d p=%dx%d groups=%d cpg=%d)\n",
                cudnnGetErrorString(st), N, C_in, H_in, W_in, C_out, kH, kW, strideH, strideW, padH, padW, groups, C_in_per_group);
        } else {
            std::fprintf(stderr,
                "DNN(cuDNN): conv2d_nchw_fp32 using cuDNN (N=%d C_in=%d H_in=%d W_in=%d -> C_out=%d H_out=%d W_out=%d k=%dx%d s=%dx%d p=%dx%d groups=%d cpg=%d)\n",
                N, C_in, H_in, W_in, C_out, H_out, W_out, kH, kW, strideH, strideW, padH, padW, groups, C_in_per_group);
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
