#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"

namespace cv { namespace dnn { namespace cuda_naive_conv {

static __global__ void conv2d_nchw_fp32_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int strideH, int strideW,
    int padH, int padW,
    int H_out, int W_out,
    int in_ldw, int out_ldw, int w_ldw, int w_ldh)
{
    int n = blockIdx.z;     // batch index
    int co = blockIdx.y;    // output channel
    int hw = blockIdx.x * blockDim.x + threadIdx.x; // flattened H_out*W_out
    if (n >= N || co >= C_out || hw >= H_out * W_out) return;

    int oh = hw / W_out;
    int ow = hw % W_out;

    float sum = bias ? bias[co] : 0.0f;

    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < kH; ++kh) {
            int ih = oh * strideH - padH + kh;
            if (ih < 0 || ih >= H_in) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int iw = ow * strideW - padW + kw;
                if (iw < 0 || iw >= W_in) continue;
                size_t in_idx = ((size_t)n * C_in + ci) * (size_t)H_in * (size_t)in_ldw + (size_t)ih * (size_t)in_ldw + (size_t)iw;
                size_t w_idx  = ((size_t)co * C_in + ci) * (size_t)kH * (size_t)w_ldh + (size_t)kh * (size_t)w_ldh + (size_t)kw;
                sum += input[in_idx] * weights[w_idx];
            }
        }
    }

    size_t out_idx = ((size_t)n * C_out + co) * (size_t)H_out * (size_t)out_ldw + (size_t)oh * (size_t)out_ldw + (size_t)ow;
    output[out_idx] = sum;
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
    int in_ldw, int out_ldw, int w_ldw, int w_ldh)
{
    int H_out = (H_in + 2 * padH - kH) / strideH + 1;
    int W_out = (W_in + 2 * padW - kW) / strideW + 1;
    int total_hw = H_out * W_out;

    int threads = 256;
    int blocks_x = (total_hw + threads - 1) / threads;
    dim3 grid(blocks_x, C_out, N);

    conv2d_nchw_fp32_kernel<<<grid, threads>>>(d_input, d_weights, d_bias, d_output,
        N, C_in, H_in, W_in, C_out, kH, kW, strideH, strideW, padH, padW, H_out, W_out, in_ldw, out_ldw, w_ldw, w_ldh);
}

}}} // namespace cv::dnn::cuda_naive_conv
