#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"

namespace cv { namespace dnn { namespace cuda_naive_conv {

static __global__ void global_avgpool2d_nchw_flat_fp32_kernel(
    const float* __restrict__ x, size_t in_step,
    float* __restrict__ y, size_t out_step,
    int N, int C, int H, int W)
{
    int n = blockIdx.y;
    int c = blockIdx.x;
    if (n >= N || c >= C) return;
    const float* xn = (const float*)((const unsigned char*)x + (size_t)n * in_step);
    float* yn = (float*)((unsigned char*)y + (size_t)n * out_step);
    int HW = H * W;
    size_t base = (size_t)c * HW;
    float sum = 0.f;
    for (int i = 0; i < HW; ++i) sum += xn[base + i];
    yn[c] = sum / (float)HW;
}

void global_avgpool2d_nchw_flat_fp32(
    const float* d_input, size_t input_step_bytes,
    float* d_output, size_t output_step_bytes,
    int N, int C, int H, int W)
{
    dim3 grid(C, N);
    dim3 block(1, 1);
    global_avgpool2d_nchw_flat_fp32_kernel<<<grid, block>>>(d_input, input_step_bytes, d_output, output_step_bytes, N, C, H, W);
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
    int threads = 256;
    int blocks_x = (H_out * W_out + threads - 1) / threads;
    dim3 grid(blocks_x, C, N);
    maxpool2d_nchw_flatrows_fp32_kernel<<<grid, threads>>>(
        d_input, input_step_bytes,
        d_output, output_step_bytes,
        N, C, H_in, W_in, H_out, W_out,
        kH, kW, sH, sW, pH, pW);
}

}}} // namespace cv::dnn::cuda_naive_conv
