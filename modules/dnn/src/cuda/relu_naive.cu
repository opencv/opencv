#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"

namespace cv { namespace dnn { namespace cuda_naive_conv {

static __global__ void relu_fp32_kernel(const float* __restrict__ x, float* __restrict__ y, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

void relu_fp32(const float* d_input, float* d_output, size_t count)
{
    int threads = 256;
    int blocks = (int)((count + threads - 1) / threads);
    relu_fp32_kernel<<<blocks, threads>>>(d_input, d_output, count);
}

static __global__ void relu_fp32_2d_kernel(const float* __restrict__ x, size_t in_step, float* __restrict__ y, size_t out_step, int rows, int cols)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        const float* xr = (const float*)((const unsigned char*)x + r * in_step);
        float* yr = (float*)((unsigned char*)y + r * out_step);
        float v = xr[c];
        yr[c] = v > 0.0f ? v : 0.0f;
    }
}

void relu_fp32_2d(const float* d_input, size_t input_step, float* d_output, size_t output_step, int rows, int cols)
{
    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1)/threads.x, (rows + threads.y - 1)/threads.y);
    relu_fp32_2d_kernel<<<blocks, threads>>>(d_input, input_step, d_output, output_step, rows, cols);
}

}}} // namespace cv::dnn::cuda_naive_conv
