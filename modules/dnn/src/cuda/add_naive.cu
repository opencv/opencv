#include <cuda_runtime.h>
#include <float.h>
#include "conv_naive.hpp"

namespace cv { namespace dnn { namespace cuda_naive_conv {

static __global__ void add2_fp32_2d_kernel(
    const float* __restrict__ a, size_t a_step,
    const float* __restrict__ b, size_t b_step,
    float* __restrict__ y, size_t y_step,
    int rows, int cols)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        const float* ar = (const float*)((const unsigned char*)a + r * a_step);
        const float* br = (const float*)((const unsigned char*)b + r * b_step);
        float* yr = (float*)((unsigned char*)y + r * y_step);
        yr[c] = ar[c] + br[c];
    }
}

void add2_fp32_2d(
    const float* d_a, size_t a_step,
    const float* d_b, size_t b_step,
    float* d_y, size_t y_step,
    int rows, int cols)
{
    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1)/threads.x, (rows + threads.y - 1)/threads.y);
    add2_fp32_2d_kernel<<<blocks, threads>>>(d_a, a_step, d_b, b_step, d_y, y_step, rows, cols);
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
    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1)/threads.x, (rows + threads.y - 1)/threads.y);
    add_inplace_fp32_2d_kernel<<<blocks, threads>>>(d_x, x_step, d_y, y_step, rows, cols);
}

}}} // namespace cv::dnn::cuda_naive_conv
