#pragma once

namespace cv { namespace dnn { namespace cuda_naive_conv {

// Launch a naive NCHW FP32 convolution on the default CUDA stream.
// Assumptions:
// - groups == 1
// - dilation_h == dilation_w == 1
// - data layout: input [N, C_in, H_in, W_in], weights [C_out, C_in, kH, kW], bias [C_out] or nullptr
// - output layout: [N, C_out, H_out, W_out]
// - All pointers refer to device (GPU) memory with contiguous layout
void conv2d_nchw_fp32(
    const float* d_input,          // N * C_in * H_in * W_in
    const float* d_weights,        // C_out * C_in * kH * kW
    const float* d_bias,           // C_out or nullptr
    float* d_output,               // N * C_out * H_out * W_out
    int N,
    int C_in,
    int H_in,
    int W_in,
    int C_out,
    int kH,
    int kW,
    int strideH,
    int strideW,
    int padH,
    int padW,
    int in_ldw,      // input row stride in elements
    int out_ldw,     // output row stride in elements
    int w_ldw,       // weights row stride in elements (advance over kw)
    int w_ldh        // weights stride over kh in elements
);

// Flat FP32 ReLU: y[i] = max(x[i], 0)
void relu_fp32(const float* d_input, float* d_output, size_t count);

// Pitch-aware FP32 ReLU on a 2D plane
void relu_fp32_2d(const float* d_input, size_t input_step, float* d_output, size_t output_step, int rows, int cols);

// Elementwise add two 2D planes (pitch-aware)
void add2_fp32_2d(const float* d_a, size_t a_step,
                  const float* d_b, size_t b_step,
                  float* d_y, size_t y_step,
                  int rows, int cols);

// Elementwise add x into y in place (pitch-aware)
void add_inplace_fp32_2d(const float* d_x, size_t x_step,
                         float* d_y, size_t y_step,
                         int rows, int cols);

// Global average pooling over HxW for NCHW flattened rows -> output is N x C
void global_avgpool2d_nchw_flat_fp32(
    const float* d_input, size_t input_step_bytes,
    float* d_output, size_t output_step_bytes,
    int N, int C, int H, int W);

// MaxPool NCHW with rows flattened to N and columns C*H*W; pitch-aware
void maxpool2d_nchw_flatrows_fp32(
    const float* d_input, size_t input_step_bytes,
    float* d_output, size_t output_step_bytes,
    int N, int C,
    int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW);

}}} // namespace cv::dnn::cuda_naive_conv
