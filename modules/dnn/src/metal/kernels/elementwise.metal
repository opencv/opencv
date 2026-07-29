#include <metal_stdlib>

using namespace metal;

struct unary_params
{
    uint count;
    float value_0;
    float value_1;
    float value_2;
};

struct prelu_params
{
    uint count;
    uint slope_count;
    uint inner_size;
};

static inline float opencv_erf_approx(float value)
{
    constexpr float coefficient_0 = 0.3275911f;
    constexpr float coefficient_1 = 1.061405429f;
    constexpr float coefficient_2 = -1.453152027f;
    constexpr float coefficient_3 = 1.421413741f;
    constexpr float coefficient_4 = -0.284496736f;
    constexpr float coefficient_5 = 0.254829592f;
    const float t = 1.0f / fma(fabs(value), coefficient_0, 1.0f);
    float result = fma(coefficient_1, t, coefficient_2);
    result = fma(result, t, coefficient_3);
    result = fma(result, t, coefficient_4);
    result = fma(result, t, coefficient_5);
    result = 1.0f - result * t * exp(-value * value);
    return copysign(result, value);
}

#define OPENCV_DNN_UNARY_KERNEL(name, expression)                                      \
kernel void name(                                                                      \
    device const float* input [[buffer(0)]],                                            \
    device float* output [[buffer(1)]],                                                 \
    constant unary_params& params [[buffer(2)]],                                       \
    uint gid [[thread_position_in_grid]])                                               \
{                                                                                      \
    if (gid >= params.count)                                                            \
        return;                                                                         \
    const float x = input[gid];                                                         \
    output[gid] = (expression);                                                         \
}

OPENCV_DNN_UNARY_KERNEL(kernel_relu_f32,
    x >= 0.0f ? x : x * params.value_0)
OPENCV_DNN_UNARY_KERNEL(kernel_relu6_f32,
    clamp(x, params.value_0, params.value_1))
OPENCV_DNN_UNARY_KERNEL(kernel_sigmoid_f32,
    x >= 0.0f ? 1.0f / (1.0f + exp(-x)) : exp(x) / (1.0f + exp(x)))
OPENCV_DNN_UNARY_KERNEL(kernel_swish_f32, x / (1.0f + exp(-x)))
OPENCV_DNN_UNARY_KERNEL(kernel_gelu_f32,
    0.5f * x * (1.0f + opencv_erf_approx(x * 0.7071067811865475f)))
OPENCV_DNN_UNARY_KERNEL(kernel_gelu_approximation_f32,
    0.5f * x * (1.0f + tanh(x * (0.7978845834732056f +
        0.035677408136300125f * x * x))))
OPENCV_DNN_UNARY_KERNEL(kernel_tanh_f32, tanh(x))
OPENCV_DNN_UNARY_KERNEL(kernel_mish_f32,
    x > -36.73f ? x * (1.0f + 2.0f * exp(-x)) /
        (1.0f + 2.0f * exp(-x) + 2.0f * exp(-2.0f * x)) : 0.0f)
OPENCV_DNN_UNARY_KERNEL(kernel_elu_f32,
    x >= 0.0f ? x : params.value_0 * (exp(x) - 1.0f))
OPENCV_DNN_UNARY_KERNEL(kernel_abs_f32, fabs(x))
OPENCV_DNN_UNARY_KERNEL(kernel_bnll_f32,
    x > 0.0f ? x + log(1.0f + exp(-x)) : log(1.0f + exp(x)))
OPENCV_DNN_UNARY_KERNEL(kernel_ceil_f32, ceil(x))
OPENCV_DNN_UNARY_KERNEL(kernel_floor_f32, floor(x))
OPENCV_DNN_UNARY_KERNEL(kernel_log_f32, log(x))
OPENCV_DNN_UNARY_KERNEL(kernel_round_f32, rint(x))
OPENCV_DNN_UNARY_KERNEL(kernel_sqrt_f32, sqrt(x))
OPENCV_DNN_UNARY_KERNEL(kernel_acos_f32, acos(x))
OPENCV_DNN_UNARY_KERNEL(kernel_acosh_f32, acosh(x))
OPENCV_DNN_UNARY_KERNEL(kernel_asin_f32, asin(x))
OPENCV_DNN_UNARY_KERNEL(kernel_asinh_f32, asinh(x))
OPENCV_DNN_UNARY_KERNEL(kernel_atan_f32, atan(x))
OPENCV_DNN_UNARY_KERNEL(kernel_atanh_f32, atanh(x))
OPENCV_DNN_UNARY_KERNEL(kernel_cos_f32, cos(x))
OPENCV_DNN_UNARY_KERNEL(kernel_cosh_f32, cosh(x))
OPENCV_DNN_UNARY_KERNEL(kernel_erf_f32, opencv_erf_approx(x))
OPENCV_DNN_UNARY_KERNEL(kernel_hard_swish_f32,
    x * clamp(x / 6.0f + 0.5f, 0.0f, 1.0f))
OPENCV_DNN_UNARY_KERNEL(kernel_sin_f32, sin(x))
OPENCV_DNN_UNARY_KERNEL(kernel_sinh_f32, sinh(x))
OPENCV_DNN_UNARY_KERNEL(kernel_softplus_f32, log(1.0f + exp(x)))
OPENCV_DNN_UNARY_KERNEL(kernel_softsign_f32, x / (1.0f + fabs(x)))
OPENCV_DNN_UNARY_KERNEL(kernel_tan_f32, tan(x))
OPENCV_DNN_UNARY_KERNEL(kernel_celu_f32,
    max(0.0f, x) + min(0.0f, params.value_0 * (exp(x / params.value_0) - 1.0f)))
OPENCV_DNN_UNARY_KERNEL(kernel_hard_sigmoid_f32,
    clamp(params.value_0 * x + params.value_1, 0.0f, 1.0f))
OPENCV_DNN_UNARY_KERNEL(kernel_selu_f32,
    params.value_1 * (x > 0.0f ? x : params.value_0 * (exp(x) - 1.0f)))
OPENCV_DNN_UNARY_KERNEL(kernel_thresholded_relu_f32,
    x > params.value_0 ? x : 0.0f)
OPENCV_DNN_UNARY_KERNEL(kernel_power_f32,
    pow(params.value_1 * x + params.value_2, params.value_0))
OPENCV_DNN_UNARY_KERNEL(kernel_exp_f32,
    exp(params.value_0 * x + params.value_1))
OPENCV_DNN_UNARY_KERNEL(kernel_sign_f32,
    x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f))
OPENCV_DNN_UNARY_KERNEL(kernel_shrink_f32,
    x > params.value_1 ? x - params.value_0 :
        (x < -params.value_1 ? x + params.value_0 : 0.0f))
OPENCV_DNN_UNARY_KERNEL(kernel_reciprocal_f32, 1.0f / x)

#undef OPENCV_DNN_UNARY_KERNEL

kernel void kernel_channels_prelu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* slope [[buffer(2)]],
    constant prelu_params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.count)
        return;
    const uint channel = (gid / params.inner_size) % params.slope_count;
    const float value = input[gid];
    output[gid] = value >= 0.0f ? value : value * slope[channel];
}

kernel void kernel_prelu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* slope [[buffer(2)]],
    constant prelu_params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.count)
        return;
    const float value = input[gid];
    output[gid] = value >= 0.0f ? value : value * slope[gid % params.slope_count];
}
