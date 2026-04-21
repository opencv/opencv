// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <opencv2/dnn/all_layers.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include <math.h>

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

cv::dnn::ActivationFunc getActivationFunc_(int type);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// Mish: x * tanh(softplus(x))
// Uses numerically stable form: x * (1 + 2*y) / (1 + 2*y + 2*y*y) where y = exp(-x) for large x
static void activationMish(const void* input, void* output,
                           size_t len, const float* /*params*/)
{
    const float* inp = (const float*)input;
    float* out = (float*)output;
    const float MISH_THRESHOLD = -36.73f;
    size_t i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    v_float32 v_threshold = vx_setall_f32(MISH_THRESHOLD);
    v_float32 one = vx_setall_f32(1.f), z = vx_setzero_f32();
    for (; i + vlanes <= len; i += vlanes) {
        v_float32 x = vx_load(inp + i);
        x = v_select(v_le(x, v_threshold), z, x);
        v_float32 y = v_exp(v_sub(z, x));
        v_float32 _2y = v_add(y, y);
        v_float32 _2ya1 = v_add(_2y, one);
        x = v_div(v_mul(x, _2ya1), v_add(_2ya1, v_mul(_2y, y)));
        vx_store(out + i, x);
    }
#endif
    for (; i < len; i++) {
        float x = inp[i];
        if (x <= MISH_THRESHOLD) { out[i] = 0.f; continue; }
        float y = expf(-x);
        float _2y = 2.f * y;
        out[i] = x * (1.f + _2y) / (1.f + _2y + _2y * y);
    }
}

// Swish/SiLU: x / (1 + exp(-x))
static void activationSwish(const void* input, void* output,
                            size_t len, const float* /*params*/)
{
    const float* inp = (const float*)input;
    float* out = (float*)output;
    size_t i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    v_float32 one = vx_setall_f32(1.f), z = vx_setzero_f32();
    for (; i + vlanes <= len; i += vlanes) {
        v_float32 x = vx_load(inp + i);
        v_float32 t = v_exp(v_sub(z, x));
        t = v_div(x, v_add(one, t));
        vx_store(out + i, t);
    }
#endif
    for (; i < len; i++) {
        float x = inp[i];
        out[i] = x / (1.f + expf(-x));
    }
}

// Sigmoid: 1 / (1 + exp(-x))
static void activationSigmoid(const void* input, void* output,
                              size_t len, const float* /*params*/)
{
    const float* inp = (const float*)input;
    float* out = (float*)output;
    size_t i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    v_float32 one = vx_setall_f32(1.f), z = vx_setzero_f32();
    for (; i + vlanes <= len; i += vlanes) {
        v_float32 x = vx_load(inp + i);
        v_float32 t = v_exp(v_sub(z, x));
        t = v_div(one, v_add(one, t));
        vx_store(out + i, t);
    }
#endif
    for (; i < len; i++) {
        float x = inp[i];
        out[i] = 1.f / (1.f + expf(-x));
    }
}

// TanH: uses v_exp SIMD for (exp(2x)-1)/(exp(2x)+1)
static void activationTanH(const void* input, void* output,
                           size_t len, const float* /*params*/)
{
    const float* inp = (const float*)input;
    float* out = (float*)output;
    size_t i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    v_float32 one = vx_setall_f32(1.f), two = vx_setall_f32(2.f);
    for (; i + vlanes <= len; i += vlanes) {
        v_float32 x = vx_load(inp + i);
        v_float32 e2x = v_exp(v_mul(two, x));
        v_float32 t = v_div(v_sub(e2x, one), v_add(e2x, one));
        vx_store(out + i, t);
    }
#endif
    for (; i < len; i++) {
        out[i] = tanhf(inp[i]);
    }
}

// ELU: x >= 0 ? x : alpha*(exp(x)-1)
static void activationELU(const void* input, void* output,
                          size_t len, const float* params)
{
    const float* inp = (const float*)input;
    float* out = (float*)output;
    float alpha = params[0];
    size_t i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    v_float32 v_alpha = vx_setall_f32(alpha);
    v_float32 one = vx_setall_f32(1.f), z = vx_setzero_f32();
    for (; i + vlanes <= len; i += vlanes) {
        v_float32 x = vx_load(inp + i);
        v_float32 t = v_mul(v_alpha, v_sub(v_exp(x), one));
        x = v_select(v_ge(x, z), x, t);
        vx_store(out + i, x);
    }
#endif
    for (; i < len; i++) {
        float x = inp[i];
        out[i] = x >= 0.f ? x : alpha * (expf(x) - 1.f);
    }
}

// HardSwish: x * clip(x/6 + 0.5, 0, 1)
static void activationHardSwish(const void* input, void* output,
                                size_t len, const float* /*params*/)
{
    const float* inp = (const float*)input;
    float* out = (float*)output;
    size_t i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    v_float32 zero = vx_setzero_f32(), one = vx_setall_f32(1.f);
    v_float32 half = vx_setall_f32(0.5f), sixth = vx_setall_f32(1.f / 6.f);
    for (; i + vlanes <= len; i += vlanes) {
        v_float32 x = vx_load(inp + i);
        v_float32 t = v_min(one, v_max(zero, v_add(v_mul(x, sixth), half)));
        vx_store(out + i, v_mul(x, t));
    }
#endif
    for (; i < len; i++) {
        float x = inp[i];
        out[i] = x * std::min(std::max(x / 6.f + 0.5f, 0.f), 1.f);
    }
}

// HardSigmoid: clip(alpha*x + beta, 0, 1)
static void activationHardSigmoid(const void* input, void* output,
                                  size_t len, const float* params)
{
    const float* inp = (const float*)input;
    float* out = (float*)output;
    float alpha = params[0];
    float beta = params[1];
    size_t i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    v_float32 v_alpha = vx_setall_f32(alpha), v_beta = vx_setall_f32(beta);
    v_float32 zero = vx_setzero_f32(), one = vx_setall_f32(1.f);
    for (; i + vlanes <= len; i += vlanes) {
        v_float32 x = vx_load(inp + i);
        x = v_min(one, v_max(zero, v_add(v_mul(v_alpha, x), v_beta)));
        vx_store(out + i, x);
    }
#endif
    for (; i < len; i++) {
        out[i] = std::min(std::max(alpha * inp[i] + beta, 0.f), 1.f);
    }
}

// GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
static void activationGELU(const void* input, void* output,
                           size_t len, const float* /*params*/)
{
    const float* inp = (const float*)input;
    float* out = (float*)output;
    size_t i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    v_float32 half = vx_setall_f32(0.5f), one = vx_setall_f32(1.f);
    v_float32 rsqrt2 = vx_setall_f32((float)M_SQRT1_2);
    for (; i + vlanes <= len; i += vlanes) {
        v_float32 x = vx_load(inp + i);
        v_float32 t = v_add(one, v_erf(v_mul(rsqrt2, x)));
        vx_store(out + i, v_mul(v_mul(half, x), t));
    }
#endif
    for (; i < len; i++) {
        float x = inp[i];
        out[i] = 0.5f * x * (1.f + erff(x * (float)M_SQRT1_2));
    }
}

// GELU approximate: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static void activationGELUApprox(const void* input, void* output,
                                 size_t len, const float* /*params*/)
{
    const float* inp = (const float*)input;
    float* out = (float*)output;
    const float sqrt2_pi = 0.7978845834732056f;    // sqrt(2/pi)
    const float coeff = 0.044715f * sqrt2_pi;
    size_t i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    v_float32 half = vx_setall_f32(0.5f), one = vx_setall_f32(1.f);
    v_float32 v_s2pi = vx_setall_f32(sqrt2_pi), v_coeff = vx_setall_f32(coeff);
    v_float32 two = vx_setall_f32(2.f);
    // Clamp to [-9, 9] to prevent overflow in exp(2*inner); tanh saturates here anyway
    v_float32 clamp_hi = vx_setall_f32(9.f), clamp_lo = vx_setall_f32(-9.f);
    for (; i + vlanes <= len; i += vlanes) {
        v_float32 x = vx_load(inp + i);
        // inner = sqrt(2/pi) * x + coeff * x^3 = x * (sqrt(2/pi) + coeff * x^2)
        v_float32 inner = v_mul(x, v_add(v_s2pi, v_mul(v_coeff, v_mul(x, x))));
        inner = v_min(v_max(inner, clamp_lo), clamp_hi);
        // tanh via exp: (exp(2*inner)-1)/(exp(2*inner)+1)
        v_float32 e2 = v_exp(v_mul(two, inner));
        v_float32 t = v_div(v_sub(e2, one), v_add(e2, one));
        vx_store(out + i, v_mul(v_mul(half, x), v_add(one, t)));
    }
#endif
    for (; i < len; i++) {
        float x = inp[i];
        float inner = x * (sqrt2_pi + coeff * x * x);
        out[i] = 0.5f * x * (1.f + tanhf(inner));
    }
}

// ReLU: max(0, x) or LeakyReLU: x >= 0 ? x : alpha*x
// params[0] = negative slope (0 for plain ReLU)
static void activationReLU(const void* input, void* output,
                           size_t len, const float* params)
{
    const float* inp = (const float*)input;
    float* out = (float*)output;
    float alpha = params ? params[0] : 0.f;
    size_t i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    v_float32 v_alpha = vx_setall_f32(alpha), z = vx_setzero_f32();
    for (; i + vlanes <= len; i += vlanes) {
        v_float32 x = vx_load(inp + i);
        x = v_select(v_ge(x, z), x, v_mul(x, v_alpha));
        vx_store(out + i, x);
    }
#endif
    for (; i < len; i++) {
        float x = inp[i];
        out[i] = x >= 0.f ? x : alpha * x;
    }
}

// Clip: clamp(x, minval, maxval)
// params[0] = minval, params[1] = maxval
static void activationClip(const void* input, void* output,
                           size_t len, const float* params)
{
    const float* inp = (const float*)input;
    float* out = (float*)output;
    float minval = params[0];
    float maxval = params[1];
    size_t i = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int vlanes = VTraits<v_float32>::vlanes();
    v_float32 v_lo = vx_setall_f32(minval), v_hi = vx_setall_f32(maxval);
    for (; i + vlanes <= len; i += vlanes) {
        v_float32 x = vx_load(inp + i);
        x = v_min(v_max(x, v_lo), v_hi);
        vx_store(out + i, x);
    }
#endif
    for (; i < len; i++) {
        out[i] = std::min(std::max(inp[i], minval), maxval);
    }
}

ActivationFunc getActivationFunc_(int type)
{
    switch (type) {
    case ACTIV_MISH: return activationMish;
    case ACTIV_SWISH: return activationSwish;
    case ACTIV_SIGMOID: return activationSigmoid;
    case ACTIV_TANH: return activationTanH;
    case ACTIV_ELU: return activationELU;
    case ACTIV_HARDSWISH: return activationHardSwish;
    case ACTIV_HARDSIGMOID: return activationHardSigmoid;
    case ACTIV_GELU: return activationGELU;
    case ACTIV_GELU_APPROX: return activationGELUApprox;
    case ACTIV_RELU: return activationReLU;
    case ACTIV_CLIP: return activationClip;
    default: return nullptr;
    }
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // cv::dnn::

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
