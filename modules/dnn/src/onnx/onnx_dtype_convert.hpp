// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// TODO: stopgap until these dtypes are native cv::Mat depths; then this conversion math moves to core (convertTo) and only ONNX glue stays in DNN.

#ifndef OPENCV_DNN_ONNX_DTYPE_CONVERT_HPP
#define OPENCV_DNN_ONNX_DTYPE_CONVERT_HPP

#include <opencv2/core.hpp>
#include <cstdint>
#include <cmath>
#include <limits>

namespace cv { namespace dnn { namespace onnx_dtype {

// ONNX TensorProto.DataType value for FLOAT8E8M0 (absent from the bundled opencv-onnx.proto).
enum { ONNX_FLOAT8E8M0 = 24 };

// Description of a sign+exponent+mantissa low-precision float (the FP8 family).
struct Fp8Fmt { int ebits, mbits, bias; bool has_inf, fnuz; };

// Returns the format for an FP8 ONNX dtype, or {0,...} for non-FP8 types.
inline Fp8Fmt fp8FmtFor(int onnxType)
{
    switch (onnxType)
    {
        case 17: return {4, 3, 7,  false, false}; // FLOAT8E4M3FN
        case 18: return {4, 3, 8,  false, true};  // FLOAT8E4M3FNUZ
        case 19: return {5, 2, 15, true,  false}; // FLOAT8E5M2
        case 20: return {5, 2, 16, false, true};  // FLOAT8E5M2FNUZ
    }
    return {0, 0, 0, false, false};
}

inline bool isFp8(int onnxType)  { return onnxType >= 17 && onnxType <= 20; }
// FP8/FP4 decode to CV_16F values; E8M0 decodes to CV_32F; INT4->CV_8S, UINT4->CV_8U.
inline bool isExoticFloat(int t) { return isFp8(t) || t == 23 /*FP4*/ || t == ONNX_FLOAT8E8M0; }
inline bool isInt4(int t)        { return t == 22; }
inline bool isUint4(int t)       { return t == 21; }
inline bool isExotic(int t)      { return isExoticFloat(t) || isInt4(t) || isUint4(t); }

inline uint32_t f2u(float f) { Cv32suf s; s.f = f; return s.u; }

// Round 'full' dropping 'shift' low bits, round to nearest, ties to even.
inline uint32_t roundRNE(uint32_t full, int shift)
{
    if (shift <= 0) return full << (-shift);
    uint32_t q = full >> shift;
    uint32_t rem = full & ((1u << shift) - 1);
    uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (q & 1))) q++;
    return q;
}

inline uint8_t f32ToFp8(float x, const Fp8Fmt& f, bool saturate)
{
    uint32_t u = f2u(x), sign = (u >> 31) & 1, e = (u >> 23) & 0xFF, m = u & 0x7FFFFF;
    const int W = f.ebits + f.mbits;
    const uint32_t sbit = sign << W, maxe = (1u << f.ebits) - 1;
    const uint8_t NaNc = f.fnuz ? 0x80 : (uint8_t)(sbit | (maxe << f.mbits) | ((1u << f.mbits) - 1));
    uint8_t maxfin;
    if (f.has_inf)    maxfin = (uint8_t)(sbit | ((maxe - 1) << f.mbits) | ((1u << f.mbits) - 1));
    else if (f.fnuz)  maxfin = (uint8_t)(sbit | (maxe << f.mbits) | ((1u << f.mbits) - 1));
    else              maxfin = (uint8_t)(sbit | (maxe << f.mbits) | ((1u << f.mbits) - 2));
    const uint8_t Infc = (uint8_t)(sbit | (maxe << f.mbits));

    if (e == 0xFF && m != 0) return NaNc;
    if (e == 0xFF && m == 0)
    {
        if (f.has_inf && !f.fnuz) return saturate ? maxfin : Infc;
        return saturate ? maxfin : NaNc;
    }
    if (e == 0 && m == 0) return f.fnuz ? 0 : (uint8_t)sbit;

    int newexp = (int)e - 127 + f.bias;
    const uint32_t full = (1u << 23) | m;
    if (newexp <= 0)
    {
        const int shift = (23 - f.mbits) + (1 - newexp);
        uint32_t mant = (shift >= 32) ? 0u : roundRNE(full, shift);
        if (mant == 0) return f.fnuz ? 0 : (uint8_t)sbit;
        return (uint8_t)(sbit | mant);
    }
    uint32_t rounded = roundRNE(full, 23 - f.mbits);
    if (rounded & (1u << (f.mbits + 1))) { rounded >>= 1; newexp++; }
    const uint32_t mant = rounded & ((1u << f.mbits) - 1);
    bool ov;
    if (f.has_inf)   ov = (uint32_t)newexp >= maxe;
    else if (f.fnuz) ov = (uint32_t)newexp > maxe;
    else             ov = (uint32_t)newexp > maxe || ((uint32_t)newexp == maxe && mant == (1u << f.mbits) - 1);
    if (ov)
    {
        if (f.has_inf && !f.fnuz) return saturate ? maxfin : Infc;
        return saturate ? maxfin : NaNc;
    }
    return (uint8_t)(sbit | ((uint32_t)newexp << f.mbits) | mant);
}

// Quiet NaN whose sign matches the source code (ONNX preserves the sign bit, e.g. the
// FNUZ NaN code 0x80 decodes to -NaN). Matching the exact bits matters for bit-exact tests.
inline float signedQNaN(uint32_t sign) { Cv32suf s; s.u = sign ? 0xFFC00000u : 0x7FC00000u; return s.f; }

inline float fp8ToF32(uint8_t code, const Fp8Fmt& f)
{
    const int W = f.ebits + f.mbits;
    const uint32_t sign = (code >> W) & 1;
    const uint32_t exp = ((uint32_t)code >> f.mbits) & ((1u << f.ebits) - 1);
    const uint32_t man = code & ((1u << f.mbits) - 1);
    const uint32_t maxe = (1u << f.ebits) - 1;
    const float s = sign ? -1.f : 1.f;
    if (f.fnuz)
    {
        if ((uint32_t)code == (1u << W)) return signedQNaN(sign);
    }
    else if (exp == maxe)
    {
        if (f.has_inf) return man == 0 ? s * std::numeric_limits<float>::infinity() : signedQNaN(sign);
        if (man == (1u << f.mbits) - 1) return signedQNaN(sign);
    }
    if (exp == 0) return s * (float)man * std::ldexp(1.0f, 1 - f.bias - f.mbits);
    return s * (1.0f + (float)man / (float)(1u << f.mbits)) * std::ldexp(1.0f, (int)exp - f.bias);
}

// E8M0: unsigned, 8-bit exponent only (no sign, no mantissa), bias 127; 0xFF == NaN.
// Conversion rounds UP (ceil to next power of two), per the ONNX reference data.
inline uint8_t f32ToE8M0(float x)
{
    if (std::isnan(x)) return 0xFF;
    const uint32_t u = f2u(x), sign = (u >> 31) & 1, e = (u >> 23) & 0xFF, m = u & 0x7FFFFF;
    if (x == 0.f) return 0;
    if (sign) return 0xFF;       // negative -> NaN (unsigned format)
    if (e == 0xFF) return 0xFF;  // inf -> NaN
    int code = (int)e + (m > 0 ? 1 : 0);
    return (uint8_t)(code > 0xFE ? 0xFE : code);
}

inline float e8m0ToF32(uint8_t c)
{
    if (c == 0xFF) return std::numeric_limits<float>::quiet_NaN();
    return std::ldexp(1.0f, (int)c - 127);
}

// FLOAT4E2M1: 1-2-1, bias 1. Magnitudes {0,.5,1,1.5,2,3,4,6}; saturating; NaN -> 0x8.
inline const float* fp4Values()
{
    static const float v[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return v;
}

inline uint8_t f32ToFp4(float x)
{
    if (std::isnan(x)) return 0x8;
    const uint32_t sign = (f2u(x) >> 31) & 1;
    const float ax = std::fabs(x);
    const float* V = fp4Values();
    int best;
    if (std::isinf(ax) || ax >= 6.0f) best = 7; // saturate to max magnitude
    else
    {
        best = 0; float bestd = std::numeric_limits<float>::max();
        for (int c = 0; c < 8; c++)
        {
            const float d = std::fabs(ax - V[c]);
            if (d < bestd) { bestd = d; best = c; }
            else if (d == bestd && (best & 1)) best = c; // tie -> even code
        }
    }
    return (uint8_t)((sign << 3) | best);
}

inline float fp4ToF32(uint8_t code) { return (code & 0x8 ? -1.f : 1.f) * fp4Values()[code & 7]; }

// INT4/UINT4: cast truncates toward zero then keeps the low 4 bits (wraps; ONNX does not clamp).
inline int8_t  f32ToInt4(float x)  { int n = (int)std::trunc(x) & 0xF; return (int8_t)(n >= 8 ? n - 16 : n); }
inline uint8_t f32ToUint4(float x) { return (uint8_t)((int)std::trunc(x) & 0xF); }
inline int8_t  int4SignExtend(uint8_t nib) { return (int8_t)((nib & 0xF) >= 8 ? (int)(nib & 0xF) - 16 : (nib & 0xF)); }

// Extract the i-th 4-bit element from a tensor packed two-per-byte (low nibble first).
inline uint8_t unpackNibble(const uchar* raw, size_t i) { return (i & 1) ? (raw[i >> 1] >> 4) & 0xF : raw[i >> 1] & 0xF; }

}}} // namespace cv::dnn::onnx_dtype

#endif // OPENCV_DNN_ONNX_DTYPE_CONVERT_HPP
