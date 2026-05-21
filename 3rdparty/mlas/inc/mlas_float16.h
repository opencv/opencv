/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_float16.h

Abstract:

    Utilities for half precision floating type conversions.  Used internally
    by MLAS on platforms without half precision support.  Provided here as
    convenience for tests or other client libraries/apps.

--*/

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>


using _mlas_fp16_ = uint16_t;

union fp32_bits {
    uint32_t u;
    float f;
};

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)

/*PreFast told us to convert them to constexpr but the compiler says we can't.*/
#pragma warning(disable : 26497)

/*Added whole bunch of casts, still can't get rid of these overflow warnings.*/
#pragma warning(disable : 26450)
#pragma warning(disable : 26451)
#endif

inline 
_mlas_fp16_
MLAS_Float2Half(float ff)
{
    constexpr fp32_bits f32infty = {255 << 23};
    constexpr fp32_bits f16max = {(127 + 16) << 23};
    constexpr fp32_bits denorm_magic = {((127 - 15) + (23 - 10) + 1) << 23};
    constexpr uint32_t sign_mask = 0x80000000u;

    auto val = static_cast<uint16_t>(0x0u);
    fp32_bits f;
    f.f = ff;

    uint32_t sign = f.u & sign_mask;
    f.u ^= sign;

    if (f.u >= f16max.u) {
        // Inf or NaN (all exponent bits set)
        val = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;  // NaN->qNaN and Inf->Inf
    } else {
        if (f.u < (113 << 23)) {
            // Subnormal or zero
            // use a magic value to align our 10 mantissa bits at the bottom of
            // the float. as long as FP addition is round-to-nearest-even this
            // just works.
            f.f += denorm_magic.f;

            // and one integer subtract of the bias later, we have our final float!
            val = static_cast<uint16_t>(f.u - denorm_magic.u);
        } else {
            uint32_t mant_odd = (f.u >> 13) & 1;  // resulting mantissa is odd

            // update exponent, rounding bias part 1
            f.u += ((uint32_t)(15 - 127) << 23) + 0xfff;
            // rounding bias part 2
            f.u += mant_odd;
            // take the bits!
            val = static_cast<uint16_t>(f.u >> 13);
        }
    }

    val |= static_cast<uint16_t>(sign >> 16);
    return val;
}

inline
float
MLAS_Half2Float(_mlas_fp16_ val)
{
    constexpr fp32_bits magic = {113 << 23};
    constexpr uint32_t shifted_exp = 0x7c00 << 13;  // exponent mask after shift
    fp32_bits o;

    o.u = (val & 0x7fff) << 13;        // exponent/mantissa bits
    uint32_t exp = shifted_exp & o.u;  // just the exponent
    o.u += (127 - 15) << 23;           // exponent adjust

    // handle exponent special cases
    if (exp == shifted_exp) {     // Inf/NaN?
        o.u += (128 - 16) << 23;  // extra exp adjust
    } else if (exp == 0) {        // Zero/Denormal?
        o.u += 1 << 23;           // extra exp adjust
        o.f -= magic.f;           // renormalize
    }

    o.u |= (val & 0x8000) << 16;  // sign bit
    return o.f;
}

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif