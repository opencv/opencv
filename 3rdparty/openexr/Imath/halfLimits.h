//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Primary origin authors:
//     Florian Kainz <kainz@ilm.com>
//     Rod Bogart <rgb@ilm.com>
//

#ifndef INCLUDED_HALF_LIMITS_H
#define INCLUDED_HALF_LIMITS_H

//------------------------------------------------------------------------
//
//	C++ standard library-style numeric_limits for class half
//
//------------------------------------------------------------------------

#include "half.h"
#include <limits>

/// @cond Doxygen_Suppress

namespace std
{

template <> class numeric_limits<half>
{
  public:
    static const bool is_specialized = true;

    static constexpr half min() IMATH_NOEXCEPT { return half(half::FromBits, 0x0400); /*HALF_MIN*/ }
    static constexpr half max() IMATH_NOEXCEPT { return half(half::FromBits, 0x7bff); /*HALF_MAX*/ }
    static constexpr half lowest() { return half(half::FromBits, 0xfbff); /* -HALF_MAX */ }

    static constexpr int digits       = HALF_MANT_DIG;
    static constexpr int digits10     = HALF_DIG;
    static constexpr int max_digits10 = HALF_DECIMAL_DIG;
    static constexpr bool is_signed   = true;
    static constexpr bool is_integer  = false;
    static constexpr bool is_exact    = false;
    static constexpr int radix        = HALF_RADIX;
    static constexpr half epsilon() IMATH_NOEXCEPT { return half(half::FromBits, 0x1400); /*HALF_EPSILON*/ }
    static constexpr half round_error() IMATH_NOEXCEPT { return half(half::FromBits, 0x3800); /*0.5*/ }

    static constexpr int min_exponent   = HALF_DENORM_MIN_EXP;
    static constexpr int min_exponent10 = HALF_DENORM_MIN_10_EXP;
    static constexpr int max_exponent   = HALF_MAX_EXP;
    static constexpr int max_exponent10 = HALF_MAX_10_EXP;

    static constexpr bool has_infinity             = true;
    static constexpr bool has_quiet_NaN            = true;
    static constexpr bool has_signaling_NaN        = true;
    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr bool has_denorm_loss          = false;
    static constexpr half infinity() IMATH_NOEXCEPT { return half(half::FromBits, 0x7c00); /*half::posInf()*/ }
    static constexpr half quiet_NaN() IMATH_NOEXCEPT { return half(half::FromBits, 0x7fff); /*half::qNan()*/ }
    static constexpr half signaling_NaN() IMATH_NOEXCEPT { return half(half::FromBits, 0x7dff); /*half::sNan()*/ }
    static constexpr half denorm_min() IMATH_NOEXCEPT { return half(half::FromBits, 0x0001); /*HALF_DENORM_MIN*/ }

    static constexpr bool is_iec559  = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_modulo  = false;

    static constexpr bool traps                    = true;
    static constexpr bool tinyness_before          = false;
    static constexpr float_round_style round_style = round_to_nearest;
};

/// @endcond

} // namespace std

#endif
