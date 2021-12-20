//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Obsolete functions provided for compatibility, deprecated in favor
// of std:: functions.
//

#ifndef INCLUDED_IMATHMATH_H
#define INCLUDED_IMATHMATH_H

#include "ImathNamespace.h"
#include "ImathPlatform.h"
#include <cmath>
#include <limits>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

//----------------------------------------------------------------------------
//
// The deprecated Math<T> methods were intended to allow templated access to
// math functions so that they would automatically choose either the double
// (e.g. sin()) or float (e.g., sinf()) version.
//
// Beginning wth C++11, this is unnecessary, as std:: versions of all these
// functions are available and are templated by type.
//
// We keep these old definitions for backward compatibility but encourage
// users to prefer the std:: versions. Some day we may remove these
// deprecated versions.
//
//----------------------------------------------------------------------------

/// @cond Doxygen_Suppress
template <class T> struct Math
{
    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T acos (T x) { return std::acos (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T asin (T x) { return std::asin (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T atan (T x) { return std::atan (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T atan2 (T x, T y) { return std::atan2 (x, y); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T cos (T x) { return std::cos (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T sin (T x) { return std::sin (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T tan (T x) { return std::tan (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T cosh (T x) { return std::cosh (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T sinh (T x) { return std::sinh (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T tanh (T x) { return std::tanh (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T exp (T x) { return std::exp (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T log (T x) { return std::log (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T log10 (T x) { return std::log10 (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T modf (T x, T* iptr)
    {
        T ival;
        T rval (std::modf (T (x), &ival));
        *iptr = ival;
        return rval;
    }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T pow (T x, T y) { return std::pow (x, y); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T sqrt (T x) { return std::sqrt (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T ceil (T x) { return std::ceil (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T fabs (T x) { return std::fabs (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T floor (T x) { return std::floor (x); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T fmod (T x, T y) { return std::fmod (x, y); }

    IMATH_DEPRECATED("use std::math functions")
    IMATH_HOSTDEVICE
    static T hypot (T x, T y) { return std::hypot (x, y); }
};
/// @endcond


/// Don Hatch's version of sin(x)/x, which is accurate for very small x.
/// Returns 1 for x == 0.
template <class T>
IMATH_HOSTDEVICE inline T
sinx_over_x (T x)
{
    if (x * x < std::numeric_limits<T>::epsilon())
        return T (1);
    else
        return std::sin (x) / x;
}

/// Compare two numbers and test if they are "approximately equal":
///
/// @return Ttrue if x1 is the same as x2 with an absolute error of
/// no more than e:
///
///	abs (x1 - x2) <= e
template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
equalWithAbsError (T x1, T x2, T e) IMATH_NOEXCEPT
{
    return ((x1 > x2) ? x1 - x2 : x2 - x1) <= e;
}

/// Compare two numbers and test if they are "approximately equal":
///
/// @return True if x1 is the same as x2 with an relative error of
/// no more than e,
///
/// abs (x1 - x2) <= e * x1
template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
equalWithRelError (T x1, T x2, T e) IMATH_NOEXCEPT
{
    return ((x1 > x2) ? x1 - x2 : x2 - x1) <= e * ((x1 > 0) ? x1 : -x1);
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHMATH_H
