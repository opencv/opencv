//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMATHFUN_H
#define INCLUDED_IMATHFUN_H

//-----------------------------------------------------------------------------
//
//	Miscellaneous utility functions
//
//-----------------------------------------------------------------------------

#include <limits>
#include <cstdint>

#include "ImathExport.h"
#include "ImathNamespace.h"
#include "ImathPlatform.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

template <class T>
IMATH_HOSTDEVICE constexpr inline T
abs (T a) IMATH_NOEXCEPT
{
    return (a > T (0)) ? a : -a;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline int
sign (T a) IMATH_NOEXCEPT
{
    return (a > T (0)) ? 1 : ((a < T (0)) ? -1 : 0);
}

template <class T, class Q>
IMATH_HOSTDEVICE constexpr inline T
lerp (T a, T b, Q t) IMATH_NOEXCEPT
{
    return (T) (a * (1 - t) + b * t);
}

template <class T, class Q>
IMATH_HOSTDEVICE constexpr inline T
ulerp (T a, T b, Q t) IMATH_NOEXCEPT
{
    return (T) ((a > b) ? (a - (a - b) * t) : (a + (b - a) * t));
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline T
lerpfactor (T m, T a, T b) IMATH_NOEXCEPT
{
    //
    // Return how far m is between a and b, that is return t such that
    // if:
    //     t = lerpfactor(m, a, b);
    // then:
    //     m = lerp(a, b, t);
    //
    // If a==b, return 0.
    //

    T d = b - a;
    T n = m - a;

    if (abs (d) > T (1) || abs (n) < std::numeric_limits<T>::max() * abs (d))
        return n / d;

    return T (0);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline T
clamp (T a, T l, T h) IMATH_NOEXCEPT
{
    return (a < l) ? l : ((a > h) ? h : a);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline int
cmp (T a, T b) IMATH_NOEXCEPT
{
    return IMATH_INTERNAL_NAMESPACE::sign (a - b);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline int
cmpt (T a, T b, T t) IMATH_NOEXCEPT
{
    return (IMATH_INTERNAL_NAMESPACE::abs (a - b) <= t) ? 0 : cmp (a, b);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline bool
iszero (T a, T t) IMATH_NOEXCEPT
{
    return (IMATH_INTERNAL_NAMESPACE::abs (a) <= t) ? 1 : 0;
}

template <class T1, class T2, class T3>
IMATH_HOSTDEVICE constexpr inline bool
equal (T1 a, T2 b, T3 t) IMATH_NOEXCEPT
{
    return IMATH_INTERNAL_NAMESPACE::abs (a - b) <= t;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline int
floor (T x) IMATH_NOEXCEPT
{
    return (x >= 0) ? int (x) : -(int (-x) + (-x > int (-x)));
}

template <class T>
IMATH_HOSTDEVICE constexpr inline int
ceil (T x) IMATH_NOEXCEPT
{
    return -floor (-x);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline int
trunc (T x) IMATH_NOEXCEPT
{
    return (x >= 0) ? int (x) : -int (-x);
}

//
// Integer division and remainder where the
// remainder of x/y has the same sign as x:
//
//	divs(x,y) == (abs(x) / abs(y)) * (sign(x) * sign(y))
//	mods(x,y) == x - y * divs(x,y)
//

IMATH_HOSTDEVICE constexpr inline int
divs (int x, int y) IMATH_NOEXCEPT
{
    return (x >= 0) ? ((y >= 0) ? (x / y) : -(x / -y)) : ((y >= 0) ? -(-x / y) : (-x / -y));
}

IMATH_HOSTDEVICE constexpr inline int
mods (int x, int y) IMATH_NOEXCEPT
{
    return (x >= 0) ? ((y >= 0) ? (x % y) : (x % -y)) : ((y >= 0) ? -(-x % y) : -(-x % -y));
}

//
// Integer division and remainder where the
// remainder of x/y is always positive:
//
//	divp(x,y) == floor (double(x) / double (y))
//	modp(x,y) == x - y * divp(x,y)
//

IMATH_HOSTDEVICE constexpr inline int
divp (int x, int y) IMATH_NOEXCEPT
{
    return (x >= 0) ? ((y >= 0) ? (x / y) : -(x / -y))
                    : ((y >= 0) ? -((y - 1 - x) / y) : ((-y - 1 - x) / -y));
}

IMATH_HOSTDEVICE constexpr inline int
modp (int x, int y) IMATH_NOEXCEPT
{
    return x - y * divp (x, y);
}

//----------------------------------------------------------
// Successor and predecessor for floating-point numbers:
//
// succf(f)     returns float(f+e), where e is the smallest
//              positive number such that float(f+e) != f.
//
// predf(f)     returns float(f-e), where e is the smallest
//              positive number such that float(f-e) != f.
//
// succd(d)     returns double(d+e), where e is the smallest
//              positive number such that double(d+e) != d.
//
// predd(d)     returns double(d-e), where e is the smallest
//              positive number such that double(d-e) != d.
//
// Exceptions:  If the input value is an infinity or a nan,
//              succf(), predf(), succd(), and predd() all
//              return the input value without changing it.
//
//----------------------------------------------------------

IMATH_EXPORT float succf (float f) IMATH_NOEXCEPT;
IMATH_EXPORT float predf (float f) IMATH_NOEXCEPT;

IMATH_EXPORT double succd (double d) IMATH_NOEXCEPT;
IMATH_EXPORT double predd (double d) IMATH_NOEXCEPT;

//
// Return true if the number is not a NaN or Infinity.
//

IMATH_HOSTDEVICE inline bool
finitef (float f) IMATH_NOEXCEPT
{
    union
    {
        float f;
        int i;
    } u;
    u.f = f;

    return (u.i & 0x7f800000) != 0x7f800000;
}

IMATH_HOSTDEVICE inline bool
finited (double d) IMATH_NOEXCEPT
{
    union
    {
        double d;
        uint64_t i;
    } u;
    u.d = d;

    return (u.i & 0x7ff0000000000000LL) != 0x7ff0000000000000LL;
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHFUN_H
