//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Algorithms applied to or in conjunction with points (Imath::Vec2
// and Imath::Vec3).
//
// The assumption made is that these functions are called much
// less often than the basic point functions or these functions
// require more support classes.
//

#ifndef INCLUDED_IMATHVECALGO_H
#define INCLUDED_IMATHVECALGO_H

#include "ImathNamespace.h"
#include "ImathVec.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

/// @cond Doxygen_Suppress
//
//  Note: doxygen doesn't understand these templates, so omit these
//  functions from the docs.
//

/// Find the projection of vector `t` onto vector `s` (`Vec2`, `Vec3`, `Vec4`)
///
/// Only defined for floating-point types, e.g. `V2f`, `V3d`, etc.
template <class Vec,
          IMATH_ENABLE_IF(!std::is_integral<typename Vec::BaseType>::value)>
IMATH_CONSTEXPR14 inline Vec
project (const Vec& s, const Vec& t) IMATH_NOEXCEPT
{
    Vec sNormalized = s.normalized();
    return sNormalized * (sNormalized ^ t);
}

/// Find a vector that is perpendicular to `s` and
/// in the same plane as `s` and `t` (`Vec2`, `Vec3`, `Vec4`)
///
/// Only defined for floating-point types, e.g. `V2f`, `V3d`, etc.
template <class Vec,
          IMATH_ENABLE_IF(!std::is_integral<typename Vec::BaseType>::value)>
constexpr inline Vec
orthogonal (const Vec& s, const Vec& t) IMATH_NOEXCEPT
{
    return t - project (s, t);
}

/// Find the direction of a ray `s` after reflection
/// off a plane with normal `t` (`Vec2`, `Vec3`, `Vec4`)
///
/// Only defined for floating-point types, e.g. `V2f`, `V3d`, etc.
template <class Vec,
          IMATH_ENABLE_IF(!std::is_integral<typename Vec::BaseType>::value)>
constexpr inline Vec
reflect (const Vec& s, const Vec& t) IMATH_NOEXCEPT
{
    return s - typename Vec::BaseType (2) * (s - project (t, s));
}

/// @endcond

/// Find the vertex of triangle `(v0, v1, v2)` that is closest to point `p`
/// (`Vec2`, `Vec3`, `Vec4`)
template <class Vec>
IMATH_CONSTEXPR14 Vec
closestVertex (const Vec& v0, const Vec& v1, const Vec& v2, const Vec& p) IMATH_NOEXCEPT
{
    Vec nearest                    = v0;
    typename Vec::BaseType neardot = (v0 - p).length2();
    typename Vec::BaseType tmp     = (v1 - p).length2();

    if (tmp < neardot)
    {
        neardot = tmp;
        nearest = v1;
    }

    tmp = (v2 - p).length2();

    if (tmp < neardot)
    {
        neardot = tmp;
        nearest = v2;
    }

    return nearest;
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHVECALGO_H
