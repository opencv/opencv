//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// A 3D line class template
//

#ifndef INCLUDED_IMATHLINE_H
#define INCLUDED_IMATHLINE_H

#include "ImathMatrix.h"
#include "ImathNamespace.h"
#include "ImathVec.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

///
/// The `Line3` class represents a 3D line, defined by a point and a
/// direction vector.
///

template <class T> class Line3
{
  public:

    /// @{
    /// @name Direct access to member fields
    
    /// A point on the line
    Vec3<T> pos;

    /// The direction of the line
    Vec3<T> dir;

    /// @}

    /// @{
    ///	@name Constructors

    /// Uninitialized by default
    IMATH_HOSTDEVICE constexpr Line3() IMATH_NOEXCEPT {}

    /// Initialize with two points. The direction is the difference
    /// between the points.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Line3 (const Vec3<T>& point1, const Vec3<T>& point2) IMATH_NOEXCEPT;

    /// @}
    
    /// @{
    /// @name Manipulation
    
    /// Set the line defined by two points. The direction is the difference
    /// between the points.
    IMATH_HOSTDEVICE void set (const Vec3<T>& point1, const Vec3<T>& point2) IMATH_NOEXCEPT;

    /// @}

    /// @{
    /// @name Utility Methods
    
    /// Return the point on the line at the given parameter value,
    ///	e.g. L(t)
    IMATH_HOSTDEVICE constexpr Vec3<T> operator() (T parameter) const IMATH_NOEXCEPT;

    /// Return the distance to the given point
    IMATH_HOSTDEVICE constexpr T distanceTo (const Vec3<T>& point) const IMATH_NOEXCEPT;
    /// Return the distance to the given line
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T distanceTo (const Line3<T>& line) const IMATH_NOEXCEPT;

    /// Return the point on the line closest to the given point
    IMATH_HOSTDEVICE constexpr Vec3<T> closestPointTo (const Vec3<T>& point) const IMATH_NOEXCEPT;

    /// Return the point on the line closest to the given line
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Vec3<T> closestPointTo (const Line3<T>& line) const IMATH_NOEXCEPT;

    /// @}
};

/// Line of type float
typedef Line3<float> Line3f;

/// Line of type double
typedef Line3<double> Line3d;

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Line3<T>::Line3 (const Vec3<T>& p0, const Vec3<T>& p1) IMATH_NOEXCEPT
{
    set (p0, p1);
}

template <class T>
IMATH_HOSTDEVICE inline void
Line3<T>::set (const Vec3<T>& p0, const Vec3<T>& p1) IMATH_NOEXCEPT
{
    pos = p0;
    dir = p1 - p0;
    dir.normalize();
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Vec3<T>
Line3<T>::operator() (T parameter) const IMATH_NOEXCEPT
{
    return pos + dir * parameter;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline T
Line3<T>::distanceTo (const Vec3<T>& point) const IMATH_NOEXCEPT
{
    return (closestPointTo (point) - point).length();
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Vec3<T>
Line3<T>::closestPointTo (const Vec3<T>& point) const IMATH_NOEXCEPT
{
    return ((point - pos) ^ dir) * dir + pos;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline T
Line3<T>::distanceTo (const Line3<T>& line) const IMATH_NOEXCEPT
{
    T d = (dir % line.dir) ^ (line.pos - pos);
    return (d >= 0) ? d : -d;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Vec3<T>
Line3<T>::closestPointTo (const Line3<T>& line) const IMATH_NOEXCEPT
{
    // Assumes the lines are normalized

    Vec3<T> posLpos = pos - line.pos;
    T c             = dir ^ posLpos;
    T a             = line.dir ^ dir;
    T f             = line.dir ^ posLpos;
    T num           = c - a * f;

    T denom = a * a - 1;

    T absDenom = ((denom >= 0) ? denom : -denom);

    if (absDenom < 1)
    {
        T absNum = ((num >= 0) ? num : -num);

        if (absNum >= absDenom * std::numeric_limits<T>::max())
            return pos;
    }

    return pos + dir * (num / denom);
}

/// Stream output, as "(pos dir)"
template <class T>
std::ostream&
operator<< (std::ostream& o, const Line3<T>& line)
{
    return o << "(" << line.pos << ", " << line.dir << ")";
}

/// Transform a line by a matrix
template <class S, class T>
IMATH_HOSTDEVICE constexpr inline Line3<S>
operator* (const Line3<S>& line, const Matrix44<T>& M) IMATH_NOEXCEPT
{
    return Line3<S> (line.pos * M, (line.pos + line.dir) * M);
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHLINE_H
