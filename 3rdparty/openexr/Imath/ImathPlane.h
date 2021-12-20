//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// A 3D plane class template
//

#ifndef INCLUDED_IMATHPLANE_H
#define INCLUDED_IMATHPLANE_H

#include "ImathExport.h"
#include "ImathNamespace.h"

#include "ImathLine.h"
#include "ImathVec.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

///
/// The `Plane3` class represents a half space in 3D, so the normal
/// may point either towards or away from origin.  The plane `P` can
/// be represented by Plane3 as either `p` or `-p` corresponding to
/// the two half-spaces on either side of the plane. Any function
/// which computes a distance will return either negative or positive
/// values for the distance indicating which half-space the point is
/// in. Note that reflection, and intersection functions will operate
/// as expected.

template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Plane3
{
  public:

    /// @{
    /// @name Direct access to member fields
    
    /// The normal to the plane
    Vec3<T> normal;
    
    /// The distance from the origin to the plane
    T distance;

    /// @}

    /// @{
    ///	@name Constructors

    /// Uninitialized by default
    IMATH_HOSTDEVICE Plane3() IMATH_NOEXCEPT {}

    /// Initialize with a normal and distance
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Plane3 (const Vec3<T>& normal, T distance) IMATH_NOEXCEPT;

    /// Initialize with a point and a normal
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Plane3 (const Vec3<T>& point, const Vec3<T>& normal) IMATH_NOEXCEPT;
    
    /// Initialize with three points
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Plane3 (const Vec3<T>& point1,
                                               const Vec3<T>& point2,
                                               const Vec3<T>& point3) IMATH_NOEXCEPT;

    /// @}
    
    /// @{
    /// @name Manipulation
    
    /// Set via a given normal and distance
    IMATH_HOSTDEVICE void set (const Vec3<T>& normal, T distance) IMATH_NOEXCEPT;

    /// Set via a given point and normal
    IMATH_HOSTDEVICE void set (const Vec3<T>& point, const Vec3<T>& normal) IMATH_NOEXCEPT;

    /// Set via three points
    IMATH_HOSTDEVICE void set (const Vec3<T>& point1, const Vec3<T>& point2, const Vec3<T>& point3) IMATH_NOEXCEPT;

    /// @}
    
    /// @{
    /// @name Utility Methods
    
    /// Determine if a line intersects the plane.
    /// @param line The line
    /// @param[out] intersection The point of intersection
    /// @return True if the line intersects the plane.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool
    intersect (const Line3<T>& line, Vec3<T>& intersection) const IMATH_NOEXCEPT;

    /// Determine if a line intersects the plane.
    /// @param line The line
    /// @param[out] parameter The parametric value of the point of intersection
    /// @return True if the line intersects the plane.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool intersectT (const Line3<T>& line, T& parameter) const IMATH_NOEXCEPT;

    /// Return the distance from a point to the plane.
    IMATH_HOSTDEVICE constexpr T distanceTo (const Vec3<T>& point) const IMATH_NOEXCEPT;

    /// Reflect the given point around the plane.
    IMATH_HOSTDEVICE constexpr Vec3<T> reflectPoint (const Vec3<T>& point) const IMATH_NOEXCEPT;

    /// Reflect the direction vector around the plane
    IMATH_HOSTDEVICE constexpr Vec3<T> reflectVector (const Vec3<T>& vec) const IMATH_NOEXCEPT;

    /// @}
};

/// Plane of type float
typedef Plane3<float> Plane3f;

/// Plane of type double
typedef Plane3<double> Plane3d;

//---------------
// Implementation
//---------------

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Plane3<T>::Plane3 (const Vec3<T>& p0, const Vec3<T>& p1, const Vec3<T>& p2) IMATH_NOEXCEPT
{
    set (p0, p1, p2);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Plane3<T>::Plane3 (const Vec3<T>& n, T d) IMATH_NOEXCEPT
{
    set (n, d);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Plane3<T>::Plane3 (const Vec3<T>& p, const Vec3<T>& n) IMATH_NOEXCEPT
{
    set (p, n);
}

template <class T>
IMATH_HOSTDEVICE inline void
Plane3<T>::set (const Vec3<T>& point1, const Vec3<T>& point2, const Vec3<T>& point3) IMATH_NOEXCEPT
{
    normal = (point2 - point1) % (point3 - point1);
    normal.normalize();
    distance = normal ^ point1;
}

template <class T>
IMATH_HOSTDEVICE inline void
Plane3<T>::set (const Vec3<T>& point, const Vec3<T>& n) IMATH_NOEXCEPT
{
    normal = n;
    normal.normalize();
    distance = normal ^ point;
}

template <class T>
IMATH_HOSTDEVICE inline void
Plane3<T>::set (const Vec3<T>& n, T d) IMATH_NOEXCEPT
{
    normal = n;
    normal.normalize();
    distance = d;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline T
Plane3<T>::distanceTo (const Vec3<T>& point) const IMATH_NOEXCEPT
{
    return (point ^ normal) - distance;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Vec3<T>
Plane3<T>::reflectPoint (const Vec3<T>& point) const IMATH_NOEXCEPT
{
    return normal * distanceTo (point) * -2.0 + point;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Vec3<T>
Plane3<T>::reflectVector (const Vec3<T>& v) const IMATH_NOEXCEPT
{
    return normal * (normal ^ v) * 2.0 - v;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Plane3<T>::intersect (const Line3<T>& line, Vec3<T>& point) const IMATH_NOEXCEPT
{
    T d = normal ^ line.dir;
    if (d == 0.0)
        return false;
    T t   = -((normal ^ line.pos) - distance) / d;
    point = line (t);
    return true;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Plane3<T>::intersectT (const Line3<T>& line, T& t) const IMATH_NOEXCEPT
{
    T d = normal ^ line.dir;
    if (d == 0.0)
        return false;
    t = -((normal ^ line.pos) - distance) / d;
    return true;
}

/// Stream output, as "(normal distance)"
template <class T>
std::ostream&
operator<< (std::ostream& o, const Plane3<T>& plane)
{
    return o << "(" << plane.normal << ", " << plane.distance << ")";
}

/// Transform a plane by a matrix
template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Plane3<T>
operator* (const Plane3<T>& plane, const Matrix44<T>& M) IMATH_NOEXCEPT
{
    //                        T
    //	                    -1
    //	Could also compute M    but that would suck.
    //

    Vec3<T> dir1 = Vec3<T> (1, 0, 0) % plane.normal;
    T dir1Len    = dir1 ^ dir1;

    Vec3<T> tmp = Vec3<T> (0, 1, 0) % plane.normal;
    T tmpLen    = tmp ^ tmp;

    if (tmpLen > dir1Len)
    {
        dir1    = tmp;
        dir1Len = tmpLen;
    }

    tmp    = Vec3<T> (0, 0, 1) % plane.normal;
    tmpLen = tmp ^ tmp;

    if (tmpLen > dir1Len)
    {
        dir1 = tmp;
    }

    Vec3<T> dir2  = dir1 % plane.normal;
    Vec3<T> point = plane.distance * plane.normal;

    return Plane3<T> (point * M, (point + dir2) * M, (point + dir1) * M);
}

/// Reflect the pla
template <class T>
IMATH_HOSTDEVICE constexpr inline Plane3<T>
operator- (const Plane3<T>& plane) IMATH_NOEXCEPT
{
    return Plane3<T> (-plane.normal, -plane.distance);
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHPLANE_H
