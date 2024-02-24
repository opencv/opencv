//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Algorithms applied to or in conjunction with Imath::Line class
//

#ifndef INCLUDED_IMATHLINEALGO_H
#define INCLUDED_IMATHLINEALGO_H

#include "ImathFun.h"
#include "ImathLine.h"
#include "ImathNamespace.h"
#include "ImathVecAlgo.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

///
/// Compute point1 and point2 such that point1 is on line1, point2
/// is on line2 and the distance between point1 and point2 is minimal.
///
/// This function returns true if point1 and point2 can be computed,
/// or false if line1 and line2 are parallel or nearly parallel.
/// This function assumes that line1.dir and line2.dir are normalized.
///

template <class T>
IMATH_CONSTEXPR14 bool
closestPoints (const Line3<T>& line1, const Line3<T>& line2, Vec3<T>& point1, Vec3<T>& point2) IMATH_NOEXCEPT
{
    Vec3<T> w = line1.pos - line2.pos;
    T d1w     = line1.dir ^ w;
    T d2w     = line2.dir ^ w;
    T d1d2    = line1.dir ^ line2.dir;
    T n1      = d1d2 * d2w - d1w;
    T n2      = d2w - d1d2 * d1w;
    T d       = 1 - d1d2 * d1d2;
    T absD    = abs (d);

    if ((absD > 1) || (abs (n1) < std::numeric_limits<T>::max() * absD && abs (n2) < std::numeric_limits<T>::max() * absD))
    {
        point1 = line1 (n1 / d);
        point2 = line2 (n2 / d);
        return true;
    }
    else
    {
        return false;
    }
}

///
/// Given a line and a triangle (v0, v1, v2), the intersect() function
/// finds the intersection of the line and the plane that contains the
/// triangle.
///
/// If the intersection point cannot be computed, either because the
/// line and the triangle's plane are nearly parallel or because the
/// triangle's area is very small, intersect() returns false.
///
/// If the intersection point is outside the triangle, intersect
/// returns false.
///
/// If the intersection point, pt, is inside the triangle, intersect()
/// computes a front-facing flag and the barycentric coordinates of
/// the intersection point, and returns true.
///
/// The front-facing flag is true if the dot product of the triangle's
/// normal, (v2-v1)%(v1-v0), and the line's direction is negative.
///
/// The barycentric coordinates have the following property:
///
///     pt = v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z
///

template <class T>
IMATH_CONSTEXPR14 bool
intersect (const Line3<T>& line,
           const Vec3<T>& v0,
           const Vec3<T>& v1,
           const Vec3<T>& v2,
           Vec3<T>& pt,
           Vec3<T>& barycentric,
           bool& front) IMATH_NOEXCEPT
{
    Vec3<T> edge0  = v1 - v0;
    Vec3<T> edge1  = v2 - v1;
    Vec3<T> normal = edge1 % edge0;

    T l = normal.length();

    if (l != 0)
        normal /= l;
    else
        return false; // zero-area triangle

    //
    // d is the distance of line.pos from the plane that contains the triangle.
    // The intersection point is at line.pos + (d/nd) * line.dir.
    //

    T d  = normal ^ (v0 - line.pos);
    T nd = normal ^ line.dir;

    if (abs (nd) > 1 || abs (d) < std::numeric_limits<T>::max() * abs (nd))
        pt = line (d / nd);
    else
        return false; // line and plane are nearly parallel

    //
    // Compute the barycentric coordinates of the intersection point.
    // The intersection is inside the triangle if all three barycentric
    // coordinates are between zero and one.
    //

    {
        Vec3<T> en = edge0.normalized();
        Vec3<T> a  = pt - v0;
        Vec3<T> b  = v2 - v0;
        Vec3<T> c  = (a - en * (en ^ a));
        Vec3<T> d  = (b - en * (en ^ b));
        T e        = c ^ d;
        T f        = d ^ d;

        if (e >= 0 && e <= f)
            barycentric.z = e / f;
        else
            return false; // outside
    }

    {
        Vec3<T> en = edge1.normalized();
        Vec3<T> a  = pt - v1;
        Vec3<T> b  = v0 - v1;
        Vec3<T> c  = (a - en * (en ^ a));
        Vec3<T> d  = (b - en * (en ^ b));
        T e        = c ^ d;
        T f        = d ^ d;

        if (e >= 0 && e <= f)
            barycentric.x = e / f;
        else
            return false; // outside
    }

    barycentric.y = 1 - barycentric.x - barycentric.z;

    if (barycentric.y < 0)
        return false; // outside

    front = ((line.dir ^ normal) < 0);
    return true;
}

///
/// Return the vertex that is closest to the given line. The returned
/// point is either v0, v1, or v2.
///

template <class T>
IMATH_CONSTEXPR14 Vec3<T>
closestVertex (const Vec3<T>& v0, const Vec3<T>& v1, const Vec3<T>& v2, const Line3<T>& l) IMATH_NOEXCEPT
{
    Vec3<T> nearest = v0;
    T neardot       = (v0 - l.closestPointTo (v0)).length2();

    T tmp = (v1 - l.closestPointTo (v1)).length2();

    if (tmp < neardot)
    {
        neardot = tmp;
        nearest = v1;
    }

    tmp = (v2 - l.closestPointTo (v2)).length2();
    if (tmp < neardot)
    {
        neardot = tmp;
        nearest = v2;
    }

    return nearest;
}

///
/// Rotate the point p around the line l by the given angle.
///

template <class T>
IMATH_CONSTEXPR14 Vec3<T>
rotatePoint (const Vec3<T> p, Line3<T> l, T angle) IMATH_NOEXCEPT
{
    //
    // Form a coordinate frame with <x,y,a>. The rotation is the in xy
    // plane.
    //

    Vec3<T> q = l.closestPointTo (p);
    Vec3<T> x = p - q;
    T radius  = x.length();

    x.normalize();
    Vec3<T> y = (x % l.dir).normalize();

    T cosangle = std::cos (angle);
    T sinangle = std::sin (angle);

    Vec3<T> r = q + x * radius * cosangle + y * radius * sinangle;

    return r;
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHLINEALGO_H
