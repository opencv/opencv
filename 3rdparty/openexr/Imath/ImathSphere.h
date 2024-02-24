//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// A 3D sphere class template
//

#ifndef INCLUDED_IMATHSPHERE_H
#define INCLUDED_IMATHSPHERE_H

#include "ImathExport.h"
#include "ImathNamespace.h"

#include "ImathBox.h"
#include "ImathLine.h"
#include "ImathVec.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

///
/// A 3D sphere
///

template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Sphere3
{
  public:

    /// @{
    /// @name Direct access to member fields
    
    /// Center
    Vec3<T> center;

    /// Radius
    T radius;

    /// @}

    /// @{
    ///	@name Constructors

    /// Default is center at (0,0,0) and radius of 0.
    IMATH_HOSTDEVICE constexpr Sphere3() : center (0, 0, 0), radius (0) {}

    /// Initialize to a given center and radius
    IMATH_HOSTDEVICE constexpr Sphere3 (const Vec3<T>& c, T r) : center (c), radius (r) {}

    /// @}
    
    /// @{
    /// @name Manipulation
    
    ///	Set the center and radius of the sphere so that it tightly
    ///	encloses Box b.
    IMATH_HOSTDEVICE void circumscribe (const Box<Vec3<T>>& box);

    /// @}
    
    /// @{
    /// @name Utility Methods
    
    ///	If the sphere and line `l` intersect, then compute the
    /// smallest `t` with `t>=0` so that `l(t)` is a point on the sphere.
    ///
    /// @param[in] l The line
    /// @param[out] intersection The point of intersection
    /// @return True if the sphere and line intersect, false if they
    ///	do not.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool
    intersect (const Line3<T>& l, Vec3<T>& intersection) const;

    ///	If the sphere and line `l` intersect, then compute the
    /// smallest `t` with `t>=0` so that `l(t)` is a point on the sphere.
    ///
    /// @param[in] l The line
    /// @param[out] t The parameter of the line at the intersection point
    /// @return True if the sphere and line intersect, false if they
    ///	do not.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool intersectT (const Line3<T>& l, T& t) const;

    /// @}
};

/// Sphere of type float
typedef Sphere3<float> Sphere3f;

/// Sphere of type double
typedef Sphere3<double> Sphere3d;

//---------------
// Implementation
//---------------

template <class T>
IMATH_HOSTDEVICE inline void
Sphere3<T>::circumscribe (const Box<Vec3<T>>& box)
{
    center = T (0.5) * (box.min + box.max);
    radius = (box.max - center).length();
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool
Sphere3<T>::intersectT (const Line3<T>& line, T& t) const
{
    bool doesIntersect = true;

    Vec3<T> v = line.pos - center;
    T B       = T (2.0) * (line.dir ^ v);
    T C       = (v ^ v) - (radius * radius);

    // compute discriminant
    // if negative, there is no intersection

    T discr = B * B - T (4.0) * C;

    if (discr < 0.0)
    {
        // line and Sphere3 do not intersect

        doesIntersect = false;
    }
    else
    {
        // t0: (-B - sqrt(B^2 - 4AC)) / 2A  (A = 1)

        T sqroot = std::sqrt (discr);
        t        = (-B - sqroot) * T (0.5);

        if (t < 0.0)
        {
            // no intersection, try t1: (-B + sqrt(B^2 - 4AC)) / 2A  (A = 1)

            t = (-B + sqroot) * T (0.5);
        }

        if (t < 0.0)
            doesIntersect = false;
    }

    return doesIntersect;
}

template <class T>
IMATH_CONSTEXPR14 bool
Sphere3<T>::intersect (const Line3<T>& line, Vec3<T>& intersection) const
{
    T t (0);

    if (intersectT (line, t))
    {
        intersection = line (t);
        return true;
    }
    else
    {
        return false;
    }
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHSPHERE_H
