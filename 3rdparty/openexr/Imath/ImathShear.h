//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// A representation of a shear transformation
//

#ifndef INCLUDED_IMATHSHEAR_H
#define INCLUDED_IMATHSHEAR_H

#include "ImathExport.h"
#include "ImathNamespace.h"

#include "ImathMath.h"
#include "ImathVec.h"
#include <iostream>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

///
/// Shear6 class template.
///
/// A shear matrix is technically defined as having a single nonzero
/// off-diagonal element; more generally, a shear transformation is
/// defined by those off-diagonal elements, so in 3D, that means there
/// are 6 possible elements/coefficients:
///
///     | X' |   |  1  YX  ZX  0 |   | X |
///     | Y' |   | XY   1  ZY  0 |   | Y |
///     | Z' | = | XZ  YZ   1  0 | = | Z |
///     | 1  |   |  0   0   0  1 |   | 1 |
///
///     X' =      X + YX * Y + ZX * Z
///     Y' = YX * X +      Y + ZY * Z
///     Z` = XZ * X + YZ * Y +      Z
///
/// See
/// https://www.cs.drexel.edu/~david/Classes/CS430/Lectures/L-04_3DTransformations.6.pdf
///
/// Those variable elements correspond to the 6 values in a Shear6.
/// So, looking at those equations, "Shear YX", for example, means
/// that for any point transformed by that matrix, its X values will
/// have some of their Y values added.  If you're talking
/// about "Axis A has values from Axis B added to it", there are 6
/// permutations for A and B (XY, XZ, YX, YZ, ZX, ZY).
///
/// Not that Maya has only three values, which represent the
/// lower/upper (depending on column/row major) triangle of the
/// matrix.  Houdini is the same as Maya (see
/// https://www.sidefx.com/docs/houdini/props/obj.html) in this
/// respect. 
///
/// There's another way to look at it. A general affine transformation
/// in 3D has 12 degrees of freedom - 12 "available" elements in the
/// 4x4 matrix since a single row/column must be (0,0,0,1).  If you
/// add up the degrees of freedom from Maya:
/// 
/// - 3 translation
/// - 3 rotation
/// - 3 scale
/// - 3 shear
///
/// You obviously get the full 12.  So technically, the Shear6 option
/// of having all 6 shear options is overkill; Imath/Shear6 has 15
/// values for a 12-degree-of-freedom transformation.  This means that
/// any nonzero values in those last 3 shear coefficients can be
/// represented in those standard 12 degrees of freedom.  Here's a
/// python example of how to do that:
///
/// 
///     >>> import imath
///     >>> M = imath.M44f()
///     >>> s = imath.V3f()
///     >>> h = imath.V3f()
///     >>> r = imath.V3f()
///     >>> t = imath.V3f()
///     # Use Shear.YX (index 3), which is an "extra" shear value
///     >>> M.setShear((0,0,0,1,0,0))
///     M44f((1, 1, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
///     >>> M.extractSHRT(s, h, r, t)
///     1
///     >>> s
///     V3f(1.41421354, 0.707106769, 1)
///     >>> h
///     V3f(1, 0, 0)
///     >>> r
///     V3f(0, -0, 0.785398185)
///     >>> t
///     V3f(0, 0, 0)
///   
/// That shows how to decompose a transform matrix with one of those
/// "extra" shear coefficients into those standard 12 degrees of
/// freedom.  But it's not necessarily intuitive; in this case, a
/// single non-zero shear coefficient resulted in a transform that has
/// non-uniform scale, a single "standard" shear value, and some
/// rotation.
///
/// So, it would seem that any transform with those extra shear
/// values set could be translated into Maya to produce the exact same
/// transformation matrix; but doing this is probably pretty
/// undesirable, since the result would have some surprising values on
/// the other transformation attributes, despite being technically
/// correct.
///
/// This usage of "degrees of freedom" is a bit hand-wavey here;
/// having a total of 12 inputs into the construction of a standard
/// transformation matrix doesn't necessarily mean that the matrix has
/// 12 true degrees of freedom, but the standard
/// translation/rotation/scale/shear matrices have the right
/// construction to ensure that.  
/// 

template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Shear6
{
  public:

    /// @{
    /// @name Direct access to members

    T xy, xz, yz, yx, zx, zy;

    /// @}

    /// Element access
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T& operator[] (int i);

    /// Element access
    IMATH_HOSTDEVICE constexpr const T& operator[] (int i) const;

    /// @{
    /// @name Constructors and Assignment

    /// Initialize to 0
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Shear6();
    
    /// Initialize to the given XY, XZ, YZ values
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Shear6 (T XY, T XZ, T YZ);

    /// Initialize to the given XY, XZ, YZ values held in (v.x, v.y, v.z)
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Shear6 (const Vec3<T>& v);

    /// Initialize to the given XY, XZ, YZ values held in (v.x, v.y, v.z)
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Shear6 (const Vec3<S>& v);

    /// Initialize to the given (XY XZ YZ YX ZX ZY) values
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Shear6 (T XY,              
                                               T XZ,
                                               T YZ,
                                               T YX,
                                               T ZX,
                                               T ZY);

    /// Copy constructor
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Shear6 (const Shear6& h);

    /// Construct from a Shear6 object of another base type
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Shear6 (const Shear6<S>& h);

    /// Assignment
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Shear6& operator= (const Shear6& h);

    /// Assignment from vector
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Shear6& operator= (const Vec3<S>& v);

    /// Destructor
    ~Shear6() = default;

    /// @}
    
    /// @{
    /// @name Compatibility with Sb

    /// Set the value
    template <class S> IMATH_HOSTDEVICE void setValue (S XY, S XZ, S YZ, S YX, S ZX, S ZY);

    /// Set the value
    template <class S> IMATH_HOSTDEVICE void setValue (const Shear6<S>& h);

    /// Return the values
    template <class S>
    IMATH_HOSTDEVICE void getValue (S& XY, S& XZ, S& YZ, S& YX, S& ZX, S& ZY) const;

    /// Return the value in `h`
    template <class S> IMATH_HOSTDEVICE void getValue (Shear6<S>& h) const;

    /// Return a raw pointer to the array of values
    IMATH_HOSTDEVICE T* getValue();

    /// Return a raw pointer to the array of values
    IMATH_HOSTDEVICE const T* getValue() const;

    /// @}

    /// @{
    /// @name Arithmetic and Comparison
    
    /// Equality
    template <class S> IMATH_HOSTDEVICE constexpr bool operator== (const Shear6<S>& h) const;

    /// Inequality
    template <class S> IMATH_HOSTDEVICE constexpr bool operator!= (const Shear6<S>& h) const;

    /// Compare two shears and test if they are "approximately equal":
    /// @return True if the coefficients of this and h are the same with
    ///	an absolute error of no more than e, i.e., for all i
    ///     abs (this[i] - h[i]) <= e
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool equalWithAbsError (const Shear6<T>& h, T e) const;

    /// Compare two shears and test if they are "approximately equal":
    /// @return True if the coefficients of this and h are the same with
    /// a relative error of no more than e, i.e., for all i
    ///     abs (this[i] - h[i]) <= e * abs (this[i])
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool equalWithRelError (const Shear6<T>& h, T e) const;

    /// Component-wise addition
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Shear6& operator+= (const Shear6& h);

    /// Component-wise addition
    IMATH_HOSTDEVICE constexpr Shear6 operator+ (const Shear6& h) const;

    /// Component-wise subtraction
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Shear6& operator-= (const Shear6& h);

    /// Component-wise subtraction
    IMATH_HOSTDEVICE constexpr Shear6 operator- (const Shear6& h) const;

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE constexpr Shear6 operator-() const;

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Shear6& negate();

    /// Component-wise multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Shear6& operator*= (const Shear6& h);
    /// Scalar multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Shear6& operator*= (T a);

    /// Component-wise multiplication
    IMATH_HOSTDEVICE constexpr Shear6 operator* (const Shear6& h) const;

    /// Scalar multiplication
    IMATH_HOSTDEVICE constexpr Shear6 operator* (T a) const;

    /// Component-wise division
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Shear6& operator/= (const Shear6& h);

    /// Scalar division
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Shear6& operator/= (T a);

    /// Component-wise division
    IMATH_HOSTDEVICE constexpr Shear6 operator/ (const Shear6& h) const;

    /// Scalar division
    IMATH_HOSTDEVICE constexpr Shear6 operator/ (T a) const;

    /// @}
    
    /// @{
    /// @name Numerical Limits
    
    /// Largest possible negative value
    IMATH_HOSTDEVICE constexpr static T baseTypeLowest() IMATH_NOEXCEPT { return std::numeric_limits<T>::lowest(); }

    /// Largest possible positive value
    IMATH_HOSTDEVICE constexpr static T baseTypeMax() IMATH_NOEXCEPT { return std::numeric_limits<T>::max(); }

    /// Smallest possible positive value
    IMATH_HOSTDEVICE constexpr static T baseTypeSmallest() IMATH_NOEXCEPT { return std::numeric_limits<T>::min(); }

    /// Smallest possible e for which 1+e != 1
    IMATH_HOSTDEVICE constexpr static T baseTypeEpsilon() IMATH_NOEXCEPT { return std::numeric_limits<T>::epsilon(); }

    /// @}

    /// Return the number of dimensions, i.e. 6
    IMATH_HOSTDEVICE constexpr static unsigned int dimensions() { return 6; }

    /// The base type: In templates that accept a parameter `V` (could
    /// be a Color4), you can refer to `T` as `V::BaseType`
    typedef T BaseType;
};

/// Stream output, as "(xy xz yz yx zx zy)"
template <class T> std::ostream& operator<< (std::ostream& s, const Shear6<T>& h);

/// Reverse multiplication: scalar * Shear6<T>
template <class S, class T>
IMATH_HOSTDEVICE constexpr Shear6<T> operator* (S a, const Shear6<T>& h);

/// 3D shear of type float
typedef Vec3<float> Shear3f;

/// 3D shear of type double
typedef Vec3<double> Shear3d;

/// Shear6 of type float
typedef Shear6<float> Shear6f;

/// Shear6 of type double
typedef Shear6<double> Shear6d;

//-----------------------
// Implementation of Shear6
//-----------------------

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline T&
Shear6<T>::operator[] (int i)
{
    return (&xy)[i]; // NOSONAR - suppress SonarCloud bug report.
}

template <class T>
IMATH_HOSTDEVICE constexpr inline const T&
Shear6<T>::operator[] (int i) const
{
    return (&xy)[i]; // NOSONAR - suppress SonarCloud bug report.
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Shear6<T>::Shear6()
{
    xy = xz = yz = yx = zx = zy = 0;
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Shear6<T>::Shear6 (T XY, T XZ, T YZ)
{
    xy = XY;
    xz = XZ;
    yz = YZ;
    yx = 0;
    zx = 0;
    zy = 0;
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Shear6<T>::Shear6 (const Vec3<T>& v)
{
    xy = v.x;
    xz = v.y;
    yz = v.z;
    yx = 0;
    zx = 0;
    zy = 0;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Shear6<T>::Shear6 (const Vec3<S>& v)
{
    xy = T (v.x);
    xz = T (v.y);
    yz = T (v.z);
    yx = 0;
    zx = 0;
    zy = 0;
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Shear6<T>::Shear6 (T XY, T XZ, T YZ, T YX, T ZX, T ZY)
{
    xy = XY;
    xz = XZ;
    yz = YZ;
    yx = YX;
    zx = ZX;
    zy = ZY;
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Shear6<T>::Shear6 (const Shear6& h)
{
    xy = h.xy;
    xz = h.xz;
    yz = h.yz;
    yx = h.yx;
    zx = h.zx;
    zy = h.zy;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Shear6<T>::Shear6 (const Shear6<S>& h)
{
    xy = T (h.xy);
    xz = T (h.xz);
    yz = T (h.yz);
    yx = T (h.yx);
    zx = T (h.zx);
    zy = T (h.zy);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Shear6<T>&
Shear6<T>::operator= (const Shear6& h)
{
    xy = h.xy;
    xz = h.xz;
    yz = h.yz;
    yx = h.yx;
    zx = h.zx;
    zy = h.zy;
    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Shear6<T>&
Shear6<T>::operator= (const Vec3<S>& v)
{
    xy = T (v.x);
    xz = T (v.y);
    yz = T (v.z);
    yx = 0;
    zx = 0;
    zy = 0;
    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Shear6<T>::setValue (S XY, S XZ, S YZ, S YX, S ZX, S ZY)
{
    xy = T (XY);
    xz = T (XZ);
    yz = T (YZ);
    yx = T (YX);
    zx = T (ZX);
    zy = T (ZY);
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Shear6<T>::setValue (const Shear6<S>& h)
{
    xy = T (h.xy);
    xz = T (h.xz);
    yz = T (h.yz);
    yx = T (h.yx);
    zx = T (h.zx);
    zy = T (h.zy);
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Shear6<T>::getValue (S& XY, S& XZ, S& YZ, S& YX, S& ZX, S& ZY) const
{
    XY = S (xy);
    XZ = S (xz);
    YZ = S (yz);
    YX = S (yx);
    ZX = S (zx);
    ZY = S (zy);
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Shear6<T>::getValue (Shear6<S>& h) const
{
    h.xy = S (xy);
    h.xz = S (xz);
    h.yz = S (yz);
    h.yx = S (yx);
    h.zx = S (zx);
    h.zy = S (zy);
}

template <class T>
IMATH_HOSTDEVICE inline T*
Shear6<T>::getValue()
{
    return (T*) &xy;
}

template <class T>
IMATH_HOSTDEVICE inline const T*
Shear6<T>::getValue() const
{
    return (const T*) &xy;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE constexpr inline bool
Shear6<T>::operator== (const Shear6<S>& h) const
{
    return xy == h.xy && xz == h.xz && yz == h.yz && yx == h.yx && zx == h.zx && zy == h.zy;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE constexpr inline bool
Shear6<T>::operator!= (const Shear6<S>& h) const
{
    return xy != h.xy || xz != h.xz || yz != h.yz || yx != h.yx || zx != h.zx || zy != h.zy;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Shear6<T>::equalWithAbsError (const Shear6<T>& h, T e) const
{
    for (int i = 0; i < 6; i++)
        if (!IMATH_INTERNAL_NAMESPACE::equalWithAbsError ((*this)[i], h[i], e))
            return false;

    return true;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Shear6<T>::equalWithRelError (const Shear6<T>& h, T e) const
{
    for (int i = 0; i < 6; i++)
        if (!IMATH_INTERNAL_NAMESPACE::equalWithRelError ((*this)[i], h[i], e))
            return false;

    return true;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Shear6<T>&
Shear6<T>::operator+= (const Shear6& h)
{
    xy += h.xy;
    xz += h.xz;
    yz += h.yz;
    yx += h.yx;
    zx += h.zx;
    zy += h.zy;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Shear6<T>
Shear6<T>::operator+ (const Shear6& h) const
{
    return Shear6 (xy + h.xy, xz + h.xz, yz + h.yz, yx + h.yx, zx + h.zx, zy + h.zy);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Shear6<T>&
Shear6<T>::operator-= (const Shear6& h)
{
    xy -= h.xy;
    xz -= h.xz;
    yz -= h.yz;
    yx -= h.yx;
    zx -= h.zx;
    zy -= h.zy;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Shear6<T>
Shear6<T>::operator- (const Shear6& h) const
{
    return Shear6 (xy - h.xy, xz - h.xz, yz - h.yz, yx - h.yx, zx - h.zx, zy - h.zy);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Shear6<T>
Shear6<T>::operator-() const
{
    return Shear6 (-xy, -xz, -yz, -yx, -zx, -zy);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Shear6<T>&
Shear6<T>::negate()
{
    xy = -xy;
    xz = -xz;
    yz = -yz;
    yx = -yx;
    zx = -zx;
    zy = -zy;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Shear6<T>&
Shear6<T>::operator*= (const Shear6& h)
{
    xy *= h.xy;
    xz *= h.xz;
    yz *= h.yz;
    yx *= h.yx;
    zx *= h.zx;
    zy *= h.zy;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Shear6<T>&
Shear6<T>::operator*= (T a)
{
    xy *= a;
    xz *= a;
    yz *= a;
    yx *= a;
    zx *= a;
    zy *= a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Shear6<T>
Shear6<T>::operator* (const Shear6& h) const
{
    return Shear6 (xy * h.xy, xz * h.xz, yz * h.yz, yx * h.yx, zx * h.zx, zy * h.zy);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Shear6<T>
Shear6<T>::operator* (T a) const
{
    return Shear6 (xy * a, xz * a, yz * a, yx * a, zx * a, zy * a);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Shear6<T>&
Shear6<T>::operator/= (const Shear6& h)
{
    xy /= h.xy;
    xz /= h.xz;
    yz /= h.yz;
    yx /= h.yx;
    zx /= h.zx;
    zy /= h.zy;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Shear6<T>&
Shear6<T>::operator/= (T a)
{
    xy /= a;
    xz /= a;
    yz /= a;
    yx /= a;
    zx /= a;
    zy /= a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Shear6<T>
Shear6<T>::operator/ (const Shear6& h) const
{
    return Shear6 (xy / h.xy, xz / h.xz, yz / h.yz, yx / h.yx, zx / h.zx, zy / h.zy);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Shear6<T>
Shear6<T>::operator/ (T a) const
{
    return Shear6 (xy / a, xz / a, yz / a, yx / a, zx / a, zy / a);
}

//-----------------------------
// Stream output implementation
//-----------------------------

template <class T>
std::ostream&
operator<< (std::ostream& s, const Shear6<T>& h)
{
    return s << '(' << h.xy << ' ' << h.xz << ' ' << h.yz << h.yx << ' ' << h.zx << ' ' << h.zy
             << ')';
}

//-----------------------------------------
// Implementation of reverse multiplication
//-----------------------------------------

template <class S, class T>
IMATH_HOSTDEVICE constexpr inline Shear6<T>
operator* (S a, const Shear6<T>& h)
{
    return Shear6<T> (a * h.xy, a * h.xz, a * h.yz, a * h.yx, a * h.zx, a * h.zy);
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHSHEAR_H
