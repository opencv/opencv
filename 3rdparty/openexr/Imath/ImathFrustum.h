//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// A viewing frustum class
//

#ifndef INCLUDED_IMATHFRUSTUM_H
#define INCLUDED_IMATHFRUSTUM_H

#include "ImathExport.h"
#include "ImathNamespace.h"

#include "ImathFun.h"
#include "ImathLine.h"
#include "ImathMatrix.h"
#include "ImathPlane.h"
#include "ImathVec.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

///
/// Template class `Frustum<T>`
///
/// The frustum is always located with the eye point at the origin
/// facing down -Z. This makes the Frustum class compatable with
/// OpenGL (or anything that assumes a camera looks down -Z, hence
/// with a right-handed coordinate system) but not with RenderMan
/// which assumes the camera looks down +Z. Additional functions are
/// provided for conversion from and from various camera coordinate
/// spaces.
///
/// nearPlane/farPlane: near/far are keywords used by Microsoft's
/// compiler, so we use nearPlane/farPlane instead to avoid
/// issues.

template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Frustum
{
  public:

    /// @{
    /// @name Constructors and Assignment
    ///

    /// Initialize with default values:
    ///  near=0.1, far=1000.0, left=-1.0, right=1.0, top=1.0, bottom=-1.0, ortho=false
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Frustum() IMATH_NOEXCEPT;

    /// Copy constructor
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Frustum (const Frustum&) IMATH_NOEXCEPT;

    /// Initialize to specific values
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14
    Frustum (T nearPlane, T farPlane, T left, T right, T top, T bottom, bool ortho = false) IMATH_NOEXCEPT;

    /// Initialize with fov and aspect 
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Frustum (T nearPlane, T farPlane, T fovx, T fovy, T aspect) IMATH_NOEXCEPT;

    /// Destructor
    virtual ~Frustum() IMATH_NOEXCEPT;

    /// Component-wise assignment
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Frustum& operator= (const Frustum&) IMATH_NOEXCEPT;

    /// @}

    /// @{
    /// @name Comparison

    /// Equality
    IMATH_HOSTDEVICE constexpr bool operator== (const Frustum<T>& src) const IMATH_NOEXCEPT;

    /// Inequality
    IMATH_HOSTDEVICE constexpr bool operator!= (const Frustum<T>& src) const IMATH_NOEXCEPT;

    /// @}
    
    /// @{
    /// @name Query
    
    /// Return true if the frustum is orthographic, false if perspective
    IMATH_HOSTDEVICE constexpr bool orthographic() const IMATH_NOEXCEPT { return _orthographic; }

    /// Return the near clipping plane
    IMATH_HOSTDEVICE constexpr T nearPlane() const IMATH_NOEXCEPT { return _nearPlane; }

    /// Return the near clipping plane
    IMATH_HOSTDEVICE constexpr T hither() const IMATH_NOEXCEPT { return _nearPlane; }

    /// Return the far clipping plane
    IMATH_HOSTDEVICE constexpr T farPlane() const IMATH_NOEXCEPT { return _farPlane; }

    /// Return the far clipping plane
    IMATH_HOSTDEVICE constexpr T yon() const IMATH_NOEXCEPT { return _farPlane; }

    /// Return the left of the frustum
    IMATH_HOSTDEVICE constexpr T left() const IMATH_NOEXCEPT { return _left; }

    /// Return the right of the frustum
    IMATH_HOSTDEVICE constexpr T right() const IMATH_NOEXCEPT { return _right; }

    /// Return the bottom of the frustum
    IMATH_HOSTDEVICE constexpr T bottom() const IMATH_NOEXCEPT { return _bottom; }

    /// Return the top of the frustum
    IMATH_HOSTDEVICE constexpr T top() const IMATH_NOEXCEPT { return _top; }

    /// Return the field of view in X
    IMATH_HOSTDEVICE constexpr T fovx() const IMATH_NOEXCEPT;

    /// Return the field of view in Y
    IMATH_HOSTDEVICE constexpr T fovy() const IMATH_NOEXCEPT;

    /// Return the aspect ratio
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T aspect() const IMATH_NOEXCEPT;

    /// Return the aspect ratio. Throw an exception if the aspect
    /// ratio is undefined.
    IMATH_CONSTEXPR14 T aspectExc() const;

    /// Return the project matrix that the frustum defines
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix44<T> projectionMatrix() const IMATH_NOEXCEPT;

    /// Return the project matrix that the frustum defines. Throw an
    /// exception if the frustum is degenerate.
    IMATH_CONSTEXPR14 Matrix44<T> projectionMatrixExc() const;

    /// Return true if the frustum is degenerate.
    IMATH_HOSTDEVICE constexpr bool degenerate() const IMATH_NOEXCEPT;

    /// @}
    
    /// @{
    /// @name Set Value
    
    /// Set functions change the entire state of the Frustum
    IMATH_HOSTDEVICE void
    set (T nearPlane, T farPlane, T left, T right, T top, T bottom, bool ortho = false) IMATH_NOEXCEPT;

    /// Set functions change the entire state of the Frustum using
    /// field of view and aspect ratio
    IMATH_HOSTDEVICE void set (T nearPlane, T farPlane, T fovx, T fovy, T aspect) IMATH_NOEXCEPT;

    /// Set functions change the entire state of the Frustum using
    /// field of view and aspect ratio. Throw an exception if `fovx`
    /// and/or `fovy` are invalid.
    void setExc (T nearPlane, T farPlane, T fovx, T fovy, T aspect);

    /// Set the near and far clipping planes
    IMATH_HOSTDEVICE void modifyNearAndFar (T nearPlane, T farPlane) IMATH_NOEXCEPT;

    /// Set the ortographic state
    IMATH_HOSTDEVICE void setOrthographic (bool) IMATH_NOEXCEPT;

    /// Set the planes in p to be the six bounding planes of the frustum, in
    /// the following order: top, right, bottom, left, near, far.
    /// Note that the planes have normals that point out of the frustum.
    IMATH_HOSTDEVICE void planes (Plane3<T> p[6]) const IMATH_NOEXCEPT;

    /// Set the planes in p to be the six bounding planes of the
    /// frustum, in the following order: top, right, bottom, left,
    /// near, far.  Note that the planes have normals that point out
    /// of the frustum.  Apply the given matrix to transform the
    /// frustum before setting the planes.
    IMATH_HOSTDEVICE void planes (Plane3<T> p[6], const Matrix44<T>& M) const IMATH_NOEXCEPT;

    /// Takes a rectangle in the screen space (i.e., -1 <= left <= right <= 1
    /// and -1 <= bottom <= top <= 1) of this Frustum, and returns a new
    /// Frustum whose near clipping-plane window is that rectangle in local
    /// space.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 IMATH_HOSTDEVICE Frustum<T>
    window (T left, T right, T top, T bottom) const IMATH_NOEXCEPT;

    /// @}

    /// @{
    /// @name Utility Methods
    
    /// Project a point in screen spaced to 3d ray
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Line3<T> projectScreenToRay (const Vec2<T>&) const IMATH_NOEXCEPT;

    /// Project a 3D point into screen coordinates
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Vec2<T> projectPointToScreen (const Vec3<T>&) const IMATH_NOEXCEPT;

    /// Project a 3D point into screen coordinates. Throw an
    /// exception if the point cannot be projected.
    IMATH_CONSTEXPR14 Vec2<T> projectPointToScreenExc (const Vec3<T>&) const;

    /// Map a z value to its depth in the frustum.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T ZToDepth (long zval,
                                                   long min,
                                                   long max) const IMATH_NOEXCEPT;
    /// Map a z value to its depth in the frustum.
    IMATH_CONSTEXPR14 T ZToDepthExc (long zval, long min, long max) const;

    /// Map a normalized z value to its depth in the frustum.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T normalizedZToDepth (T zval) const IMATH_NOEXCEPT;

    /// Map a normalized z value to its depth in the frustum. Throw an
    /// exception on error.
    IMATH_CONSTEXPR14 T normalizedZToDepthExc (T zval) const;

    /// Map depth to z value.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 long
    DepthToZ (T depth, long zmin, long zmax) const IMATH_NOEXCEPT;

    /// Map depth to z value. Throw an exception on error.
    IMATH_CONSTEXPR14 long DepthToZExc (T depth, long zmin, long zmax) const;

    /// Compute worldRadius
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T worldRadius (const Vec3<T>& p, T radius) const IMATH_NOEXCEPT;

    /// Compute worldRadius. Throw an exception on error.
    IMATH_CONSTEXPR14 T worldRadiusExc (const Vec3<T>& p, T radius) const;

    /// Compute screen radius
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T screenRadius (const Vec3<T>& p, T radius) const IMATH_NOEXCEPT;

    /// Compute screen radius. Throw an exception on error.
    IMATH_CONSTEXPR14 T screenRadiusExc (const Vec3<T>& p, T radius) const;

    /// @}
    
  protected:

    /// Map point from screen space to local space
    IMATH_HOSTDEVICE constexpr Vec2<T> screenToLocal (const Vec2<T>&) const IMATH_NOEXCEPT;

    /// Map point from local space to screen space
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Vec2<T>
    localToScreen (const Vec2<T>&) const IMATH_NOEXCEPT;

    /// Map point from local space to screen space. Throw an exception
    /// on error.
    IMATH_CONSTEXPR14 Vec2<T> localToScreenExc (const Vec2<T>&) const;

  protected:

    /// @cond Doxygen_Suppress

    T _nearPlane;
    T _farPlane;
    T _left;
    T _right;
    T _top;
    T _bottom;
    bool _orthographic;

    /// @endcond
};

template <class T> IMATH_CONSTEXPR14 inline Frustum<T>::Frustum() IMATH_NOEXCEPT
{
    set (T (0.1), T (1000.0), T (-1.0), T (1.0), T (1.0), T (-1.0), false);
}

template <class T> IMATH_CONSTEXPR14 inline Frustum<T>::Frustum (const Frustum& f) IMATH_NOEXCEPT
{
    *this = f;
}

template <class T>
IMATH_CONSTEXPR14 inline Frustum<T>::Frustum (T n, T f, T l, T r, T t, T b, bool o) IMATH_NOEXCEPT
{
    set (n, f, l, r, t, b, o);
}

template <class T>
IMATH_CONSTEXPR14 inline Frustum<T>::Frustum (T nearPlane, T farPlane, T fovx, T fovy, T aspect) IMATH_NOEXCEPT
{
    set (nearPlane, farPlane, fovx, fovy, aspect);
}

template <class T> Frustum<T>::~Frustum() IMATH_NOEXCEPT
{}

template <class T>
IMATH_CONSTEXPR14 inline const Frustum<T>&
Frustum<T>::operator= (const Frustum& f) IMATH_NOEXCEPT
{
    _nearPlane    = f._nearPlane;
    _farPlane     = f._farPlane;
    _left         = f._left;
    _right        = f._right;
    _top          = f._top;
    _bottom       = f._bottom;
    _orthographic = f._orthographic;

    return *this;
}

template <class T>
constexpr inline bool
Frustum<T>::operator== (const Frustum<T>& src) const IMATH_NOEXCEPT
{
    return _nearPlane == src._nearPlane && _farPlane == src._farPlane && _left == src._left &&
           _right == src._right && _top == src._top && _bottom == src._bottom &&
           _orthographic == src._orthographic;
}

template <class T>
constexpr inline bool
Frustum<T>::operator!= (const Frustum<T>& src) const IMATH_NOEXCEPT
{
    return !operator== (src);
}

template <class T>
inline void
Frustum<T>::set (T n, T f, T l, T r, T t, T b, bool o) IMATH_NOEXCEPT
{
    _nearPlane    = n;
    _farPlane     = f;
    _left         = l;
    _right        = r;
    _bottom       = b;
    _top          = t;
    _orthographic = o;
}

template <class T>
inline void
Frustum<T>::modifyNearAndFar (T n, T f) IMATH_NOEXCEPT
{
    if (_orthographic)
    {
        _nearPlane = n;
    }
    else
    {
        Line3<T> lowerLeft (Vec3<T> (0, 0, 0), Vec3<T> (_left, _bottom, -_nearPlane));
        Line3<T> upperRight (Vec3<T> (0, 0, 0), Vec3<T> (_right, _top, -_nearPlane));
        Plane3<T> nearPlane (Vec3<T> (0, 0, -1), n);

        Vec3<T> ll = Vec3<T> (0, 0, 0);
        Vec3<T> ur = Vec3<T> (0, 0, 0);
        nearPlane.intersect (lowerLeft, ll);
        nearPlane.intersect (upperRight, ur);

        _left      = ll.x;
        _right     = ur.x;
        _top       = ur.y;
        _bottom    = ll.y;
        _nearPlane = n;
        _farPlane  = f;
    }

    _farPlane = f;
}

template <class T>
inline void
Frustum<T>::setOrthographic (bool ortho) IMATH_NOEXCEPT
{
    _orthographic = ortho;
}

template <class T>
inline void
Frustum<T>::setExc (T nearPlane, T farPlane, T fovx, T fovy, T aspect)
{
    if (fovx != T (0) && fovy != T (0))
        throw std::domain_error ("fovx and fovy cannot both be non-zero.");

    const T two = static_cast<T> (2);

    if (fovx != T (0))
    {
        _right  = nearPlane * std::tan (fovx / two);
        _left   = -_right;
        _top    = ((_right - _left) / aspect) / two;
        _bottom = -_top;
    }
    else
    {
        _top    = nearPlane * std::tan (fovy / two);
        _bottom = -_top;
        _right  = (_top - _bottom) * aspect / two;
        _left   = -_right;
    }
    _nearPlane    = nearPlane;
    _farPlane     = farPlane;
    _orthographic = false;
}

template <class T>
inline void
Frustum<T>::set (T nearPlane, T farPlane, T fovx, T fovy, T aspect) IMATH_NOEXCEPT
{
    const T two = static_cast<T> (2);

    if (fovx != T (0))
    {
        _right  = nearPlane * std::tan (fovx / two);
        _left   = -_right;
        _top    = ((_right - _left) / aspect) / two;
        _bottom = -_top;
    }
    else
    {
        _top    = nearPlane * std::tan (fovy / two);
        _bottom = -_top;
        _right  = (_top - _bottom) * aspect / two;
        _left   = -_right;
    }
    _nearPlane    = nearPlane;
    _farPlane     = farPlane;
    _orthographic = false;
}

template <class T>
constexpr inline T
Frustum<T>::fovx() const IMATH_NOEXCEPT
{
    return std::atan2 (_right, _nearPlane) - std::atan2 (_left, _nearPlane);
}

template <class T>
constexpr inline T
Frustum<T>::fovy() const IMATH_NOEXCEPT
{
    return std::atan2 (_top, _nearPlane) - std::atan2 (_bottom, _nearPlane);
}

template <class T>
IMATH_CONSTEXPR14 inline T
Frustum<T>::aspectExc() const
{
    T rightMinusLeft = _right - _left;
    T topMinusBottom = _top - _bottom;

    if (abs (topMinusBottom) < T (1) && abs (rightMinusLeft) > std::numeric_limits<T>::max() * abs (topMinusBottom))
    {
        throw std::domain_error ("Bad viewing frustum: "
                                 "aspect ratio cannot be computed.");
    }

    return rightMinusLeft / topMinusBottom;
}

template <class T>
IMATH_CONSTEXPR14 inline T
Frustum<T>::aspect() const IMATH_NOEXCEPT
{
    T rightMinusLeft = _right - _left;
    T topMinusBottom = _top - _bottom;
    return rightMinusLeft / topMinusBottom;
}

template <class T>
IMATH_CONSTEXPR14 inline Matrix44<T>
Frustum<T>::projectionMatrixExc() const
{
    T rightPlusLeft  = _right + _left;
    T rightMinusLeft = _right - _left;

    T topPlusBottom  = _top + _bottom;
    T topMinusBottom = _top - _bottom;

    T farPlusNear  = _farPlane + _nearPlane;
    T farMinusNear = _farPlane - _nearPlane;

    if ((abs (rightMinusLeft) < T (1) &&
         abs (rightPlusLeft) > std::numeric_limits<T>::max() * abs (rightMinusLeft)) ||
        (abs (topMinusBottom) < T (1) &&
         abs (topPlusBottom) > std::numeric_limits<T>::max() * abs (topMinusBottom)) ||
        (abs (farMinusNear) < 1 && abs (farPlusNear) > std::numeric_limits<T>::max() * abs (farMinusNear)))
    {
        throw std::domain_error ("Bad viewing frustum: "
                                 "projection matrix cannot be computed.");
    }

    if (_orthographic)
    {
        T tx = -rightPlusLeft / rightMinusLeft;
        T ty = -topPlusBottom / topMinusBottom;
        T tz = -farPlusNear / farMinusNear;

        if ((abs (rightMinusLeft) < T (1) && T (2) > std::numeric_limits<T>::max() * abs (rightMinusLeft)) ||
            (abs (topMinusBottom) < T (1) && T (2) > std::numeric_limits<T>::max() * abs (topMinusBottom)) ||
            (abs (farMinusNear) < T (1) && T (2) > std::numeric_limits<T>::max() * abs (farMinusNear)))
        {
            throw std::domain_error ("Bad viewing frustum: "
                                     "projection matrix cannot be computed.");
        }

        T A = T (2) / rightMinusLeft;
        T B = T (2) / topMinusBottom;
        T C = T (-2) / farMinusNear;

        return Matrix44<T> (A, 0, 0, 0, 0, B, 0, 0, 0, 0, C, 0, tx, ty, tz, 1.f);
    }
    else
    {
        T A = rightPlusLeft / rightMinusLeft;
        T B = topPlusBottom / topMinusBottom;
        T C = -farPlusNear / farMinusNear;

        T farTimesNear = T (-2) * _farPlane * _nearPlane;
        if (abs (farMinusNear) < T (1) && abs (farTimesNear) > std::numeric_limits<T>::max() * abs (farMinusNear))
        {
            throw std::domain_error ("Bad viewing frustum: "
                                     "projection matrix cannot be computed.");
        }

        T D = farTimesNear / farMinusNear;

        T twoTimesNear = T (2) * _nearPlane;

        if ((abs (rightMinusLeft) < T (1) &&
             abs (twoTimesNear) > std::numeric_limits<T>::max() * abs (rightMinusLeft)) ||
            (abs (topMinusBottom) < T (1) &&
             abs (twoTimesNear) > std::numeric_limits<T>::max() * abs (topMinusBottom)))
        {
            throw std::domain_error ("Bad viewing frustum: "
                                     "projection matrix cannot be computed.");
        }

        T E = twoTimesNear / rightMinusLeft;
        T F = twoTimesNear / topMinusBottom;

        return Matrix44<T> (E, 0, 0, 0, 0, F, 0, 0, A, B, C, -1, 0, 0, D, 0);
    }
}

template <class T>
IMATH_CONSTEXPR14 inline Matrix44<T>
Frustum<T>::projectionMatrix() const IMATH_NOEXCEPT
{
    T rightPlusLeft  = _right + _left;
    T rightMinusLeft = _right - _left;

    T topPlusBottom  = _top + _bottom;
    T topMinusBottom = _top - _bottom;

    T farPlusNear  = _farPlane + _nearPlane;
    T farMinusNear = _farPlane - _nearPlane;

    if (_orthographic)
    {
        T tx = -rightPlusLeft / rightMinusLeft;
        T ty = -topPlusBottom / topMinusBottom;
        T tz = -farPlusNear / farMinusNear;

        T A = T (2) / rightMinusLeft;
        T B = T (2) / topMinusBottom;
        T C = T (-2) / farMinusNear;

        return Matrix44<T> (A, 0, 0, 0, 0, B, 0, 0, 0, 0, C, 0, tx, ty, tz, 1.f);
    }
    else
    {
        T A = rightPlusLeft / rightMinusLeft;
        T B = topPlusBottom / topMinusBottom;
        T C = -farPlusNear / farMinusNear;

        T farTimesNear = T (-2) * _farPlane * _nearPlane;

        T D = farTimesNear / farMinusNear;

        T twoTimesNear = T (2) * _nearPlane;

        T E = twoTimesNear / rightMinusLeft;
        T F = twoTimesNear / topMinusBottom;

        return Matrix44<T> (E, 0, 0, 0, 0, F, 0, 0, A, B, C, -1, 0, 0, D, 0);
    }
}

template <class T>
constexpr inline bool
Frustum<T>::degenerate() const IMATH_NOEXCEPT
{
    return (_nearPlane == _farPlane) || (_left == _right) || (_top == _bottom);
}

template <class T>
IMATH_CONSTEXPR14 inline Frustum<T>
Frustum<T>::window (T l, T r, T t, T b) const IMATH_NOEXCEPT
{
    // move it to 0->1 space

    Vec2<T> bl = screenToLocal (Vec2<T> (l, b));
    Vec2<T> tr = screenToLocal (Vec2<T> (r, t));

    return Frustum<T> (_nearPlane, _farPlane, bl.x, tr.x, tr.y, bl.y, _orthographic);
}

template <class T>
constexpr inline Vec2<T>
Frustum<T>::screenToLocal (const Vec2<T>& s) const IMATH_NOEXCEPT
{
    return Vec2<T> (_left + (_right - _left) * (1.f + s.x) / 2.f,
                    _bottom + (_top - _bottom) * (1.f + s.y) / 2.f);
}

template <class T>
IMATH_CONSTEXPR14 inline Vec2<T>
Frustum<T>::localToScreenExc (const Vec2<T>& p) const
{
    T leftPlusRight  = _left - T (2) * p.x + _right;
    T leftMinusRight = _left - _right;
    T bottomPlusTop  = _bottom - T (2) * p.y + _top;
    T bottomMinusTop = _bottom - _top;

    if ((abs (leftMinusRight) < T (1) &&
         abs (leftPlusRight) > std::numeric_limits<T>::max() * abs (leftMinusRight)) ||
        (abs (bottomMinusTop) < T (1) &&
         abs (bottomPlusTop) > std::numeric_limits<T>::max() * abs (bottomMinusTop)))
    {
        throw std::domain_error ("Bad viewing frustum: "
                                 "local-to-screen transformation cannot be computed");
    }

    return Vec2<T> (leftPlusRight / leftMinusRight, bottomPlusTop / bottomMinusTop);
}

template <class T>
IMATH_CONSTEXPR14 inline Vec2<T>
Frustum<T>::localToScreen (const Vec2<T>& p) const IMATH_NOEXCEPT
{
    T leftPlusRight  = _left - T (2) * p.x + _right;
    T leftMinusRight = _left - _right;
    T bottomPlusTop  = _bottom - T (2) * p.y + _top;
    T bottomMinusTop = _bottom - _top;

    return Vec2<T> (leftPlusRight / leftMinusRight, bottomPlusTop / bottomMinusTop);
}

template <class T>
IMATH_CONSTEXPR14 inline Line3<T>
Frustum<T>::projectScreenToRay (const Vec2<T>& p) const IMATH_NOEXCEPT
{
    Vec2<T> point = screenToLocal (p);
    if (orthographic())
        return Line3<T> (Vec3<T> (point.x, point.y, 0.0), Vec3<T> (point.x, point.y, -1.0));
    else
        return Line3<T> (Vec3<T> (0, 0, 0), Vec3<T> (point.x, point.y, -_nearPlane));
}

template <class T>
IMATH_CONSTEXPR14 Vec2<T>
Frustum<T>::projectPointToScreenExc (const Vec3<T>& point) const
{
    if (orthographic() || point.z == T (0))
        return localToScreenExc (Vec2<T> (point.x, point.y));
    else
        return localToScreenExc (
            Vec2<T> (point.x * _nearPlane / -point.z, point.y * _nearPlane / -point.z));
}

template <class T>
IMATH_CONSTEXPR14 Vec2<T>
Frustum<T>::projectPointToScreen (const Vec3<T>& point) const IMATH_NOEXCEPT
{
    if (orthographic() || point.z == T (0))
        return localToScreen (Vec2<T> (point.x, point.y));
    else
        return localToScreen (
            Vec2<T> (point.x * _nearPlane / -point.z, point.y * _nearPlane / -point.z));
}

template <class T>
IMATH_CONSTEXPR14 T
Frustum<T>::ZToDepthExc (long zval, long zmin, long zmax) const
{
    int zdiff = zmax - zmin;

    if (zdiff == 0)
    {
        throw std::domain_error ("Bad call to Frustum::ZToDepth: zmax == zmin");
    }

    if (zval > zmax + 1)
        zval -= zdiff;

    T fzval = (T (zval) - T (zmin)) / T (zdiff);
    return normalizedZToDepthExc (fzval);
}

template <class T>
IMATH_CONSTEXPR14 T
Frustum<T>::ZToDepth (long zval, long zmin, long zmax) const IMATH_NOEXCEPT
{
    int zdiff = zmax - zmin;

    if (zval > zmax + 1)
        zval -= zdiff;

    T fzval = (T (zval) - T (zmin)) / T (zdiff);
    return normalizedZToDepth (fzval);
}

template <class T>
IMATH_CONSTEXPR14 T
Frustum<T>::normalizedZToDepthExc (T zval) const
{
    T Zp = zval * T (2) - T (1);

    if (_orthographic)
    {
        return -(Zp * (_farPlane - _nearPlane) + (_farPlane + _nearPlane)) / T (2);
    }
    else
    {
        T farTimesNear = 2 * _farPlane * _nearPlane;
        T farMinusNear = Zp * (_farPlane - _nearPlane) - _farPlane - _nearPlane;

        if (abs (farMinusNear) < 1 && abs (farTimesNear) > std::numeric_limits<T>::max() * abs (farMinusNear))
        {
            throw std::domain_error ("Frustum::normalizedZToDepth cannot be computed: "
                                     "near and far clipping planes of the viewing frustum "
                                     "may be too close to each other");
        }

        return farTimesNear / farMinusNear;
    }
}

template <class T>
IMATH_CONSTEXPR14 T
Frustum<T>::normalizedZToDepth (T zval) const IMATH_NOEXCEPT
{
    T Zp = zval * T (2) - T (1);

    if (_orthographic)
    {
        return -(Zp * (_farPlane - _nearPlane) + (_farPlane + _nearPlane)) / T (2);
    }
    else
    {
        T farTimesNear = 2 * _farPlane * _nearPlane;
        T farMinusNear = Zp * (_farPlane - _nearPlane) - _farPlane - _nearPlane;

        return farTimesNear / farMinusNear;
    }
}

template <class T>
IMATH_CONSTEXPR14 long
Frustum<T>::DepthToZExc (T depth, long zmin, long zmax) const
{
    long zdiff     = zmax - zmin;
    T farMinusNear = _farPlane - _nearPlane;

    if (_orthographic)
    {
        T farPlusNear = T (2) * depth + _farPlane + _nearPlane;

        if (abs (farMinusNear) < T (1) && abs (farPlusNear) > std::numeric_limits<T>::max() * abs (farMinusNear))
        {
            throw std::domain_error ("Bad viewing frustum: "
                                     "near and far clipping planes "
                                     "are too close to each other");
        }

        T Zp = -farPlusNear / farMinusNear;
        return long (0.5 * (Zp + 1) * zdiff) + zmin;
    }
    else
    {
        // Perspective

        T farTimesNear = T (2) * _farPlane * _nearPlane;
        if (abs (depth) < T (1) && abs (farTimesNear) > std::numeric_limits<T>::max() * abs (depth))
        {
            throw std::domain_error ("Bad call to DepthToZ function: "
                                     "value of `depth' is too small");
        }

        T farPlusNear = farTimesNear / depth + _farPlane + _nearPlane;
        if (abs (farMinusNear) < T (1) && abs (farPlusNear) > std::numeric_limits<T>::max() * abs (farMinusNear))
        {
            throw std::domain_error ("Bad viewing frustum: "
                                     "near and far clipping planes "
                                     "are too close to each other");
        }

        T Zp = farPlusNear / farMinusNear;
        return long (0.5 * (Zp + 1) * zdiff) + zmin;
    }
}

template <class T>
IMATH_CONSTEXPR14 long
Frustum<T>::DepthToZ (T depth, long zmin, long zmax) const IMATH_NOEXCEPT
{
    long zdiff     = zmax - zmin;
    T farMinusNear = _farPlane - _nearPlane;

    if (_orthographic)
    {
        T farPlusNear = T (2) * depth + _farPlane + _nearPlane;

        T Zp = -farPlusNear / farMinusNear;
        return long (0.5 * (Zp + 1) * zdiff) + zmin;
    }
    else
    {
        // Perspective

        T farTimesNear = T (2) * _farPlane * _nearPlane;

        T farPlusNear = farTimesNear / depth + _farPlane + _nearPlane;

        T Zp = farPlusNear / farMinusNear;
        return long (0.5 * (Zp + 1) * zdiff) + zmin;
    }
}

template <class T>
IMATH_CONSTEXPR14 T
Frustum<T>::screenRadiusExc (const Vec3<T>& p, T radius) const
{
    // Derivation:
    // Consider X-Z plane.
    // X coord of projection of p = xp = p.x * (-_nearPlane / p.z)
    // Let q be p + (radius, 0, 0).
    // X coord of projection of q = xq = (p.x - radius)  * (-_nearPlane / p.z)
    // X coord of projection of segment from p to q = r = xp - xq
    //         = radius * (-_nearPlane / p.z)
    // A similar analysis holds in the Y-Z plane.
    // So r is the quantity we want to return.

    if (abs (p.z) > T (1) || abs (-_nearPlane) < std::numeric_limits<T>::max() * abs (p.z))
    {
        return radius * (-_nearPlane / p.z);
    }
    else
    {
        throw std::domain_error ("Bad call to Frustum::screenRadius: "
                                 "magnitude of `p' is too small");
    }

    return radius * (-_nearPlane / p.z);
}

template <class T>
IMATH_CONSTEXPR14 T
Frustum<T>::screenRadius (const Vec3<T>& p, T radius) const IMATH_NOEXCEPT
{
    // Derivation:
    // Consider X-Z plane.
    // X coord of projection of p = xp = p.x * (-_nearPlane / p.z)
    // Let q be p + (radius, 0, 0).
    // X coord of projection of q = xq = (p.x - radius)  * (-_nearPlane / p.z)
    // X coord of projection of segment from p to q = r = xp - xq
    //         = radius * (-_nearPlane / p.z)
    // A similar analysis holds in the Y-Z plane.
    // So r is the quantity we want to return.

    return radius * (-_nearPlane / p.z);
}

template <class T>
IMATH_CONSTEXPR14 T
Frustum<T>::worldRadiusExc (const Vec3<T>& p, T radius) const
{
    if (abs (-_nearPlane) > T (1) || abs (p.z) < std::numeric_limits<T>::max() * abs (-_nearPlane))
    {
        return radius * (p.z / -_nearPlane);
    }
    else
    {
        throw std::domain_error ("Bad viewing frustum: "
                                 "near clipping plane is too close to zero");
    }
}

template <class T>
IMATH_CONSTEXPR14 T
Frustum<T>::worldRadius (const Vec3<T>& p, T radius) const IMATH_NOEXCEPT
{
    return radius * (p.z / -_nearPlane);
}

template <class T>
void
Frustum<T>::planes (Plane3<T> p[6]) const IMATH_NOEXCEPT
{
    //
    //        Plane order: Top, Right, Bottom, Left, Near, Far.
    //  Normals point outwards.
    //

    if (!_orthographic)
    {
        Vec3<T> a (_left, _bottom, -_nearPlane);
        Vec3<T> b (_left, _top, -_nearPlane);
        Vec3<T> c (_right, _top, -_nearPlane);
        Vec3<T> d (_right, _bottom, -_nearPlane);
        Vec3<T> o (0, 0, 0);

        p[0].set (o, c, b);
        p[1].set (o, d, c);
        p[2].set (o, a, d);
        p[3].set (o, b, a);
    }
    else
    {
        p[0].set (Vec3<T> (0, 1, 0), _top);
        p[1].set (Vec3<T> (1, 0, 0), _right);
        p[2].set (Vec3<T> (0, -1, 0), -_bottom);
        p[3].set (Vec3<T> (-1, 0, 0), -_left);
    }
    p[4].set (Vec3<T> (0, 0, 1), -_nearPlane);
    p[5].set (Vec3<T> (0, 0, -1), _farPlane);
}

template <class T>
void
Frustum<T>::planes (Plane3<T> p[6], const Matrix44<T>& M) const IMATH_NOEXCEPT
{
    //
    //  Plane order: Top, Right, Bottom, Left, Near, Far.
    //  Normals point outwards.
    //

    Vec3<T> a = Vec3<T> (_left, _bottom, -_nearPlane) * M;
    Vec3<T> b = Vec3<T> (_left, _top, -_nearPlane) * M;
    Vec3<T> c = Vec3<T> (_right, _top, -_nearPlane) * M;
    Vec3<T> d = Vec3<T> (_right, _bottom, -_nearPlane) * M;
    if (!_orthographic)
    {
        double s    = _farPlane / double (_nearPlane);
        T farLeft   = (T) (s * _left);
        T farRight  = (T) (s * _right);
        T farTop    = (T) (s * _top);
        T farBottom = (T) (s * _bottom);
        Vec3<T> e   = Vec3<T> (farLeft, farBottom, -_farPlane) * M;
        Vec3<T> f   = Vec3<T> (farLeft, farTop, -_farPlane) * M;
        Vec3<T> g   = Vec3<T> (farRight, farTop, -_farPlane) * M;
        Vec3<T> o   = Vec3<T> (0, 0, 0) * M;
        p[0].set (o, c, b);
        p[1].set (o, d, c);
        p[2].set (o, a, d);
        p[3].set (o, b, a);
        p[4].set (a, d, c);
        p[5].set (e, f, g);
    }
    else
    {
        Vec3<T> e = Vec3<T> (_left, _bottom, -_farPlane) * M;
        Vec3<T> f = Vec3<T> (_left, _top, -_farPlane) * M;
        Vec3<T> g = Vec3<T> (_right, _top, -_farPlane) * M;
        Vec3<T> h = Vec3<T> (_right, _bottom, -_farPlane) * M;
        p[0].set (c, g, f);
        p[1].set (d, h, g);
        p[2].set (a, e, h);
        p[3].set (b, f, e);
        p[4].set (a, d, c);
        p[5].set (e, f, g);
    }
}

/// Frustum of type float
typedef Frustum<float> Frustumf;

/// Frustum of type double
typedef Frustum<double> Frustumd;

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#if defined _WIN32 || defined _WIN64
#    ifdef _redef_near
#        define near
#    endif
#    ifdef _redef_far
#        define far
#    endif
#endif

#endif // INCLUDED_IMATHFRUSTUM_H
