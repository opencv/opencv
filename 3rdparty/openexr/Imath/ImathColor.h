//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// 3-channel and 4-channel color representations
//

#ifndef INCLUDED_IMATHCOLOR_H
#define INCLUDED_IMATHCOLOR_H

#include "ImathExport.h"
#include "ImathNamespace.h"

#include "ImathVec.h"
#include "half.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

///
/// 3-channel color class that inherits from Vec3.
///
/// This class does not impose interpretation on the channels, which
/// can represent either rgb or hsv color values.
///
/// Note: because Color3 inherits from Vec3, its member fields are
/// called `x`, `y`, and `z`.

template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Color3 : public Vec3<T>
{
  public:

    /// @{
    /// @name Constructors and Assignemt

    /// No initialization by default
    IMATH_HOSTDEVICE Color3() IMATH_NOEXCEPT;                         

    /// Initialize to (a a a)
    IMATH_HOSTDEVICE constexpr explicit Color3 (T a) IMATH_NOEXCEPT;  

    /// Initialize to (a b c)
    IMATH_HOSTDEVICE constexpr Color3 (T a, T b, T c) IMATH_NOEXCEPT; 

    /// Construct from Color3
    IMATH_HOSTDEVICE constexpr Color3 (const Color3& c) IMATH_NOEXCEPT; 

    /// Construct from Vec3
    template <class S> IMATH_HOSTDEVICE constexpr Color3 (const Vec3<S>& v) IMATH_NOEXCEPT; 

    /// Destructor
    ~Color3() = default;

    /// Component-wise assignment
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color3& operator= (const Color3& c) IMATH_NOEXCEPT; 

    /// @}
    
    /// @{
    /// @name Arithmetic
    
    /// Component-wise addition
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color3& operator+= (const Color3& c) IMATH_NOEXCEPT; 

    /// Component-wise addition
    IMATH_HOSTDEVICE constexpr Color3 operator+ (const Color3& c) const IMATH_NOEXCEPT;  

    /// Component-wise subtraction
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color3& operator-= (const Color3& c) IMATH_NOEXCEPT; 

    /// Component-wise subtraction
    IMATH_HOSTDEVICE constexpr Color3 operator- (const Color3& c) const IMATH_NOEXCEPT; 

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE constexpr Color3 operator-() const IMATH_NOEXCEPT; 

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color3& negate() IMATH_NOEXCEPT; 

    /// Component-wise multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color3& operator*= (const Color3& c) IMATH_NOEXCEPT; 

    /// Component-wise multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color3& operator*= (T a) IMATH_NOEXCEPT;  

    /// Component-wise multiplication
    IMATH_HOSTDEVICE constexpr Color3 operator* (const Color3& c) const IMATH_NOEXCEPT;  

    /// Component-wise multiplication
    IMATH_HOSTDEVICE constexpr Color3 operator* (T a) const IMATH_NOEXCEPT;  

    /// Component-wise division
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color3& operator/= (const Color3& c) IMATH_NOEXCEPT; 

    /// Component-wise division
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color3& operator/= (T a) IMATH_NOEXCEPT; 

    /// Component-wise division
    IMATH_HOSTDEVICE constexpr Color3 operator/ (const Color3& c) const IMATH_NOEXCEPT;  

    /// Component-wise division
    IMATH_HOSTDEVICE constexpr Color3 operator/ (T a) const IMATH_NOEXCEPT;  

    /// @}
};

///
/// A 4-channel color class: 3 channels plus alpha. 
///
/// For convenience, the fields are named `r`, `g`, and `b`, although
/// this class does not impose interpretation on the channels, which
/// can represent either rgb or hsv color values.
///

template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Color4
{
  public:

    /// @{
    /// @name Direct access to elements

    T r, g, b, a;

    /// @}

    /// @{
    /// @name Constructors and Assignment

    /// No initialization by default
    IMATH_HOSTDEVICE Color4() IMATH_NOEXCEPT;                                      

    /// Initialize to `(a a a a)`
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 explicit Color4 (T a) IMATH_NOEXCEPT;       

    /// Initialize to `(a b c d)`
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Color4 (T a, T b, T c, T d) IMATH_NOEXCEPT; 

    /// Construct from Color4
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Color4 (const Color4& v) IMATH_NOEXCEPT; 

    /// Construct from Color4
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Color4 (const Color4<S>& v) IMATH_NOEXCEPT; 

    /// Destructor
    ~Color4() = default;

    /// Assignment
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color4& operator= (const Color4& v) IMATH_NOEXCEPT; 

    /// Component-wise value
    IMATH_HOSTDEVICE T& operator[] (int i) IMATH_NOEXCEPT; 

    /// Component-wise value
    IMATH_HOSTDEVICE const T& operator[] (int i) const IMATH_NOEXCEPT; 

    /// @}
    
    /// @{
    /// @name Arithmetic and Comparison
    
    /// Equality
    template <class S> IMATH_HOSTDEVICE constexpr bool operator== (const Color4<S>& v) const IMATH_NOEXCEPT; 

    /// Inequality
    template <class S> IMATH_HOSTDEVICE constexpr bool operator!= (const Color4<S>& v) const IMATH_NOEXCEPT; 

    /// Component-wise addition
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color4& operator+= (const Color4& v) IMATH_NOEXCEPT; 

    /// Component-wise addition
    IMATH_HOSTDEVICE constexpr Color4 operator+ (const Color4& v) const IMATH_NOEXCEPT; 

    /// Component-wise subtraction
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color4& operator-= (const Color4& v) IMATH_NOEXCEPT; 

    /// Component-wise subtraction
    IMATH_HOSTDEVICE constexpr Color4 operator- (const Color4& v) const IMATH_NOEXCEPT; 

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE constexpr Color4 operator-() const IMATH_NOEXCEPT; 

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color4& negate() IMATH_NOEXCEPT; 

    /// Component-wise multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color4& operator*= (const Color4& v) IMATH_NOEXCEPT; 

    /// Component-wise multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color4& operator*= (T a) IMATH_NOEXCEPT; 

    /// Component-wise multiplication
    IMATH_HOSTDEVICE constexpr Color4 operator* (const Color4& v) const IMATH_NOEXCEPT; 

    /// Component-wise multiplication
    IMATH_HOSTDEVICE constexpr Color4 operator* (T a) const IMATH_NOEXCEPT; 

    /// Component-wise division
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color4& operator/= (const Color4& v) IMATH_NOEXCEPT; 

    /// Component-wise division
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Color4& operator/= (T a) IMATH_NOEXCEPT; 

    /// Component-wise division
    IMATH_HOSTDEVICE constexpr Color4 operator/ (const Color4& v) const IMATH_NOEXCEPT; 

    /// Component-wise division
    IMATH_HOSTDEVICE constexpr Color4 operator/ (T a) const IMATH_NOEXCEPT; 

    /// @}

    /// @{
    /// @name Numeric Limits
    
    /// Number of dimensions (channels), i.e. 4 for a Color4
    IMATH_HOSTDEVICE constexpr static unsigned int dimensions() IMATH_NOEXCEPT { return 4; }

    /// Largest possible negative value
    IMATH_HOSTDEVICE constexpr static T baseTypeLowest() IMATH_NOEXCEPT { return std::numeric_limits<T>::lowest(); }

    /// Largest possible positive value
    IMATH_HOSTDEVICE constexpr static T baseTypeMax() IMATH_NOEXCEPT { return std::numeric_limits<T>::max(); }

    /// Smallest possible positive value
    IMATH_HOSTDEVICE constexpr static T baseTypeSmallest() IMATH_NOEXCEPT { return std::numeric_limits<T>::min(); }

    /// Smallest possible e for which 1+e != 1
    IMATH_HOSTDEVICE constexpr static T baseTypeEpsilon() IMATH_NOEXCEPT { return std::numeric_limits<T>::epsilon(); }

    /// @}
    
    /// The base type: In templates that accept a parameter `V` (could
    /// be a Color4), you can refer to `T` as `V::BaseType`
    typedef T BaseType;

    /// @{
    /// @name Compatibilty with Sb

    /// Set the value
    template <class S> IMATH_HOSTDEVICE void setValue (S a, S b, S c, S d) IMATH_NOEXCEPT; 

    /// Set the value
    template <class S> IMATH_HOSTDEVICE void setValue (const Color4<S>& v) IMATH_NOEXCEPT; 

    /// Return the value
    template <class S> IMATH_HOSTDEVICE void getValue (S& a, S& b, S& c, S& d) const IMATH_NOEXCEPT; 

    /// Return the value
    template <class S> IMATH_HOSTDEVICE void getValue (Color4<S>& v) const IMATH_NOEXCEPT; 

    /// Return raw pointer to the value
    IMATH_HOSTDEVICE T* getValue() IMATH_NOEXCEPT; 

    /// Return raw pointer to the value
    IMATH_HOSTDEVICE const T* getValue() const IMATH_NOEXCEPT; 

    /// @}
};

/// Stream output, as "(r g b a)"
template <class T> std::ostream& operator<< (std::ostream& s, const Color4<T>& v);

/// Reverse multiplication: S * Color4
template <class S, class T>
IMATH_HOSTDEVICE constexpr Color4<T> operator* (S a, const Color4<T>& v) IMATH_NOEXCEPT;

/// 3 float channels
typedef Color3<float> Color3f;

/// 3 half channels
typedef Color3<half> Color3h;

/// 3 8-bit integer channels
typedef Color3<unsigned char> Color3c;

/// 3 half channels
typedef Color3<half> C3h;

/// 3 float channels
typedef Color3<float> C3f;

/// 3 8-bit integer channels
typedef Color3<unsigned char> C3c;

/// 4 float channels
typedef Color4<float> Color4f;

/// 4 half channels
typedef Color4<half> Color4h;

/// 4 8-bit integer channels
typedef Color4<unsigned char> Color4c;

/// 4 float channels
typedef Color4<float> C4f;

/// 4 half channels
typedef Color4<half> C4h;

/// 4 8-bit integer channels
typedef Color4<unsigned char> C4c;

/// Packed 32-bit integer
typedef unsigned int PackedColor;

//
// Implementation of Color3
//

template <class T> IMATH_HOSTDEVICE inline Color3<T>::Color3() IMATH_NOEXCEPT : Vec3<T>()
{
    // empty
}

template <class T> IMATH_HOSTDEVICE constexpr inline Color3<T>::Color3 (T a) IMATH_NOEXCEPT : Vec3<T> (a)
{
    // empty
}

template <class T> IMATH_HOSTDEVICE constexpr inline Color3<T>::Color3 (T a, T b, T c) IMATH_NOEXCEPT : Vec3<T> (a, b, c)
{
    // empty
}

template <class T> IMATH_HOSTDEVICE constexpr inline Color3<T>::Color3 (const Color3& c) IMATH_NOEXCEPT : Vec3<T> (c)
{
    // empty
}

template <class T>
template <class S>
IMATH_HOSTDEVICE constexpr inline Color3<T>::Color3 (const Vec3<S>& v) IMATH_NOEXCEPT : Vec3<T> (v)
{
    //empty
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color3<T>&
Color3<T>::operator= (const Color3& c) IMATH_NOEXCEPT
{
    *((Vec3<T>*) this) = c;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color3<T>&
Color3<T>::operator+= (const Color3& c) IMATH_NOEXCEPT
{
    *((Vec3<T>*) this) += c;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color3<T>
Color3<T>::operator+ (const Color3& c) const IMATH_NOEXCEPT
{
    return Color3 (*(Vec3<T>*) this + (const Vec3<T>&) c);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color3<T>&
Color3<T>::operator-= (const Color3& c) IMATH_NOEXCEPT
{
    *((Vec3<T>*) this) -= c;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color3<T>
Color3<T>::operator- (const Color3& c) const IMATH_NOEXCEPT
{
    return Color3 (*(Vec3<T>*) this - (const Vec3<T>&) c);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color3<T>
Color3<T>::operator-() const IMATH_NOEXCEPT
{
    return Color3 (-(*(Vec3<T>*) this));
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color3<T>&
Color3<T>::negate() IMATH_NOEXCEPT
{
    ((Vec3<T>*) this)->negate();
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color3<T>&
Color3<T>::operator*= (const Color3& c) IMATH_NOEXCEPT
{
    *((Vec3<T>*) this) *= c;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color3<T>&
Color3<T>::operator*= (T a) IMATH_NOEXCEPT
{
    *((Vec3<T>*) this) *= a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color3<T>
Color3<T>::operator* (const Color3& c) const IMATH_NOEXCEPT
{
    return Color3 (*(Vec3<T>*) this * (const Vec3<T>&) c);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color3<T>
Color3<T>::operator* (T a) const IMATH_NOEXCEPT
{
    return Color3 (*(Vec3<T>*) this * a);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color3<T>&
Color3<T>::operator/= (const Color3& c) IMATH_NOEXCEPT
{
    *((Vec3<T>*) this) /= c;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color3<T>&
Color3<T>::operator/= (T a) IMATH_NOEXCEPT
{
    *((Vec3<T>*) this) /= a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color3<T>
Color3<T>::operator/ (const Color3& c) const IMATH_NOEXCEPT
{
    return Color3 (*(Vec3<T>*) this / (const Vec3<T>&) c);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color3<T>
Color3<T>::operator/ (T a) const IMATH_NOEXCEPT
{
    return Color3 (*(Vec3<T>*) this / a);
}

//
// Implementation of Color4
//

template <class T>
IMATH_HOSTDEVICE inline T&
Color4<T>::operator[] (int i) IMATH_NOEXCEPT
{
    return (&r)[i];
}

template <class T>
IMATH_HOSTDEVICE inline const T&
Color4<T>::operator[] (int i) const IMATH_NOEXCEPT
{
    return (&r)[i];
}

template <class T>
IMATH_HOSTDEVICE inline Color4<T>::Color4() IMATH_NOEXCEPT
{
    // empty
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Color4<T>::Color4 (T x) IMATH_NOEXCEPT
{
    r = g = b = a = x;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Color4<T>::Color4 (T x, T y, T z, T w) IMATH_NOEXCEPT
{
    r = x;
    g = y;
    b = z;
    a = w;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Color4<T>::Color4 (const Color4& v) IMATH_NOEXCEPT
{
    r = v.r;
    g = v.g;
    b = v.b;
    a = v.a;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Color4<T>::Color4 (const Color4<S>& v) IMATH_NOEXCEPT
{
    r = T (v.r);
    g = T (v.g);
    b = T (v.b);
    a = T (v.a);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color4<T>&
Color4<T>::operator= (const Color4& v) IMATH_NOEXCEPT
{
    r = v.r;
    g = v.g;
    b = v.b;
    a = v.a;
    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Color4<T>::setValue (S x, S y, S z, S w) IMATH_NOEXCEPT
{
    r = T (x);
    g = T (y);
    b = T (z);
    a = T (w);
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Color4<T>::setValue (const Color4<S>& v) IMATH_NOEXCEPT
{
    r = T (v.r);
    g = T (v.g);
    b = T (v.b);
    a = T (v.a);
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Color4<T>::getValue (S& x, S& y, S& z, S& w) const IMATH_NOEXCEPT
{
    x = S (r);
    y = S (g);
    z = S (b);
    w = S (a);
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Color4<T>::getValue (Color4<S>& v) const IMATH_NOEXCEPT
{
    v.r = S (r);
    v.g = S (g);
    v.b = S (b);
    v.a = S (a);
}

template <class T>
IMATH_HOSTDEVICE inline T*
Color4<T>::getValue() IMATH_NOEXCEPT
{
    return (T*) &r;
}

template <class T>
IMATH_HOSTDEVICE inline const T*
Color4<T>::getValue() const IMATH_NOEXCEPT
{
    return (const T*) &r;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE constexpr inline bool
Color4<T>::operator== (const Color4<S>& v) const IMATH_NOEXCEPT
{
    return r == v.r && g == v.g && b == v.b && a == v.a;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE constexpr inline bool
Color4<T>::operator!= (const Color4<S>& v) const IMATH_NOEXCEPT
{
    return r != v.r || g != v.g || b != v.b || a != v.a;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color4<T>&
Color4<T>::operator+= (const Color4& v) IMATH_NOEXCEPT
{
    r += v.r;
    g += v.g;
    b += v.b;
    a += v.a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color4<T>
Color4<T>::operator+ (const Color4& v) const IMATH_NOEXCEPT
{
    return Color4 (r + v.r, g + v.g, b + v.b, a + v.a);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color4<T>&
Color4<T>::operator-= (const Color4& v) IMATH_NOEXCEPT
{
    r -= v.r;
    g -= v.g;
    b -= v.b;
    a -= v.a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color4<T>
Color4<T>::operator- (const Color4& v) const IMATH_NOEXCEPT
{
    return Color4 (r - v.r, g - v.g, b - v.b, a - v.a);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color4<T>
Color4<T>::operator-() const IMATH_NOEXCEPT
{
    return Color4 (-r, -g, -b, -a);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color4<T>&
Color4<T>::negate() IMATH_NOEXCEPT
{
    r = -r;
    g = -g;
    b = -b;
    a = -a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color4<T>&
Color4<T>::operator*= (const Color4& v) IMATH_NOEXCEPT
{
    r *= v.r;
    g *= v.g;
    b *= v.b;
    a *= v.a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color4<T>&
Color4<T>::operator*= (T x) IMATH_NOEXCEPT
{
    r *= x;
    g *= x;
    b *= x;
    a *= x;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color4<T>
Color4<T>::operator* (const Color4& v) const IMATH_NOEXCEPT
{
    return Color4 (r * v.r, g * v.g, b * v.b, a * v.a);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color4<T>
Color4<T>::operator* (T x) const IMATH_NOEXCEPT
{
    return Color4 (r * x, g * x, b * x, a * x);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color4<T>&
Color4<T>::operator/= (const Color4& v) IMATH_NOEXCEPT
{
    r /= v.r;
    g /= v.g;
    b /= v.b;
    a /= v.a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Color4<T>&
Color4<T>::operator/= (T x) IMATH_NOEXCEPT
{
    r /= x;
    g /= x;
    b /= x;
    a /= x;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color4<T>
Color4<T>::operator/ (const Color4& v) const IMATH_NOEXCEPT
{
    return Color4 (r / v.r, g / v.g, b / v.b, a / v.a);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Color4<T>
Color4<T>::operator/ (T x) const IMATH_NOEXCEPT
{
    return Color4 (r / x, g / x, b / x, a / x);
}

template <class T>
std::ostream&
operator<< (std::ostream& s, const Color4<T>& v)
{
    return s << '(' << v.r << ' ' << v.g << ' ' << v.b << ' ' << v.a << ')';
}

//
// Implementation of reverse multiplication
//

template <class S, class T>
IMATH_HOSTDEVICE constexpr inline Color4<T>
operator* (S x, const Color4<T>& v) IMATH_NOEXCEPT
{
    return Color4<T> (x * v.r, x * v.g, x * v.b, x * v.a);
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHCOLOR_H
