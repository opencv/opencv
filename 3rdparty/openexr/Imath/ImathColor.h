///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004-2012, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////



#ifndef INCLUDED_IMATHCOLOR_H
#define INCLUDED_IMATHCOLOR_H

//----------------------------------------------------
//
//	A three and four component color class template.
//
//----------------------------------------------------

#include "ImathVec.h"
#include "ImathNamespace.h"
#include "half.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER


template <class T>
class Color3: public Vec3 <T>
{
  public:

    //-------------
    // Constructors
    //-------------

    Color3 ();			// no initialization
    explicit Color3 (T a);	// (a a a)
    Color3 (T a, T b, T c);	// (a b c)


    //---------------------------------
    // Copy constructors and assignment
    //---------------------------------

    Color3 (const Color3 &c);
    template <class S> Color3 (const Vec3<S> &v);

    const Color3 &	operator = (const Color3 &c);


    //------------------------
    // Component-wise addition
    //------------------------

    const Color3 &	operator += (const Color3 &c);
    Color3		operator + (const Color3 &c) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    const Color3 &	operator -= (const Color3 &c);
    Color3		operator - (const Color3 &c) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    Color3		operator - () const;
    const Color3 &	negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    const Color3 &	operator *= (const Color3 &c);
    const Color3 &	operator *= (T a);
    Color3		operator * (const Color3 &c) const;
    Color3		operator * (T a) const;


    //------------------------
    // Component-wise division
    //------------------------

    const Color3 &	operator /= (const Color3 &c);
    const Color3 &	operator /= (T a);
    Color3		operator / (const Color3 &c) const;
    Color3		operator / (T a) const;
};

template <class T> class Color4
{
  public:

    //-------------------
    // Access to elements
    //-------------------

    T			r, g, b, a;

    T &			operator [] (int i);
    const T &		operator [] (int i) const;


    //-------------
    // Constructors
    //-------------

    Color4 ();			    	// no initialization
    explicit Color4 (T a);		// (a a a a)
    Color4 (T a, T b, T c, T d);	// (a b c d)


    //---------------------------------
    // Copy constructors and assignment
    //---------------------------------

    Color4 (const Color4 &v);
    template <class S> Color4 (const Color4<S> &v);

    const Color4 &	operator = (const Color4 &v);


    //----------------------
    // Compatibility with Sb
    //----------------------

    template <class S>
    void		setValue (S a, S b, S c, S d);

    template <class S>
    void		setValue (const Color4<S> &v);

    template <class S>
    void		getValue (S &a, S &b, S &c, S &d) const;

    template <class S>
    void		getValue (Color4<S> &v) const;

    T *			getValue();
    const T *		getValue() const;


    //---------
    // Equality
    //---------

    template <class S>
    bool		operator == (const Color4<S> &v) const;

    template <class S>
    bool		operator != (const Color4<S> &v) const;


    //------------------------
    // Component-wise addition
    //------------------------

    const Color4 &	operator += (const Color4 &v);
    Color4		operator + (const Color4 &v) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    const Color4 &	operator -= (const Color4 &v);
    Color4		operator - (const Color4 &v) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    Color4		operator - () const;
    const Color4 &	negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    const Color4 &	operator *= (const Color4 &v);
    const Color4 &	operator *= (T a);
    Color4		operator * (const Color4 &v) const;
    Color4		operator * (T a) const;


    //------------------------
    // Component-wise division
    //------------------------

    const Color4 &	operator /= (const Color4 &v);
    const Color4 &	operator /= (T a);
    Color4		operator / (const Color4 &v) const;
    Color4		operator / (T a) const;


    //----------------------------------------------------------
    // Number of dimensions, i.e. number of elements in a Color4
    //----------------------------------------------------------

    static unsigned int	dimensions() {return 4;}


    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    static T		baseTypeMin()		{return limits<T>::min();}
    static T		baseTypeMax()		{return limits<T>::max();}
    static T		baseTypeSmallest()	{return limits<T>::smallest();}
    static T		baseTypeEpsilon()	{return limits<T>::epsilon();}


    //--------------------------------------------------------------
    // Base type -- in templates, which accept a parameter, V, which
    // could be a Color4<T>, you can refer to T as
    // V::BaseType
    //--------------------------------------------------------------

    typedef T		BaseType;
};

//--------------
// Stream output
//--------------

template <class T>
std::ostream &	operator << (std::ostream &s, const Color4<T> &v);

//----------------------------------------------------
// Reverse multiplication: S * Color4<T>
//----------------------------------------------------

template <class S, class T> Color4<T>	operator * (S a, const Color4<T> &v);

//-------------------------
// Typedefs for convenience
//-------------------------

typedef Color3<float>		Color3f;
typedef Color3<half>		Color3h;
typedef Color3<unsigned char>	Color3c;
typedef Color3<half>		C3h;
typedef Color3<float>		C3f;
typedef Color3<unsigned char>	C3c;
typedef Color4<float>		Color4f;
typedef Color4<half>		Color4h;
typedef Color4<unsigned char>	Color4c;
typedef Color4<float>		C4f;
typedef Color4<half>		C4h;
typedef Color4<unsigned char>	C4c;
typedef unsigned int		PackedColor;


//-------------------------
// Implementation of Color3
//-------------------------

template <class T>
inline
Color3<T>::Color3 (): Vec3 <T> ()
{
    // empty
}

template <class T>
inline
Color3<T>::Color3 (T a): Vec3 <T> (a)
{
    // empty
}

template <class T>
inline
Color3<T>::Color3 (T a, T b, T c): Vec3 <T> (a, b, c)
{
    // empty
}

template <class T>
inline
Color3<T>::Color3 (const Color3 &c): Vec3 <T> (c)
{
    // empty
}

template <class T>
template <class S>
inline
Color3<T>::Color3 (const Vec3<S> &v): Vec3 <T> (v)
{
    //empty
}

template <class T>
inline const Color3<T> &
Color3<T>::operator = (const Color3 &c)
{
    *((Vec3<T> *) this) = c;
    return *this;
}

template <class T>
inline const Color3<T> &
Color3<T>::operator += (const Color3 &c)
{
    *((Vec3<T> *) this) += c;
    return *this;
}

template <class T>
inline Color3<T>	
Color3<T>::operator + (const Color3 &c) const
{
    return Color3 (*(Vec3<T> *)this + (const Vec3<T> &)c);
}

template <class T>
inline const Color3<T> &
Color3<T>::operator -= (const Color3 &c)
{
    *((Vec3<T> *) this) -= c;
    return *this;
}

template <class T>
inline Color3<T>	
Color3<T>::operator - (const Color3 &c) const
{
    return Color3 (*(Vec3<T> *)this - (const Vec3<T> &)c);
}

template <class T>
inline Color3<T>	
Color3<T>::operator - () const
{
    return Color3 (-(*(Vec3<T> *)this));
}

template <class T>
inline const Color3<T> &
Color3<T>::negate ()
{
    ((Vec3<T> *) this)->negate();
    return *this;
}

template <class T>
inline const Color3<T> &
Color3<T>::operator *= (const Color3 &c)
{
    *((Vec3<T> *) this) *= c;
    return *this;
}

template <class T>
inline const Color3<T> &
Color3<T>::operator *= (T a)
{
    *((Vec3<T> *) this) *= a;
    return *this;
}

template <class T>
inline Color3<T>	
Color3<T>::operator * (const Color3 &c) const
{
    return Color3 (*(Vec3<T> *)this * (const Vec3<T> &)c);
}

template <class T>
inline Color3<T>	
Color3<T>::operator * (T a) const
{
    return Color3 (*(Vec3<T> *)this * a);
}

template <class T>
inline const Color3<T> &
Color3<T>::operator /= (const Color3 &c)
{
    *((Vec3<T> *) this) /= c;
    return *this;
}

template <class T>
inline const Color3<T> &
Color3<T>::operator /= (T a)
{
    *((Vec3<T> *) this) /= a;
    return *this;
}

template <class T>
inline Color3<T>	
Color3<T>::operator / (const Color3 &c) const
{
    return Color3 (*(Vec3<T> *)this / (const Vec3<T> &)c);
}

template <class T>
inline Color3<T>	
Color3<T>::operator / (T a) const
{
    return Color3 (*(Vec3<T> *)this / a);
}

//-----------------------
// Implementation of Color4
//-----------------------

template <class T>
inline T &
Color4<T>::operator [] (int i)
{
    return (&r)[i];
}

template <class T>
inline const T &
Color4<T>::operator [] (int i) const
{
    return (&r)[i];
}

template <class T>
inline
Color4<T>::Color4 ()
{
    // empty
}

template <class T>
inline
Color4<T>::Color4 (T x)
{
    r = g = b = a = x;
}

template <class T>
inline
Color4<T>::Color4 (T x, T y, T z, T w)
{
    r = x;
    g = y;
    b = z;
    a = w;
}

template <class T>
inline
Color4<T>::Color4 (const Color4 &v)
{
    r = v.r;
    g = v.g;
    b = v.b;
    a = v.a;
}

template <class T>
template <class S>
inline
Color4<T>::Color4 (const Color4<S> &v)
{
    r = T (v.r);
    g = T (v.g);
    b = T (v.b);
    a = T (v.a);
}

template <class T>
inline const Color4<T> &
Color4<T>::operator = (const Color4 &v)
{
    r = v.r;
    g = v.g;
    b = v.b;
    a = v.a;
    return *this;
}

template <class T>
template <class S>
inline void
Color4<T>::setValue (S x, S y, S z, S w)
{
    r = T (x);
    g = T (y);
    b = T (z);
    a = T (w);
}

template <class T>
template <class S>
inline void
Color4<T>::setValue (const Color4<S> &v)
{
    r = T (v.r);
    g = T (v.g);
    b = T (v.b);
    a = T (v.a);
}

template <class T>
template <class S>
inline void
Color4<T>::getValue (S &x, S &y, S &z, S &w) const
{
    x = S (r);
    y = S (g);
    z = S (b);
    w = S (a);
}

template <class T>
template <class S>
inline void
Color4<T>::getValue (Color4<S> &v) const
{
    v.r = S (r);
    v.g = S (g);
    v.b = S (b);
    v.a = S (a);
}

template <class T>
inline T *
Color4<T>::getValue()
{
    return (T *) &r;
}

template <class T>
inline const T *
Color4<T>::getValue() const
{
    return (const T *) &r;
}

template <class T>
template <class S>
inline bool
Color4<T>::operator == (const Color4<S> &v) const
{
    return r == v.r && g == v.g && b == v.b && a == v.a;
}

template <class T>
template <class S>
inline bool
Color4<T>::operator != (const Color4<S> &v) const
{
    return r != v.r || g != v.g || b != v.b || a != v.a;
}

template <class T>
inline const Color4<T> &
Color4<T>::operator += (const Color4 &v)
{
    r += v.r;
    g += v.g;
    b += v.b;
    a += v.a;
    return *this;
}

template <class T>
inline Color4<T>
Color4<T>::operator + (const Color4 &v) const
{
    return Color4 (r + v.r, g + v.g, b + v.b, a + v.a);
}

template <class T>
inline const Color4<T> &
Color4<T>::operator -= (const Color4 &v)
{
    r -= v.r;
    g -= v.g;
    b -= v.b;
    a -= v.a;
    return *this;
}

template <class T>
inline Color4<T>
Color4<T>::operator - (const Color4 &v) const
{
    return Color4 (r - v.r, g - v.g, b - v.b, a - v.a);
}

template <class T>
inline Color4<T>
Color4<T>::operator - () const
{
    return Color4 (-r, -g, -b, -a);
}

template <class T>
inline const Color4<T> &
Color4<T>::negate ()
{
    r = -r;
    g = -g;
    b = -b;
    a = -a;
    return *this;
}

template <class T>
inline const Color4<T> &
Color4<T>::operator *= (const Color4 &v)
{
    r *= v.r;
    g *= v.g;
    b *= v.b;
    a *= v.a;
    return *this;
}

template <class T>
inline const Color4<T> &
Color4<T>::operator *= (T x)
{
    r *= x;
    g *= x;
    b *= x;
    a *= x;
    return *this;
}

template <class T>
inline Color4<T>
Color4<T>::operator * (const Color4 &v) const
{
    return Color4 (r * v.r, g * v.g, b * v.b, a * v.a);
}

template <class T>
inline Color4<T>
Color4<T>::operator * (T x) const
{
    return Color4 (r * x, g * x, b * x, a * x);
}

template <class T>
inline const Color4<T> &
Color4<T>::operator /= (const Color4 &v)
{
    r /= v.r;
    g /= v.g;
    b /= v.b;
    a /= v.a;
    return *this;
}

template <class T>
inline const Color4<T> &
Color4<T>::operator /= (T x)
{
    r /= x;
    g /= x;
    b /= x;
    a /= x;
    return *this;
}

template <class T>
inline Color4<T>
Color4<T>::operator / (const Color4 &v) const
{
    return Color4 (r / v.r, g / v.g, b / v.b, a / v.a);
}

template <class T>
inline Color4<T>
Color4<T>::operator / (T x) const
{
    return Color4 (r / x, g / x, b / x, a / x);
}


template <class T>
std::ostream &
operator << (std::ostream &s, const Color4<T> &v)
{
    return s << '(' << v.r << ' ' << v.g << ' ' << v.b << ' ' << v.a << ')';
}

//-----------------------------------------
// Implementation of reverse multiplication
//-----------------------------------------

template <class S, class T>
inline Color4<T>
operator * (S x, const Color4<T> &v)
{
    return Color4<T> (x * v.r, x * v.g, x * v.b, x * v.a);
}


IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHCOLOR_H 
