///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IMATHVEC_H
#define INCLUDED_IMATHVEC_H

//----------------------------------------------------
//
//	2D and 3D point/vector class templates!
//
//----------------------------------------------------

#include "ImathExc.h"
#include "ImathLimits.h"
#include "ImathMath.h"

#include <iostream>

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
// suppress exception specification warnings
#pragma warning(disable:4290)
#endif


namespace Imath {


template <class T> class Vec2
{
  public:

    //-------------------
    // Access to elements
    //-------------------

    T			x, y;

    T &			operator [] (int i);
    const T &		operator [] (int i) const;


    //-------------
    // Constructors
    //-------------

    Vec2 ();                        // no initialization
    explicit Vec2 (T a);            // (a a)
    Vec2 (T a, T b);                // (a b)


    //---------------------------------
    // Copy constructors and assignment
    //---------------------------------

    Vec2 (const Vec2 &v);
    template <class S> Vec2 (const Vec2<S> &v);

    const Vec2 &	operator = (const Vec2 &v);


    //----------------------
    // Compatibility with Sb
    //----------------------

    template <class S>
    void		setValue (S a, S b);

    template <class S>
    void		setValue (const Vec2<S> &v);

    template <class S>
    void		getValue (S &a, S &b) const;

    template <class S>
    void		getValue (Vec2<S> &v) const;

    T *			getValue ();
    const T *		getValue () const;

    
    //---------
    // Equality
    //---------

    template <class S>
    bool		operator == (const Vec2<S> &v) const;

    template <class S>
    bool		operator != (const Vec2<S> &v) const;


    //-----------------------------------------------------------------------
    // Compare two vectors and test if they are "approximately equal":
    //
    // equalWithAbsError (v, e)
    //
    //	    Returns true if the coefficients of this and v are the same with
    //	    an absolute error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e
    //
    // equalWithRelError (v, e)
    //
    //	    Returns true if the coefficients of this and v are the same with
    //	    a relative error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e * abs (this[i])
    //-----------------------------------------------------------------------

    bool		equalWithAbsError (const Vec2<T> &v, T e) const;
    bool		equalWithRelError (const Vec2<T> &v, T e) const;

    //------------
    // Dot product
    //------------

    T			dot (const Vec2 &v) const;
    T			operator ^ (const Vec2 &v) const;


    //------------------------------------------------
    // Right-handed cross product, i.e. z component of
    // Vec3 (this->x, this->y, 0) % Vec3 (v.x, v.y, 0)
    //------------------------------------------------

    T			cross (const Vec2 &v) const;
    T			operator % (const Vec2 &v) const;


    //------------------------
    // Component-wise addition
    //------------------------

    const Vec2 &	operator += (const Vec2 &v);
    Vec2		operator + (const Vec2 &v) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    const Vec2 &	operator -= (const Vec2 &v);
    Vec2		operator - (const Vec2 &v) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    Vec2		operator - () const;
    const Vec2 &	negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    const Vec2 &	operator *= (const Vec2 &v);
    const Vec2 &	operator *= (T a);
    Vec2		operator * (const Vec2 &v) const;
    Vec2		operator * (T a) const;


    //------------------------
    // Component-wise division
    //------------------------

    const Vec2 &	operator /= (const Vec2 &v);
    const Vec2 &	operator /= (T a);
    Vec2		operator / (const Vec2 &v) const;
    Vec2		operator / (T a) const;


    //----------------------------------------------------------------
    // Length and normalization:  If v.length() is 0.0, v.normalize()
    // and v.normalized() produce a null vector; v.normalizeExc() and
    // v.normalizedExc() throw a NullVecExc.
    // v.normalizeNonNull() and v.normalizedNonNull() are slightly
    // faster than the other normalization routines, but if v.length()
    // is 0.0, the result is undefined.
    //----------------------------------------------------------------

    T			length () const;
    T			length2 () const;

    const Vec2 &	normalize ();           // modifies *this
    const Vec2 &	normalizeExc () throw (Iex::MathExc);
    const Vec2 &	normalizeNonNull ();

    Vec2<T>		normalized () const;	// does not modify *this
    Vec2<T>		normalizedExc () const throw (Iex::MathExc);
    Vec2<T>		normalizedNonNull () const;


    //--------------------------------------------------------
    // Number of dimensions, i.e. number of elements in a Vec2
    //--------------------------------------------------------

    static unsigned int	dimensions() {return 2;}


    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    static T		baseTypeMin()		{return limits<T>::min();}
    static T		baseTypeMax()		{return limits<T>::max();}
    static T		baseTypeSmallest()	{return limits<T>::smallest();}
    static T		baseTypeEpsilon()	{return limits<T>::epsilon();}


    //--------------------------------------------------------------
    // Base type -- in templates, which accept a parameter, V, which
    // could be either a Vec2<T> or a Vec3<T>, you can refer to T as
    // V::BaseType
    //--------------------------------------------------------------

    typedef T		BaseType;
};


template <class T> class Vec3
{
  public:

    //-------------------
    // Access to elements
    //-------------------

    T			x, y, z;

    T &			operator [] (int i);
    const T &		operator [] (int i) const;


    //-------------
    // Constructors
    //-------------

    Vec3 ();			   // no initialization
    explicit Vec3 (T a);           // (a a a)
    Vec3 (T a, T b, T c);	   // (a b c)


    //---------------------------------
    // Copy constructors and assignment
    //---------------------------------

    Vec3 (const Vec3 &v);
    template <class S> Vec3 (const Vec3<S> &v);

    const Vec3 &	operator = (const Vec3 &v);


    //----------------------
    // Compatibility with Sb
    //----------------------

    template <class S>
    void		setValue (S a, S b, S c);

    template <class S>
    void		setValue (const Vec3<S> &v);

    template <class S>
    void		getValue (S &a, S &b, S &c) const;

    template <class S>
    void		getValue (Vec3<S> &v) const;

    T *			getValue();
    const T *		getValue() const;


    //---------
    // Equality
    //---------

    template <class S>
    bool		operator == (const Vec3<S> &v) const;

    template <class S>
    bool		operator != (const Vec3<S> &v) const;

    //-----------------------------------------------------------------------
    // Compare two vectors and test if they are "approximately equal":
    //
    // equalWithAbsError (v, e)
    //
    //	    Returns true if the coefficients of this and v are the same with
    //	    an absolute error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e
    //
    // equalWithRelError (v, e)
    //
    //	    Returns true if the coefficients of this and v are the same with
    //	    a relative error of no more than e, i.e., for all i
    //
    //      abs (this[i] - v[i]) <= e * abs (this[i])
    //-----------------------------------------------------------------------

    bool		equalWithAbsError (const Vec3<T> &v, T e) const;
    bool		equalWithRelError (const Vec3<T> &v, T e) const;

    //------------
    // Dot product
    //------------

    T			dot (const Vec3 &v) const;
    T			operator ^ (const Vec3 &v) const;


    //---------------------------
    // Right-handed cross product
    //---------------------------

    Vec3		cross (const Vec3 &v) const;
    const Vec3 &	operator %= (const Vec3 &v);
    Vec3		operator % (const Vec3 &v) const;


    //------------------------
    // Component-wise addition
    //------------------------

    const Vec3 &	operator += (const Vec3 &v);
    Vec3		operator + (const Vec3 &v) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    const Vec3 &	operator -= (const Vec3 &v);
    Vec3		operator - (const Vec3 &v) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    Vec3		operator - () const;
    const Vec3 &	negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    const Vec3 &	operator *= (const Vec3 &v);
    const Vec3 &	operator *= (T a);
    Vec3		operator * (const Vec3 &v) const;
    Vec3		operator * (T a) const;


    //------------------------
    // Component-wise division
    //------------------------

    const Vec3 &	operator /= (const Vec3 &v);
    const Vec3 &	operator /= (T a);
    Vec3		operator / (const Vec3 &v) const;
    Vec3		operator / (T a) const;


    //----------------------------------------------------------------
    // Length and normalization:  If v.length() is 0.0, v.normalize()
    // and v.normalized() produce a null vector; v.normalizeExc() and
    // v.normalizedExc() throw a NullVecExc.
    // v.normalizeNonNull() and v.normalizedNonNull() are slightly
    // faster than the other normalization routines, but if v.length()
    // is 0.0, the result is undefined.
    //----------------------------------------------------------------

    T			length () const;
    T			length2 () const;

    const Vec3 &	normalize ();           // modifies *this
    const Vec3 &	normalizeExc () throw (Iex::MathExc);
    const Vec3 &	normalizeNonNull ();

    Vec3<T>		normalized () const;	// does not modify *this
    Vec3<T>		normalizedExc () const throw (Iex::MathExc);
    Vec3<T>		normalizedNonNull () const;


    //--------------------------------------------------------
    // Number of dimensions, i.e. number of elements in a Vec3
    //--------------------------------------------------------

    static unsigned int	dimensions() {return 3;}


    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    static T		baseTypeMin()		{return limits<T>::min();}
    static T		baseTypeMax()		{return limits<T>::max();}
    static T		baseTypeSmallest()	{return limits<T>::smallest();}
    static T		baseTypeEpsilon()	{return limits<T>::epsilon();}


    //--------------------------------------------------------------
    // Base type -- in templates, which accept a parameter, V, which
    // could be either a Vec2<T> or a Vec3<T>, you can refer to T as
    // V::BaseType
    //--------------------------------------------------------------

    typedef T		BaseType;
};


//--------------
// Stream output
//--------------

template <class T>
std::ostream &	operator << (std::ostream &s, const Vec2<T> &v);

template <class T>
std::ostream &	operator << (std::ostream &s, const Vec3<T> &v);


//----------------------------------------------------
// Reverse multiplication: S * Vec2<T> and S * Vec3<T>
//----------------------------------------------------

template <class T> Vec2<T>	operator * (T a, const Vec2<T> &v);
template <class T> Vec3<T>	operator * (T a, const Vec3<T> &v);


//-------------------------
// Typedefs for convenience
//-------------------------

typedef Vec2 <short>  V2s;
typedef Vec2 <int>    V2i;
typedef Vec2 <float>  V2f;
typedef Vec2 <double> V2d;
typedef Vec3 <short>  V3s;
typedef Vec3 <int>    V3i;
typedef Vec3 <float>  V3f;
typedef Vec3 <double> V3d;


//-------------------------------------------------------------------
// Specializations for Vec2<short>, Vec2<int>, Vec3<short>, Vec3<int>
//-------------------------------------------------------------------

// Vec2<short>

template <> short
Vec2<short>::length () const;

template <> const Vec2<short> &
Vec2<short>::normalize ();

template <> const Vec2<short> &
Vec2<short>::normalizeExc () throw (Iex::MathExc);

template <> const Vec2<short> &
Vec2<short>::normalizeNonNull ();

template <> Vec2<short>
Vec2<short>::normalized () const;

template <> Vec2<short>
Vec2<short>::normalizedExc () const throw (Iex::MathExc);

template <> Vec2<short>
Vec2<short>::normalizedNonNull () const;


// Vec2<int>

template <> int
Vec2<int>::length () const;

template <> const Vec2<int> &
Vec2<int>::normalize ();

template <> const Vec2<int> &
Vec2<int>::normalizeExc () throw (Iex::MathExc);

template <> const Vec2<int> &
Vec2<int>::normalizeNonNull ();

template <> Vec2<int>
Vec2<int>::normalized () const;

template <> Vec2<int>
Vec2<int>::normalizedExc () const throw (Iex::MathExc);

template <> Vec2<int>
Vec2<int>::normalizedNonNull () const;


// Vec3<short>

template <> short
Vec3<short>::length () const;

template <> const Vec3<short> &
Vec3<short>::normalize ();

template <> const Vec3<short> &
Vec3<short>::normalizeExc () throw (Iex::MathExc);

template <> const Vec3<short> &
Vec3<short>::normalizeNonNull ();

template <> Vec3<short>
Vec3<short>::normalized () const;

template <> Vec3<short>
Vec3<short>::normalizedExc () const throw (Iex::MathExc);

template <> Vec3<short>
Vec3<short>::normalizedNonNull () const;


// Vec3<int>

template <> int
Vec3<int>::length () const;

template <> const Vec3<int> &
Vec3<int>::normalize ();

template <> const Vec3<int> &
Vec3<int>::normalizeExc () throw (Iex::MathExc);

template <> const Vec3<int> &
Vec3<int>::normalizeNonNull ();

template <> Vec3<int>
Vec3<int>::normalized () const;

template <> Vec3<int>
Vec3<int>::normalizedExc () const throw (Iex::MathExc);

template <> Vec3<int>
Vec3<int>::normalizedNonNull () const;


//------------------------
// Implementation of Vec2:
//------------------------

template <class T>
inline T &
Vec2<T>::operator [] (int i)
{
    return (&x)[i];
}

template <class T>
inline const T &
Vec2<T>::operator [] (int i) const
{
    return (&x)[i];
}

template <class T>
inline
Vec2<T>::Vec2 ()
{
    // empty
}

template <class T>
inline
Vec2<T>::Vec2 (T a)
{
    x = y = a;
}

template <class T>
inline
Vec2<T>::Vec2 (T a, T b)
{
    x = a;
    y = b;
}

template <class T>
inline
Vec2<T>::Vec2 (const Vec2 &v)
{
    x = v.x;
    y = v.y;
}

template <class T>
template <class S>
inline
Vec2<T>::Vec2 (const Vec2<S> &v)
{
    x = T (v.x);
    y = T (v.y);
}

template <class T>
inline const Vec2<T> &
Vec2<T>::operator = (const Vec2 &v)
{
    x = v.x;
    y = v.y;
    return *this;
}

template <class T>
template <class S>
inline void
Vec2<T>::setValue (S a, S b)
{
    x = T (a);
    y = T (b);
}

template <class T>
template <class S>
inline void
Vec2<T>::setValue (const Vec2<S> &v)
{
    x = T (v.x);
    y = T (v.y);
}

template <class T>
template <class S>
inline void
Vec2<T>::getValue (S &a, S &b) const
{
    a = S (x);
    b = S (y);
}

template <class T>
template <class S>
inline void
Vec2<T>::getValue (Vec2<S> &v) const
{
    v.x = S (x);
    v.y = S (y);
}

template <class T>
inline T *
Vec2<T>::getValue()
{
    return (T *) &x;
}

template <class T>
inline const T *
Vec2<T>::getValue() const
{
    return (const T *) &x;
}

template <class T>
template <class S>
inline bool
Vec2<T>::operator == (const Vec2<S> &v) const
{
    return x == v.x && y == v.y;
}

template <class T>
template <class S>
inline bool
Vec2<T>::operator != (const Vec2<S> &v) const
{
    return x != v.x || y != v.y;
}

template <class T>
bool
Vec2<T>::equalWithAbsError (const Vec2<T> &v, T e) const
{
    for (int i = 0; i < 2; i++)
	if (!Imath::equalWithAbsError ((*this)[i], v[i], e))
	    return false;

    return true;
}

template <class T>
bool
Vec2<T>::equalWithRelError (const Vec2<T> &v, T e) const
{
    for (int i = 0; i < 2; i++)
	if (!Imath::equalWithRelError ((*this)[i], v[i], e))
	    return false;

    return true;
}

template <class T>
inline T
Vec2<T>::dot (const Vec2 &v) const
{
    return x * v.x + y * v.y;
}

template <class T>
inline T
Vec2<T>::operator ^ (const Vec2 &v) const
{
    return dot (v);
}

template <class T>
inline T
Vec2<T>::cross (const Vec2 &v) const
{
    return x * v.y - y * v.x;

}

template <class T>
inline T
Vec2<T>::operator % (const Vec2 &v) const
{
    return x * v.y - y * v.x;
}

template <class T>
inline const Vec2<T> &
Vec2<T>::operator += (const Vec2 &v)
{
    x += v.x;
    y += v.y;
    return *this;
}

template <class T>
inline Vec2<T>
Vec2<T>::operator + (const Vec2 &v) const
{
    return Vec2 (x + v.x, y + v.y);
}

template <class T>
inline const Vec2<T> &
Vec2<T>::operator -= (const Vec2 &v)
{
    x -= v.x;
    y -= v.y;
    return *this;
}

template <class T>
inline Vec2<T>
Vec2<T>::operator - (const Vec2 &v) const
{
    return Vec2 (x - v.x, y - v.y);
}

template <class T>
inline Vec2<T>
Vec2<T>::operator - () const
{
    return Vec2 (-x, -y);
}

template <class T>
inline const Vec2<T> &
Vec2<T>::negate ()
{
    x = -x;
    y = -y;
    return *this;
}

template <class T>
inline const Vec2<T> &
Vec2<T>::operator *= (const Vec2 &v)
{
    x *= v.x;
    y *= v.y;
    return *this;
}

template <class T>
inline const Vec2<T> &
Vec2<T>::operator *= (T a)
{
    x *= a;
    y *= a;
    return *this;
}

template <class T>
inline Vec2<T>
Vec2<T>::operator * (const Vec2 &v) const
{
    return Vec2 (x * v.x, y * v.y);
}

template <class T>
inline Vec2<T>
Vec2<T>::operator * (T a) const
{
    return Vec2 (x * a, y * a);
}

template <class T>
inline const Vec2<T> &
Vec2<T>::operator /= (const Vec2 &v)
{
    x /= v.x;
    y /= v.y;
    return *this;
}

template <class T>
inline const Vec2<T> &
Vec2<T>::operator /= (T a)
{
    x /= a;
    y /= a;
    return *this;
}

template <class T>
inline Vec2<T>
Vec2<T>::operator / (const Vec2 &v) const
{
    return Vec2 (x / v.x, y / v.y);
}

template <class T>
inline Vec2<T>
Vec2<T>::operator / (T a) const
{
    return Vec2 (x / a, y / a);
}

template <class T>
inline T
Vec2<T>::length () const
{
    return Math<T>::sqrt (dot (*this));
}

template <class T>
inline T
Vec2<T>::length2 () const
{
    return dot (*this);
}

template <class T>
const Vec2<T> &
Vec2<T>::normalize ()
{
    T l = length();

    if (l != 0)
    {
	x /= l;
	y /= l;
    }

    return *this;
}

template <class T>
const Vec2<T> &
Vec2<T>::normalizeExc () throw (Iex::MathExc)
{
    T l = length();

    if (l == 0)
	throw NullVecExc ("Cannot normalize null vector.");

    x /= l;
    y /= l;
    return *this;
}

template <class T>
inline
const Vec2<T> &
Vec2<T>::normalizeNonNull ()
{
    T l = length();
    x /= l;
    y /= l;
    return *this;
}

template <class T>
Vec2<T>
Vec2<T>::normalized () const
{
    T l = length();

    if (l == 0)
	return Vec2 (T (0));

    return Vec2 (x / l, y / l);
}

template <class T>
Vec2<T>
Vec2<T>::normalizedExc () const throw (Iex::MathExc)
{
    T l = length();

    if (l == 0)
	throw NullVecExc ("Cannot normalize null vector.");

    return Vec2 (x / l, y / l);
}

template <class T>
inline
Vec2<T>
Vec2<T>::normalizedNonNull () const
{
    T l = length();
    return Vec2 (x / l, y / l);
}


//-----------------------
// Implementation of Vec3
//-----------------------

template <class T>
inline T &
Vec3<T>::operator [] (int i)
{
    return (&x)[i];
}

template <class T>
inline const T &
Vec3<T>::operator [] (int i) const
{
    return (&x)[i];
}

template <class T>
inline
Vec3<T>::Vec3 ()
{
    // empty
}

template <class T>
inline
Vec3<T>::Vec3 (T a)
{
    x = y = z = a;
}

template <class T>
inline
Vec3<T>::Vec3 (T a, T b, T c)
{
    x = a;
    y = b;
    z = c;
}

template <class T>
inline
Vec3<T>::Vec3 (const Vec3 &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
}

template <class T>
template <class S>
inline
Vec3<T>::Vec3 (const Vec3<S> &v)
{
    x = T (v.x);
    y = T (v.y);
    z = T (v.z);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator = (const Vec3 &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    return *this;
}

template <class T>
template <class S>
inline void
Vec3<T>::setValue (S a, S b, S c)
{
    x = T (a);
    y = T (b);
    z = T (c);
}

template <class T>
template <class S>
inline void
Vec3<T>::setValue (const Vec3<S> &v)
{
    x = T (v.x);
    y = T (v.y);
    z = T (v.z);
}

template <class T>
template <class S>
inline void
Vec3<T>::getValue (S &a, S &b, S &c) const
{
    a = S (x);
    b = S (y);
    c = S (z);
}

template <class T>
template <class S>
inline void
Vec3<T>::getValue (Vec3<S> &v) const
{
    v.x = S (x);
    v.y = S (y);
    v.z = S (z);
}

template <class T>
inline T *
Vec3<T>::getValue()
{
    return (T *) &x;
}

template <class T>
inline const T *
Vec3<T>::getValue() const
{
    return (const T *) &x;
}

template <class T>
template <class S>
inline bool
Vec3<T>::operator == (const Vec3<S> &v) const
{
    return x == v.x && y == v.y && z == v.z;
}

template <class T>
template <class S>
inline bool
Vec3<T>::operator != (const Vec3<S> &v) const
{
    return x != v.x || y != v.y || z != v.z;
}

template <class T>
bool
Vec3<T>::equalWithAbsError (const Vec3<T> &v, T e) const
{
    for (int i = 0; i < 3; i++)
	if (!Imath::equalWithAbsError ((*this)[i], v[i], e))
	    return false;

    return true;
}

template <class T>
bool
Vec3<T>::equalWithRelError (const Vec3<T> &v, T e) const
{
    for (int i = 0; i < 3; i++)
	if (!Imath::equalWithRelError ((*this)[i], v[i], e))
	    return false;

    return true;
}

template <class T>
inline T
Vec3<T>::dot (const Vec3 &v) const
{
    return x * v.x + y * v.y + z * v.z;
}

template <class T>
inline T
Vec3<T>::operator ^ (const Vec3 &v) const
{
    return dot (v);
}

template <class T>
inline Vec3<T>
Vec3<T>::cross (const Vec3 &v) const
{
    return Vec3 (y * v.z - z * v.y,
		 z * v.x - x * v.z,
		 x * v.y - y * v.x);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator %= (const Vec3 &v)
{
    T a = y * v.z - z * v.y;
    T b = z * v.x - x * v.z;
    T c = x * v.y - y * v.x;
    x = a;
    y = b;
    z = c;
    return *this;
}

template <class T>
inline Vec3<T>
Vec3<T>::operator % (const Vec3 &v) const
{
    return Vec3 (y * v.z - z * v.y,
		 z * v.x - x * v.z,
		 x * v.y - y * v.x);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator += (const Vec3 &v)
{
    x += v.x;
    y += v.y;
    z += v.z;
    return *this;
}

template <class T>
inline Vec3<T>
Vec3<T>::operator + (const Vec3 &v) const
{
    return Vec3 (x + v.x, y + v.y, z + v.z);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator -= (const Vec3 &v)
{
    x -= v.x;
    y -= v.y;
    z -= v.z;
    return *this;
}

template <class T>
inline Vec3<T>
Vec3<T>::operator - (const Vec3 &v) const
{
    return Vec3 (x - v.x, y - v.y, z - v.z);
}

template <class T>
inline Vec3<T>
Vec3<T>::operator - () const
{
    return Vec3 (-x, -y, -z);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::negate ()
{
    x = -x;
    y = -y;
    z = -z;
    return *this;
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator *= (const Vec3 &v)
{
    x *= v.x;
    y *= v.y;
    z *= v.z;
    return *this;
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator *= (T a)
{
    x *= a;
    y *= a;
    z *= a;
    return *this;
}

template <class T>
inline Vec3<T>
Vec3<T>::operator * (const Vec3 &v) const
{
    return Vec3 (x * v.x, y * v.y, z * v.z);
}

template <class T>
inline Vec3<T>
Vec3<T>::operator * (T a) const
{
    return Vec3 (x * a, y * a, z * a);
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator /= (const Vec3 &v)
{
    x /= v.x;
    y /= v.y;
    z /= v.z;
    return *this;
}

template <class T>
inline const Vec3<T> &
Vec3<T>::operator /= (T a)
{
    x /= a;
    y /= a;
    z /= a;
    return *this;
}

template <class T>
inline Vec3<T>
Vec3<T>::operator / (const Vec3 &v) const
{
    return Vec3 (x / v.x, y / v.y, z / v.z);
}

template <class T>
inline Vec3<T>
Vec3<T>::operator / (T a) const
{
    return Vec3 (x / a, y / a, z / a);
}


template <class T>
inline T
Vec3<T>::length () const
{
    return Math<T>::sqrt (dot (*this));
}

template <class T>
inline T
Vec3<T>::length2 () const
{
    return dot (*this);
}

template <class T>
const Vec3<T> &
Vec3<T>::normalize ()
{
    T l = length();

    if (l != 0)
    {
	x /= l;
	y /= l;
	z /= l;
    }

    return *this;
}

template <class T>
const Vec3<T> &
Vec3<T>::normalizeExc () throw (Iex::MathExc)
{
    T l = length();

    if (l == 0)
	throw NullVecExc ("Cannot normalize null vector.");

    x /= l;
    y /= l;
    z /= l;
    return *this;
}

template <class T>
inline
const Vec3<T> &
Vec3<T>::normalizeNonNull ()
{
    T l = length();
    x /= l;
    y /= l;
    z /= l;
    return *this;
}

template <class T>
Vec3<T>
Vec3<T>::normalized () const
{
    T l = length();

    if (l == 0)
	return Vec3 (T (0));

    return Vec3 (x / l, y / l, z / l);
}

template <class T>
Vec3<T>
Vec3<T>::normalizedExc () const throw (Iex::MathExc)
{
    T l = length();

    if (l == 0)
	throw NullVecExc ("Cannot normalize null vector.");

    return Vec3 (x / l, y / l, z / l);
}

template <class T>
inline
Vec3<T>
Vec3<T>::normalizedNonNull () const
{
    T l = length();
    return Vec3 (x / l, y / l, z / l);
}


//-----------------------------
// Stream output implementation
//-----------------------------

template <class T>
std::ostream &
operator << (std::ostream &s, const Vec2<T> &v)
{
    return s << '(' << v.x << ' ' << v.y << ')';
}

template <class T>
std::ostream &
operator << (std::ostream &s, const Vec3<T> &v)
{
    return s << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
}


//-----------------------------------------
// Implementation of reverse multiplication
//-----------------------------------------

template <class T>
inline Vec2<T>
operator * (T a, const Vec2<T> &v)
{
    return Vec2<T> (a * v.x, a * v.y);
}

template <class T>
inline Vec3<T>
operator * (T a, const Vec3<T> &v)
{
    return Vec3<T> (a * v.x, a * v.y, a * v.z);
}


#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
#pragma warning(default:4290)
#endif

} // namespace Imath

#endif
