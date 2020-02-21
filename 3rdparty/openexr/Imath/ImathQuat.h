///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002-2012, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IMATHQUAT_H
#define INCLUDED_IMATHQUAT_H

//----------------------------------------------------------------------
//
//	template class Quat<T>
//
//	"Quaternions came from Hamilton ... and have been an unmixed
//	evil to those who have touched them in any way. Vector is a
//	useless survival ... and has never been of the slightest use
//	to any creature."
//
//	    - Lord Kelvin
//
//	This class implements the quaternion numerical type -- you
//      will probably want to use this class to represent orientations
//	in R3 and to convert between various euler angle reps. You
//	should probably use Imath::Euler<> for that.
//
//----------------------------------------------------------------------

#include "ImathExc.h"
#include "ImathMatrix.h"
#include "ImathNamespace.h"

#include <iostream>
#include <algorithm>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
// Disable MS VC++ warnings about conversion from double to float
#pragma warning(disable:4244)
#endif

template <class T>
class Quat
{
  public:

    T			r;	    // real part
    Vec3<T>		v;	    // imaginary vector


    //-----------------------------------------------------
    //	Constructors - default constructor is identity quat
    //-----------------------------------------------------

    Quat ();

    template <class S>
    Quat (const Quat<S> &q);

    Quat (T s, T i, T j, T k);

    Quat (T s, Vec3<T> d);

    static Quat<T> identity ();


    //-------------------------------------------------
    //	Basic Algebra - Operators and Methods
    //  The operator return values are *NOT* normalized
    //
    //  operator^ and euclideanInnnerProduct() both
    //            implement the 4D dot product
    //
    //  operator/ uses the inverse() quaternion
    //
    //	operator~ is conjugate -- if (S+V) is quat then
    //		  the conjugate (S+V)* == (S-V)
    //
    //  some operators (*,/,*=,/=) treat the quat as
    //	a 4D vector when one of the operands is scalar
    //-------------------------------------------------

    const Quat<T> &	operator =	(const Quat<T> &q);
    const Quat<T> &	operator *=	(const Quat<T> &q);
    const Quat<T> &	operator *=	(T t);
    const Quat<T> &	operator /=	(const Quat<T> &q);
    const Quat<T> &	operator /=	(T t);
    const Quat<T> &	operator +=	(const Quat<T> &q);
    const Quat<T> &	operator -=	(const Quat<T> &q);
    T &			operator []	(int index);	// as 4D vector
    T			operator []	(int index) const;

    template <class S> bool operator == (const Quat<S> &q) const;
    template <class S> bool operator != (const Quat<S> &q) const;

    Quat<T> &		invert ();		// this -> 1 / this
    Quat<T>		inverse () const;
    Quat<T> &		normalize ();		// returns this
    Quat<T>		normalized () const;
    T			length () const;	// in R4
    Vec3<T>             rotateVector(const Vec3<T> &original) const;
    T                   euclideanInnerProduct(const Quat<T> &q) const;

    //-----------------------
    //	Rotation conversion
    //-----------------------

    Quat<T> &		setAxisAngle (const Vec3<T> &axis, T radians);

    Quat<T> &		setRotation (const Vec3<T> &fromDirection,
				     const Vec3<T> &toDirection);

    T			angle () const;
    Vec3<T>		axis () const;

    Matrix33<T>		toMatrix33 () const;
    Matrix44<T>		toMatrix44 () const;

    Quat<T>		log () const;
    Quat<T>		exp () const;


  private:

    void		setRotationInternal (const Vec3<T> &f0,
					     const Vec3<T> &t0,
					     Quat<T> &q);
};


template<class T>
Quat<T>			slerp (const Quat<T> &q1, const Quat<T> &q2, T t);

template<class T>
Quat<T>			slerpShortestArc
                              (const Quat<T> &q1, const Quat<T> &q2, T t);


template<class T>
Quat<T>			squad (const Quat<T> &q1, const Quat<T> &q2, 
			       const Quat<T> &qa, const Quat<T> &qb, T t);

template<class T>
void			intermediate (const Quat<T> &q0, const Quat<T> &q1, 
				      const Quat<T> &q2, const Quat<T> &q3,
				      Quat<T> &qa, Quat<T> &qb);

template<class T>
Matrix33<T>		operator * (const Matrix33<T> &M, const Quat<T> &q);

template<class T>
Matrix33<T>		operator * (const Quat<T> &q, const Matrix33<T> &M);

template<class T>
std::ostream &		operator << (std::ostream &o, const Quat<T> &q);

template<class T>
Quat<T>			operator * (const Quat<T> &q1, const Quat<T> &q2);

template<class T>
Quat<T>			operator / (const Quat<T> &q1, const Quat<T> &q2);

template<class T>
Quat<T>			operator / (const Quat<T> &q, T t);

template<class T>
Quat<T>			operator * (const Quat<T> &q, T t);

template<class T>
Quat<T>			operator * (T t, const Quat<T> &q);

template<class T>
Quat<T>			operator + (const Quat<T> &q1, const Quat<T> &q2);

template<class T>
Quat<T>			operator - (const Quat<T> &q1, const Quat<T> &q2);

template<class T>
Quat<T>			operator ~ (const Quat<T> &q);

template<class T>
Quat<T>			operator - (const Quat<T> &q);

template<class T>
Vec3<T>			operator * (const Vec3<T> &v, const Quat<T> &q);


//--------------------
// Convenient typedefs
//--------------------

typedef Quat<float>	Quatf;
typedef Quat<double>	Quatd;


//---------------
// Implementation
//---------------

template<class T>
inline
Quat<T>::Quat (): r (1), v (0, 0, 0)
{
    // empty
}


template<class T>
template <class S>
inline
Quat<T>::Quat (const Quat<S> &q): r (q.r), v (q.v)
{
    // empty
}


template<class T>
inline
Quat<T>::Quat (T s, T i, T j, T k): r (s), v (i, j, k)
{
    // empty
}


template<class T>
inline
Quat<T>::Quat (T s, Vec3<T> d): r (s), v (d)
{
    // empty
}


template<class T>
inline Quat<T>
Quat<T>::identity ()
{
    return Quat<T>();
}

template<class T>
inline const Quat<T> &
Quat<T>::operator = (const Quat<T> &q)
{
    r = q.r;
    v = q.v;
    return *this;
}


template<class T>
inline const Quat<T> &
Quat<T>::operator *= (const Quat<T> &q)
{
    T rtmp = r * q.r - (v ^ q.v);
    v = r * q.v + v * q.r + v % q.v;
    r = rtmp;
    return *this;
}


template<class T>
inline const Quat<T> &
Quat<T>::operator *= (T t)
{
    r *= t;
    v *= t;
    return *this;
}


template<class T>
inline const Quat<T> &
Quat<T>::operator /= (const Quat<T> &q)
{
    *this = *this * q.inverse();
    return *this;
}


template<class T>
inline const Quat<T> &
Quat<T>::operator /= (T t)
{
    r /= t;
    v /= t;
    return *this;
}


template<class T>
inline const Quat<T> &
Quat<T>::operator += (const Quat<T> &q)
{
    r += q.r;
    v += q.v;
    return *this;
}


template<class T>
inline const Quat<T> &
Quat<T>::operator -= (const Quat<T> &q)
{
    r -= q.r;
    v -= q.v;
    return *this;
}


template<class T>
inline T &
Quat<T>::operator [] (int index)
{
    return index ? v[index - 1] : r;
}


template<class T>
inline T
Quat<T>::operator [] (int index) const
{
    return index ? v[index - 1] : r;
}


template <class T>
template <class S>
inline bool
Quat<T>::operator == (const Quat<S> &q) const
{
    return r == q.r && v == q.v;
}


template <class T>
template <class S>
inline bool
Quat<T>::operator != (const Quat<S> &q) const
{
    return r != q.r || v != q.v;
}


template<class T>
inline T
operator ^ (const Quat<T>& q1 ,const Quat<T>& q2)
{
    return q1.r * q2.r + (q1.v ^ q2.v);
}


template <class T>
inline T
Quat<T>::length () const
{
    return Math<T>::sqrt (r * r + (v ^ v));
}


template <class T>
inline Quat<T> &
Quat<T>::normalize ()
{
    if (T l = length())
    {
	r /= l;
	v /= l;
    }
    else
    {
	r = 1;
	v = Vec3<T> (0);
    }

    return *this;
}


template <class T>
inline Quat<T>
Quat<T>::normalized () const
{
    if (T l = length())
	return Quat (r / l, v / l);

    return Quat();
}


template<class T>
inline Quat<T>
Quat<T>::inverse () const
{
    //
    // 1    Q*
    // - = ----   where Q* is conjugate (operator~)
    // Q   Q* Q   and (Q* Q) == Q ^ Q (4D dot)
    //

    T qdot = *this ^ *this;
    return Quat (r / qdot, -v / qdot);
}


template<class T>
inline Quat<T> &
Quat<T>::invert ()
{
    T qdot = (*this) ^ (*this);
    r /= qdot;
    v = -v / qdot;
    return *this;
}


template<class T>
inline Vec3<T>
Quat<T>::rotateVector(const Vec3<T>& original) const
{
    //
    // Given a vector p and a quaternion q (aka this),
    // calculate p' = qpq*
    //
    // Assumes unit quaternions (because non-unit
    // quaternions cannot be used to rotate vectors
    // anyway).
    //

    Quat<T> vec (0, original);  // temporarily promote grade of original
    Quat<T> inv (*this);
    inv.v *= -1;                // unit multiplicative inverse
    Quat<T> result = *this * vec * inv;
    return result.v;
}


template<class T>
inline T 
Quat<T>::euclideanInnerProduct (const Quat<T> &q) const
{
    return r * q.r + v.x * q.v.x + v.y * q.v.y + v.z * q.v.z;
}


template<class T>
T
angle4D (const Quat<T> &q1, const Quat<T> &q2)
{
    //
    // Compute the angle between two quaternions,
    // interpreting the quaternions as 4D vectors.
    //

    Quat<T> d = q1 - q2;
    T lengthD = Math<T>::sqrt (d ^ d);

    Quat<T> s = q1 + q2;
    T lengthS = Math<T>::sqrt (s ^ s);

    return 2 * Math<T>::atan2 (lengthD, lengthS);
}


template<class T>
Quat<T>
slerp (const Quat<T> &q1, const Quat<T> &q2, T t)
{
    //
    // Spherical linear interpolation.
    // Assumes q1 and q2 are normalized and that q1 != -q2.
    //
    // This method does *not* interpolate along the shortest
    // arc between q1 and q2.  If you desire interpolation
    // along the shortest arc, and q1^q2 is negative, then
    // consider calling slerpShortestArc(), below, or flipping
    // the second quaternion explicitly.
    //
    // The implementation of squad() depends on a slerp()
    // that interpolates as is, without the automatic
    // flipping.
    //
    // Don Hatch explains the method we use here on his
    // web page, The Right Way to Calculate Stuff, at
    // http://www.plunk.org/~hatch/rightway.php
    //

    T a = angle4D (q1, q2);
    T s = 1 - t;

    Quat<T> q = sinx_over_x (s * a) / sinx_over_x (a) * s * q1 +
	        sinx_over_x (t * a) / sinx_over_x (a) * t * q2;

    return q.normalized();
}


template<class T>
Quat<T>
slerpShortestArc (const Quat<T> &q1, const Quat<T> &q2, T t)
{
    //
    // Spherical linear interpolation along the shortest
    // arc from q1 to either q2 or -q2, whichever is closer.
    // Assumes q1 and q2 are unit quaternions.
    //

    if ((q1 ^ q2) >= 0)
        return slerp (q1, q2, t);
    else
        return slerp (q1, -q2, t);
}


template<class T>
Quat<T>
spline (const Quat<T> &q0, const Quat<T> &q1,
        const Quat<T> &q2, const Quat<T> &q3,
	T t)
{
    //
    // Spherical Cubic Spline Interpolation -
    // from Advanced Animation and Rendering
    // Techniques by Watt and Watt, Page 366:
    // A spherical curve is constructed using three
    // spherical linear interpolations of a quadrangle
    // of unit quaternions: q1, qa, qb, q2.
    // Given a set of quaternion keys: q0, q1, q2, q3,
    // this routine does the interpolation between
    // q1 and q2 by constructing two intermediate
    // quaternions: qa and qb. The qa and qb are 
    // computed by the intermediate function to 
    // guarantee the continuity of tangents across
    // adjacent cubic segments. The qa represents in-tangent
    // for q1 and the qb represents the out-tangent for q2.
    // 
    // The q1 q2 is the cubic segment being interpolated. 
    // The q0 is from the previous adjacent segment and q3 is 
    // from the next adjacent segment. The q0 and q3 are used
    // in computing qa and qb.
    // 

    Quat<T> qa = intermediate (q0, q1, q2);
    Quat<T> qb = intermediate (q1, q2, q3);
    Quat<T> result = squad (q1, qa, qb, q2, t);

    return result;
}


template<class T>
Quat<T>
squad (const Quat<T> &q1, const Quat<T> &qa,
       const Quat<T> &qb, const Quat<T> &q2,
       T t)
{
    //
    // Spherical Quadrangle Interpolation -
    // from Advanced Animation and Rendering
    // Techniques by Watt and Watt, Page 366:
    // It constructs a spherical cubic interpolation as 
    // a series of three spherical linear interpolations 
    // of a quadrangle of unit quaternions. 
    //     
  
    Quat<T> r1 = slerp (q1, q2, t);
    Quat<T> r2 = slerp (qa, qb, t);
    Quat<T> result = slerp (r1, r2, 2 * t * (1 - t));

    return result;
}


template<class T>
Quat<T>
intermediate (const Quat<T> &q0, const Quat<T> &q1, const Quat<T> &q2)
{
    //
    // From advanced Animation and Rendering
    // Techniques by Watt and Watt, Page 366:
    // computing the inner quadrangle 
    // points (qa and qb) to guarantee tangent
    // continuity.
    // 

    Quat<T> q1inv = q1.inverse();
    Quat<T> c1 = q1inv * q2;
    Quat<T> c2 = q1inv * q0;
    Quat<T> c3 = (T) (-0.25) * (c2.log() + c1.log());
    Quat<T> qa = q1 * c3.exp();
    qa.normalize();
    return qa;
}


template <class T>
inline Quat<T>
Quat<T>::log () const
{
    //
    // For unit quaternion, from Advanced Animation and 
    // Rendering Techniques by Watt and Watt, Page 366:
    //

    T theta = Math<T>::acos (std::min (r, (T) 1.0));

    if (theta == 0)
	return Quat<T> (0, v);
    
    T sintheta = Math<T>::sin (theta);
    
    T k;
    if (abs (sintheta) < 1 && abs (theta) >= limits<T>::max() * abs (sintheta))
	k = 1;
    else
	k = theta / sintheta;

    return Quat<T> ((T) 0, v.x * k, v.y * k, v.z * k);
}


template <class T>
inline Quat<T>
Quat<T>::exp () const
{
    //
    // For pure quaternion (zero scalar part):
    // from Advanced Animation and Rendering
    // Techniques by Watt and Watt, Page 366:
    //

    T theta = v.length();
    T sintheta = Math<T>::sin (theta);
    
    T k;
    if (abs (theta) < 1 && abs (sintheta) >= limits<T>::max() * abs (theta))
	k = 1;
    else
	k = sintheta / theta;

    T costheta = Math<T>::cos (theta);

    return Quat<T> (costheta, v.x * k, v.y * k, v.z * k);
}


template <class T>
inline T
Quat<T>::angle () const
{
    return 2 * Math<T>::atan2 (v.length(), r);
}


template <class T>
inline Vec3<T>
Quat<T>::axis () const
{
    return v.normalized();
}


template <class T>
inline Quat<T> &
Quat<T>::setAxisAngle (const Vec3<T> &axis, T radians)
{
    r = Math<T>::cos (radians / 2);
    v = axis.normalized() * Math<T>::sin (radians / 2);
    return *this;
}


template <class T>
Quat<T> &
Quat<T>::setRotation (const Vec3<T> &from, const Vec3<T> &to)
{
    //
    // Create a quaternion that rotates vector from into vector to,
    // such that the rotation is around an axis that is the cross
    // product of from and to.
    //
    // This function calls function setRotationInternal(), which is
    // numerically accurate only for rotation angles that are not much
    // greater than pi/2.  In order to achieve good accuracy for angles
    // greater than pi/2, we split large angles in half, and rotate in
    // two steps.
    //

    //
    // Normalize from and to, yielding f0 and t0.
    //

    Vec3<T> f0 = from.normalized();
    Vec3<T> t0 = to.normalized();

    if ((f0 ^ t0) >= 0)
    {
	//
	// The rotation angle is less than or equal to pi/2.
	//

	setRotationInternal (f0, t0, *this);
    }
    else
    {
	//
	// The angle is greater than pi/2.  After computing h0,
	// which is halfway between f0 and t0, we rotate first
	// from f0 to h0, then from h0 to t0.
	//

	Vec3<T> h0 = (f0 + t0).normalized();

	if ((h0 ^ h0) != 0)
	{
	    setRotationInternal (f0, h0, *this);

	    Quat<T> q;
	    setRotationInternal (h0, t0, q);

	    *this *= q;
	}
	else
	{
	    //
	    // f0 and t0 point in exactly opposite directions.
	    // Pick an arbitrary axis that is orthogonal to f0,
	    // and rotate by pi.
	    //

	    r = T (0);

	    Vec3<T> f02 = f0 * f0;

	    if (f02.x <= f02.y && f02.x <= f02.z)
		v = (f0 % Vec3<T> (1, 0, 0)).normalized();
	    else if (f02.y <= f02.z)
		v = (f0 % Vec3<T> (0, 1, 0)).normalized();
	    else
		v = (f0 % Vec3<T> (0, 0, 1)).normalized();
	}
    }

    return *this;
}


template <class T>
void
Quat<T>::setRotationInternal (const Vec3<T> &f0, const Vec3<T> &t0, Quat<T> &q)
{
    //
    // The following is equivalent to setAxisAngle(n,2*phi),
    // where the rotation axis, n, is orthogonal to the f0 and
    // t0 vectors, and 2*phi is the angle between f0 and t0.
    //
    // This function is called by setRotation(), above; it assumes
    // that f0 and t0 are normalized and that the angle between
    // them is not much greater than pi/2.  This function becomes
    // numerically inaccurate if f0 and t0 point into nearly
    // opposite directions.
    //

    //
    // Find a normalized vector, h0, that is halfway between f0 and t0.
    // The angle between f0 and h0 is phi.
    //

    Vec3<T> h0 = (f0 + t0).normalized();

    //
    // Store the rotation axis and rotation angle.
    //

    q.r = f0 ^ h0;	//  f0 ^ h0 == cos (phi)
    q.v = f0 % h0;	// (f0 % h0).length() == sin (phi)
}


template<class T>
Matrix33<T>
Quat<T>::toMatrix33() const
{
    return Matrix33<T> (1 - 2 * (v.y * v.y + v.z * v.z),
			    2 * (v.x * v.y + v.z * r),
			    2 * (v.z * v.x - v.y * r),

			    2 * (v.x * v.y - v.z * r),
		        1 - 2 * (v.z * v.z + v.x * v.x),
			    2 * (v.y * v.z + v.x * r),

			    2 * (v.z * v.x + v.y * r),
			    2 * (v.y * v.z - v.x * r),
		        1 - 2 * (v.y * v.y + v.x * v.x));
}

template<class T>
Matrix44<T>
Quat<T>::toMatrix44() const
{
    return Matrix44<T> (1 - 2 * (v.y * v.y + v.z * v.z),
			    2 * (v.x * v.y + v.z * r),
			    2 * (v.z * v.x - v.y * r),
			    0,
			    2 * (v.x * v.y - v.z * r),
		        1 - 2 * (v.z * v.z + v.x * v.x),
			    2 * (v.y * v.z + v.x * r),
			    0,
			    2 * (v.z * v.x + v.y * r),
			    2 * (v.y * v.z - v.x * r),
		        1 - 2 * (v.y * v.y + v.x * v.x),
			    0,
			    0,
			    0,
			    0,
			    1);
}


template<class T>
inline Matrix33<T>
operator * (const Matrix33<T> &M, const Quat<T> &q)
{
    return M * q.toMatrix33();
}


template<class T>
inline Matrix33<T>
operator * (const Quat<T> &q, const Matrix33<T> &M)
{
    return q.toMatrix33() * M;
}


template<class T>
std::ostream &
operator << (std::ostream &o, const Quat<T> &q)
{
    return o << "(" << q.r
	     << " " << q.v.x
	     << " " << q.v.y
	     << " " << q.v.z
	     << ")";
}


template<class T>
inline Quat<T>
operator * (const Quat<T> &q1, const Quat<T> &q2)
{
    return Quat<T> (q1.r * q2.r - (q1.v ^ q2.v),
		    q1.r * q2.v + q1.v * q2.r + q1.v % q2.v);
}


template<class T>
inline Quat<T>
operator / (const Quat<T> &q1, const Quat<T> &q2)
{
    return q1 * q2.inverse();
}


template<class T>
inline Quat<T>
operator / (const Quat<T> &q, T t)
{
    return Quat<T> (q.r / t, q.v / t);
}


template<class T>
inline Quat<T>
operator * (const Quat<T> &q, T t)
{
    return Quat<T> (q.r * t, q.v * t);
}


template<class T>
inline Quat<T>
operator * (T t, const Quat<T> &q)
{
    return Quat<T> (q.r * t, q.v * t);
}


template<class T>
inline Quat<T>
operator + (const Quat<T> &q1, const Quat<T> &q2)
{
    return Quat<T> (q1.r + q2.r, q1.v + q2.v);
}


template<class T>
inline Quat<T>
operator - (const Quat<T> &q1, const Quat<T> &q2)
{
    return Quat<T> (q1.r - q2.r, q1.v - q2.v);
}


template<class T>
inline Quat<T>
operator ~ (const Quat<T> &q)
{
    return Quat<T> (q.r, -q.v);
}


template<class T>
inline Quat<T>
operator - (const Quat<T> &q)
{
    return Quat<T> (-q.r, -q.v);
}


template<class T>
inline Vec3<T>
operator * (const Vec3<T> &v, const Quat<T> &q)
{
    Vec3<T> a = q.v % v;
    Vec3<T> b = q.v % a;
    return v + T (2) * (q.r * a + b);
}

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
#pragma warning(default:4244)
#endif

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHQUAT_H
