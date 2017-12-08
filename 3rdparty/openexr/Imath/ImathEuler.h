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



#ifndef INCLUDED_IMATHEULER_H
#define INCLUDED_IMATHEULER_H

//----------------------------------------------------------------------
//
//	template class Euler<T>
//
//      This class represents euler angle orientations. The class
//	inherits from Vec3 to it can be freely cast. The additional
//	information is the euler priorities rep. This class is
//	essentially a rip off of Ken Shoemake's GemsIV code. It has
//	been modified minimally to make it more understandable, but
//	hardly enough to make it easy to grok completely.
//
//	There are 24 possible combonations of Euler angle
//	representations of which 12 are common in CG and you will
//	probably only use 6 of these which in this scheme are the
//	non-relative-non-repeating types. 
//
//	The representations can be partitioned according to two
//	criteria:
//
//	   1) Are the angles measured relative to a set of fixed axis
//	      or relative to each other (the latter being what happens
//	      when rotation matrices are multiplied together and is
//	      almost ubiquitous in the cg community)
//
//	   2) Is one of the rotations repeated (ala XYX rotation)
//
//	When you construct a given representation from scratch you
//	must order the angles according to their priorities. So, the
//	easiest is a softimage or aerospace (yaw/pitch/roll) ordering
//	of ZYX. 
//
//	    float x_rot = 1;
//	    float y_rot = 2;
//	    float z_rot = 3;
//
//	    Eulerf angles(z_rot, y_rot, x_rot, Eulerf::ZYX);
//		-or-
//	    Eulerf angles( V3f(z_rot,y_rot,z_rot), Eulerf::ZYX );
//
//	If instead, the order was YXZ for instance you would have to
//	do this:
//
//	    float x_rot = 1;
//	    float y_rot = 2;
//	    float z_rot = 3;
//
//	    Eulerf angles(y_rot, x_rot, z_rot, Eulerf::YXZ);
//		-or-
//	    Eulerf angles( V3f(y_rot,x_rot,z_rot), Eulerf::YXZ );
//
//	Notice how the order you put the angles into the three slots
//	should correspond to the enum (YXZ) ordering. The input angle
//	vector is called the "ijk" vector -- not an "xyz" vector. The
//	ijk vector order is the same as the enum. If you treat the
//	Euler<> as a Vec<> (which it inherts from) you will find the
//	angles are ordered in the same way, i.e.:
//
//	    V3f v = angles;
//	    // v.x == y_rot, v.y == x_rot, v.z == z_rot
//
//	If you just want the x, y, and z angles stored in a vector in
//	that order, you can do this:
//
//	    V3f v = angles.toXYZVector()
//	    // v.x == x_rot, v.y == y_rot, v.z == z_rot
//
//	If you want to set the Euler with an XYZVector use the
//	optional layout argument:
//
//	    Eulerf angles(x_rot, y_rot, z_rot, 
//			  Eulerf::YXZ,
//		          Eulerf::XYZLayout);
//
//	This is the same as:
//
//	    Eulerf angles(y_rot, x_rot, z_rot, Eulerf::YXZ);
//	    
//	Note that this won't do anything intelligent if you have a
//	repeated axis in the euler angles (e.g. XYX)
//
//	If you need to use the "relative" versions of these, you will
//	need to use the "r" enums. 
//
//      The units of the rotation angles are assumed to be radians.
//
//----------------------------------------------------------------------


#include "ImathMath.h"
#include "ImathVec.h"
#include "ImathQuat.h"
#include "ImathMatrix.h"
#include "ImathLimits.h"
#include "ImathNamespace.h"

#include <iostream>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
// Disable MS VC++ warnings about conversion from double to float
#pragma warning(disable:4244)
#endif

template <class T>
class Euler : public Vec3<T>
{
  public:
 
    using Vec3<T>::x;
    using Vec3<T>::y;
    using Vec3<T>::z;

    enum Order
    {
	//
	//  All 24 possible orderings
	//

	XYZ	= 0x0101,	// "usual" orderings
	XZY	= 0x0001,
	YZX	= 0x1101,
	YXZ	= 0x1001,
	ZXY	= 0x2101,
	ZYX	= 0x2001,
	
	XZX	= 0x0011,	// first axis repeated
	XYX	= 0x0111,
	YXY	= 0x1011,
	YZY	= 0x1111,
	ZYZ	= 0x2011,
	ZXZ	= 0x2111,

	XYZr	= 0x2000,	// relative orderings -- not common
	XZYr	= 0x2100,
	YZXr	= 0x1000,
	YXZr	= 0x1100,
	ZXYr	= 0x0000,
	ZYXr	= 0x0100,
	
	XZXr	= 0x2110,	// relative first axis repeated 
	XYXr	= 0x2010,
	YXYr	= 0x1110,
	YZYr	= 0x1010,
	ZYZr	= 0x0110,
	ZXZr	= 0x0010,
	//          ||||
	//          VVVV
	//  Legend: ABCD
	//  A -> Initial Axis (0==x, 1==y, 2==z)
	//  B -> Parity Even (1==true)
	//  C -> Initial Repeated (1==true)
	//  D -> Frame Static (1==true)
	//

	Legal	=   XYZ | XZY | YZX | YXZ | ZXY | ZYX |
		    XZX | XYX | YXY | YZY | ZYZ | ZXZ |
		    XYZr| XZYr| YZXr| YXZr| ZXYr| ZYXr|
		    XZXr| XYXr| YXYr| YZYr| ZYZr| ZXZr,

	Min	= 0x0000,
	Max	= 0x2111,
	Default	= XYZ
    };

    enum Axis { X = 0, Y = 1, Z = 2 };

    enum InputLayout { XYZLayout, IJKLayout };

    //--------------------------------------------------------------------
    //	Constructors -- all default to ZYX non-relative ala softimage
    //			(where there is no argument to specify it)
    //
    // The Euler-from-matrix constructors assume that the matrix does
    // not include shear or non-uniform scaling, but the constructors
    // do not examine the matrix to verify this assumption.  If necessary,
    // you can adjust the matrix by calling the removeScalingAndShear()
    // function, defined in ImathMatrixAlgo.h.
    //--------------------------------------------------------------------

    Euler();
    Euler(const Euler&);
    Euler(Order p);
    Euler(const Vec3<T> &v, Order o = Default, InputLayout l = IJKLayout);
    Euler(T i, T j, T k, Order o = Default, InputLayout l = IJKLayout);
    Euler(const Euler<T> &euler, Order newp);
    Euler(const Matrix33<T> &, Order o = Default);
    Euler(const Matrix44<T> &, Order o = Default);

    //---------------------------------
    //  Algebraic functions/ Operators
    //---------------------------------

    const Euler<T>&	operator=  (const Euler<T>&);
    const Euler<T>&	operator=  (const Vec3<T>&);

    //--------------------------------------------------------
    //	Set the euler value
    //  This does NOT convert the angles, but setXYZVector() 
    //	does reorder the input vector.
    //--------------------------------------------------------

    static bool		legal(Order);

    void		setXYZVector(const Vec3<T> &);

    Order		order() const;
    void		setOrder(Order);

    void		set(Axis initial,
			    bool relative,
			    bool parityEven,
			    bool firstRepeats);

    //------------------------------------------------------------
    //	Conversions, toXYZVector() reorders the angles so that
    //  the X rotation comes first, followed by the Y and Z
    //  in cases like XYX ordering, the repeated angle will be
    //	in the "z" component
    //
    // The Euler-from-matrix extract() functions assume that the
    // matrix does not include shear or non-uniform scaling, but
    // the extract() functions do not examine the matrix to verify
    // this assumption.  If necessary, you can adjust the matrix
    // by calling the removeScalingAndShear() function, defined
    // in ImathMatrixAlgo.h.
    //------------------------------------------------------------

    void		extract(const Matrix33<T>&);
    void		extract(const Matrix44<T>&);
    void		extract(const Quat<T>&);

    Matrix33<T>		toMatrix33() const;
    Matrix44<T>		toMatrix44() const;
    Quat<T>		toQuat() const;
    Vec3<T>		toXYZVector() const;

    //---------------------------------------------------
    //	Use this function to unpack angles from ijk form
    //---------------------------------------------------

    void		angleOrder(int &i, int &j, int &k) const;

    //---------------------------------------------------
    //	Use this function to determine mapping from xyz to ijk
    // - reshuffles the xyz to match the order
    //---------------------------------------------------
    
    void		angleMapping(int &i, int &j, int &k) const;

    //----------------------------------------------------------------------
    //
    //  Utility methods for getting continuous rotations. None of these
    //  methods change the orientation given by its inputs (or at least
    //  that is the intent).
    //
    //    angleMod() converts an angle to its equivalent in [-PI, PI]
    //
    //    simpleXYZRotation() adjusts xyzRot so that its components differ
    //                        from targetXyzRot by no more than +-PI
    //
    //    nearestRotation() adjusts xyzRot so that its components differ
    //                      from targetXyzRot by as little as possible.
    //                      Note that xyz here really means ijk, because
    //                      the order must be provided.
    //
    //    makeNear() adjusts "this" Euler so that its components differ
    //               from target by as little as possible. This method
    //               might not make sense for Eulers with different order
    //               and it probably doesn't work for repeated axis and
    //               relative orderings (TODO).
    //
    //-----------------------------------------------------------------------

    static float	angleMod (T angle);
    static void		simpleXYZRotation (Vec3<T> &xyzRot,
					   const Vec3<T> &targetXyzRot);
    static void		nearestRotation (Vec3<T> &xyzRot,
					 const Vec3<T> &targetXyzRot,
					 Order order = XYZ);

    void		makeNear (const Euler<T> &target);

    bool		frameStatic() const { return _frameStatic; }
    bool		initialRepeated() const { return _initialRepeated; }
    bool		parityEven() const { return _parityEven; }
    Axis		initialAxis() const { return _initialAxis; }

  protected:

    bool		_frameStatic	 : 1;	// relative or static rotations
    bool		_initialRepeated : 1;	// init axis repeated as last
    bool		_parityEven	 : 1;	// "parity of axis permutation"
#if defined _WIN32 || defined _WIN64
    Axis		_initialAxis	 ;	// First axis of rotation
#else
    Axis		_initialAxis	 : 2;	// First axis of rotation
#endif
};


//--------------------
// Convenient typedefs
//--------------------

typedef Euler<float>	Eulerf;
typedef Euler<double>	Eulerd;


//---------------
// Implementation
//---------------

template<class T>
inline void
 Euler<T>::angleOrder(int &i, int &j, int &k) const
{
    i = _initialAxis;
    j = _parityEven ? (i+1)%3 : (i > 0 ? i-1 : 2);
    k = _parityEven ? (i > 0 ? i-1 : 2) : (i+1)%3;
}

template<class T>
inline void
 Euler<T>::angleMapping(int &i, int &j, int &k) const
{
    int m[3];

    m[_initialAxis] = 0;
    m[(_initialAxis+1) % 3] = _parityEven ? 1 : 2;
    m[(_initialAxis+2) % 3] = _parityEven ? 2 : 1;
    i = m[0];
    j = m[1];
    k = m[2];
}

template<class T>
inline void
Euler<T>::setXYZVector(const Vec3<T> &v)
{
    int i,j,k;
    angleMapping(i,j,k);
    (*this)[i] = v.x;
    (*this)[j] = v.y;
    (*this)[k] = v.z;
}

template<class T>
inline Vec3<T>
Euler<T>::toXYZVector() const
{
    int i,j,k;
    angleMapping(i,j,k);
    return Vec3<T>((*this)[i],(*this)[j],(*this)[k]);
}


template<class T>
Euler<T>::Euler() :
    Vec3<T>(0,0,0),
    _frameStatic(true),
    _initialRepeated(false),
    _parityEven(true),
    _initialAxis(X)
{}

template<class T>
Euler<T>::Euler(typename Euler<T>::Order p) :
    Vec3<T>(0,0,0),
    _frameStatic(true),
    _initialRepeated(false),
    _parityEven(true),
    _initialAxis(X)
{
    setOrder(p);
}

template<class T>
inline Euler<T>::Euler( const Vec3<T> &v, 
			typename Euler<T>::Order p, 
			typename Euler<T>::InputLayout l ) 
{
    setOrder(p); 
    if ( l == XYZLayout ) setXYZVector(v);
    else { x = v.x; y = v.y; z = v.z; }
}

template<class T>
inline Euler<T>::Euler(const Euler<T> &euler)
{
    operator=(euler);
}

template<class T>
inline Euler<T>::Euler(const Euler<T> &euler,Order p)
{
    setOrder(p);
    Matrix33<T> M = euler.toMatrix33();
    extract(M);
}

template<class T>
inline Euler<T>::Euler( T xi, T yi, T zi, 
			typename Euler<T>::Order p,
			typename Euler<T>::InputLayout l)
{
    setOrder(p);
    if ( l == XYZLayout ) setXYZVector(Vec3<T>(xi,yi,zi));
    else { x = xi; y = yi; z = zi; }
}

template<class T>
inline Euler<T>::Euler( const Matrix33<T> &M, typename Euler::Order p )
{
    setOrder(p);
    extract(M);
}

template<class T>
inline Euler<T>::Euler( const Matrix44<T> &M, typename Euler::Order p )
{
    setOrder(p);
    extract(M);
}

template<class T>
inline void Euler<T>::extract(const Quat<T> &q)
{
    extract(q.toMatrix33());
}

template<class T>
void Euler<T>::extract(const Matrix33<T> &M)
{
    int i,j,k;
    angleOrder(i,j,k);

    if (_initialRepeated)
    {
	//
	// Extract the first angle, x.
	// 

	x = Math<T>::atan2 (M[j][i], M[k][i]);

	//
	// Remove the x rotation from M, so that the remaining
	// rotation, N, is only around two axes, and gimbal lock
	// cannot occur.
	//

	Vec3<T> r (0, 0, 0);
	r[i] = (_parityEven? -x: x);

	Matrix44<T> N;
	N.rotate (r);

	N = N * Matrix44<T> (M[0][0], M[0][1], M[0][2], 0,
			     M[1][0], M[1][1], M[1][2], 0,
			     M[2][0], M[2][1], M[2][2], 0,
			     0,       0,       0,       1);
	//
	// Extract the other two angles, y and z, from N.
	//

	T sy = Math<T>::sqrt (N[j][i]*N[j][i] + N[k][i]*N[k][i]);
	y = Math<T>::atan2 (sy, N[i][i]);
	z = Math<T>::atan2 (N[j][k], N[j][j]);
    }
    else
    {
	//
	// Extract the first angle, x.
	// 

	x = Math<T>::atan2 (M[j][k], M[k][k]);

	//
	// Remove the x rotation from M, so that the remaining
	// rotation, N, is only around two axes, and gimbal lock
	// cannot occur.
	//

	Vec3<T> r (0, 0, 0);
	r[i] = (_parityEven? -x: x);

	Matrix44<T> N;
	N.rotate (r);

	N = N * Matrix44<T> (M[0][0], M[0][1], M[0][2], 0,
			     M[1][0], M[1][1], M[1][2], 0,
			     M[2][0], M[2][1], M[2][2], 0,
			     0,       0,       0,       1);
	//
	// Extract the other two angles, y and z, from N.
	//

	T cy = Math<T>::sqrt (N[i][i]*N[i][i] + N[i][j]*N[i][j]);
	y = Math<T>::atan2 (-N[i][k], cy);
	z = Math<T>::atan2 (-N[j][i], N[j][j]);
    }

    if (!_parityEven)
	*this *= -1;

    if (!_frameStatic)
    {
	T t = x;
	x = z;
	z = t;
    }
}

template<class T>
void Euler<T>::extract(const Matrix44<T> &M)
{
    int i,j,k;
    angleOrder(i,j,k);

    if (_initialRepeated)
    {
	//
	// Extract the first angle, x.
	// 

	x = Math<T>::atan2 (M[j][i], M[k][i]);

	//
	// Remove the x rotation from M, so that the remaining
	// rotation, N, is only around two axes, and gimbal lock
	// cannot occur.
	//

	Vec3<T> r (0, 0, 0);
	r[i] = (_parityEven? -x: x);

	Matrix44<T> N;
	N.rotate (r);
	N = N * M;

	//
	// Extract the other two angles, y and z, from N.
	//

	T sy = Math<T>::sqrt (N[j][i]*N[j][i] + N[k][i]*N[k][i]);
	y = Math<T>::atan2 (sy, N[i][i]);
	z = Math<T>::atan2 (N[j][k], N[j][j]);
    }
    else
    {
	//
	// Extract the first angle, x.
	// 

	x = Math<T>::atan2 (M[j][k], M[k][k]);

	//
	// Remove the x rotation from M, so that the remaining
	// rotation, N, is only around two axes, and gimbal lock
	// cannot occur.
	//

	Vec3<T> r (0, 0, 0);
	r[i] = (_parityEven? -x: x);

	Matrix44<T> N;
	N.rotate (r);
	N = N * M;

	//
	// Extract the other two angles, y and z, from N.
	//

	T cy = Math<T>::sqrt (N[i][i]*N[i][i] + N[i][j]*N[i][j]);
	y = Math<T>::atan2 (-N[i][k], cy);
	z = Math<T>::atan2 (-N[j][i], N[j][j]);
    }

    if (!_parityEven)
	*this *= -1;

    if (!_frameStatic)
    {
	T t = x;
	x = z;
	z = t;
    }
}

template<class T>
Matrix33<T> Euler<T>::toMatrix33() const
{
    int i,j,k;
    angleOrder(i,j,k);

    Vec3<T> angles;

    if ( _frameStatic ) angles = (*this);
    else angles = Vec3<T>(z,y,x);

    if ( !_parityEven ) angles *= -1.0;

    T ci = Math<T>::cos(angles.x);
    T cj = Math<T>::cos(angles.y);
    T ch = Math<T>::cos(angles.z);
    T si = Math<T>::sin(angles.x);
    T sj = Math<T>::sin(angles.y);
    T sh = Math<T>::sin(angles.z);

    T cc = ci*ch;
    T cs = ci*sh;
    T sc = si*ch;
    T ss = si*sh;

    Matrix33<T> M;

    if ( _initialRepeated )
    {
	M[i][i] = cj;	  M[j][i] =  sj*si;    M[k][i] =  sj*ci;
	M[i][j] = sj*sh;  M[j][j] = -cj*ss+cc; M[k][j] = -cj*cs-sc;
	M[i][k] = -sj*ch; M[j][k] =  cj*sc+cs; M[k][k] =  cj*cc-ss;
    }
    else
    {
	M[i][i] = cj*ch; M[j][i] = sj*sc-cs; M[k][i] = sj*cc+ss;
	M[i][j] = cj*sh; M[j][j] = sj*ss+cc; M[k][j] = sj*cs-sc;
	M[i][k] = -sj;	 M[j][k] = cj*si;    M[k][k] = cj*ci;
    }

    return M;
}

template<class T>
Matrix44<T> Euler<T>::toMatrix44() const
{
    int i,j,k;
    angleOrder(i,j,k);

    Vec3<T> angles;

    if ( _frameStatic ) angles = (*this);
    else angles = Vec3<T>(z,y,x);

    if ( !_parityEven ) angles *= -1.0;

    T ci = Math<T>::cos(angles.x);
    T cj = Math<T>::cos(angles.y);
    T ch = Math<T>::cos(angles.z);
    T si = Math<T>::sin(angles.x);
    T sj = Math<T>::sin(angles.y);
    T sh = Math<T>::sin(angles.z);

    T cc = ci*ch;
    T cs = ci*sh;
    T sc = si*ch;
    T ss = si*sh;

    Matrix44<T> M;

    if ( _initialRepeated )
    {
	M[i][i] = cj;	  M[j][i] =  sj*si;    M[k][i] =  sj*ci;
	M[i][j] = sj*sh;  M[j][j] = -cj*ss+cc; M[k][j] = -cj*cs-sc;
	M[i][k] = -sj*ch; M[j][k] =  cj*sc+cs; M[k][k] =  cj*cc-ss;
    }
    else
    {
	M[i][i] = cj*ch; M[j][i] = sj*sc-cs; M[k][i] = sj*cc+ss;
	M[i][j] = cj*sh; M[j][j] = sj*ss+cc; M[k][j] = sj*cs-sc;
	M[i][k] = -sj;	 M[j][k] = cj*si;    M[k][k] = cj*ci;
    }

    return M;
}

template<class T>
Quat<T> Euler<T>::toQuat() const
{
    Vec3<T> angles;
    int i,j,k;
    angleOrder(i,j,k);

    if ( _frameStatic ) angles = (*this);
    else angles = Vec3<T>(z,y,x);

    if ( !_parityEven ) angles.y = -angles.y;

    T ti = angles.x*0.5;
    T tj = angles.y*0.5;
    T th = angles.z*0.5;
    T ci = Math<T>::cos(ti);
    T cj = Math<T>::cos(tj);
    T ch = Math<T>::cos(th);
    T si = Math<T>::sin(ti);
    T sj = Math<T>::sin(tj);
    T sh = Math<T>::sin(th);
    T cc = ci*ch;
    T cs = ci*sh;
    T sc = si*ch;
    T ss = si*sh;

    T parity = _parityEven ? 1.0 : -1.0;

    Quat<T> q;
    Vec3<T> a;

    if ( _initialRepeated )
    {
	a[i]	= cj*(cs + sc);
	a[j]	= sj*(cc + ss) * parity,
	a[k]	= sj*(cs - sc);
	q.r	= cj*(cc - ss);
    }
    else
    {
	a[i]	= cj*sc - sj*cs,
	a[j]	= (cj*ss + sj*cc) * parity,
	a[k]	= cj*cs - sj*sc;
	q.r	= cj*cc + sj*ss;
    }

    q.v = a;

    return q;
}

template<class T>
inline bool
Euler<T>::legal(typename Euler<T>::Order order)
{
    return (order & ~Legal) ? false : true;
}

template<class T>
typename Euler<T>::Order
Euler<T>::order() const
{
    int foo = (_initialAxis == Z ? 0x2000 : (_initialAxis == Y ? 0x1000 : 0));

    if (_parityEven)	  foo |= 0x0100;
    if (_initialRepeated) foo |= 0x0010;
    if (_frameStatic)	  foo++;

    return (Order)foo;
}

template<class T>
inline void Euler<T>::setOrder(typename Euler<T>::Order p)
{
    set( p & 0x2000 ? Z : (p & 0x1000 ? Y : X),	// initial axis
	 !(p & 0x1),	    			// static?
	 !!(p & 0x100),				// permutation even?
	 !!(p & 0x10));				// initial repeats?
}

template<class T>
void Euler<T>::set(typename Euler<T>::Axis axis,
		   bool relative,
		   bool parityEven,
		   bool firstRepeats)
{
    _initialAxis	= axis;
    _frameStatic	= !relative;
    _parityEven		= parityEven;
    _initialRepeated	= firstRepeats;
}

template<class T>
const Euler<T>& Euler<T>::operator= (const Euler<T> &euler)
{
    x = euler.x;
    y = euler.y;
    z = euler.z;
    _initialAxis = euler._initialAxis;
    _frameStatic = euler._frameStatic;
    _parityEven	 = euler._parityEven;
    _initialRepeated = euler._initialRepeated;
    return *this;
}

template<class T>
const Euler<T>& Euler<T>::operator= (const Vec3<T> &v)
{
    x = v.x;
    y = v.y;
    z = v.z;
    return *this;
}

template<class T>
std::ostream& operator << (std::ostream &o, const Euler<T> &euler)
{
    char a[3] = { 'X', 'Y', 'Z' };

    const char* r = euler.frameStatic() ? "" : "r";
    int i,j,k;
    euler.angleOrder(i,j,k);

    if ( euler.initialRepeated() ) k = i;

    return o << "("
	     << euler.x << " "
	     << euler.y << " "
	     << euler.z << " "
	     << a[i] << a[j] << a[k] << r << ")";
}

template <class T>
float
Euler<T>::angleMod (T angle)
{
    angle = fmod(T (angle), T (2 * M_PI));

    if (angle < -M_PI)	angle += 2 * M_PI;
    if (angle > +M_PI)	angle -= 2 * M_PI;

    return angle;
}

template <class T>
void
Euler<T>::simpleXYZRotation (Vec3<T> &xyzRot, const Vec3<T> &targetXyzRot)
{
    Vec3<T> d  = xyzRot - targetXyzRot;
    xyzRot[0]  = targetXyzRot[0] + angleMod(d[0]);
    xyzRot[1]  = targetXyzRot[1] + angleMod(d[1]);
    xyzRot[2]  = targetXyzRot[2] + angleMod(d[2]);
}

template <class T>
void
Euler<T>::nearestRotation (Vec3<T> &xyzRot, const Vec3<T> &targetXyzRot,
			   Order order)
{
    int i,j,k;
    Euler<T> e (0,0,0, order);
    e.angleOrder(i,j,k);

    simpleXYZRotation(xyzRot, targetXyzRot);

    Vec3<T> otherXyzRot;
    otherXyzRot[i] = M_PI+xyzRot[i];
    otherXyzRot[j] = M_PI-xyzRot[j];
    otherXyzRot[k] = M_PI+xyzRot[k];

    simpleXYZRotation(otherXyzRot, targetXyzRot);
	    
    Vec3<T> d  = xyzRot - targetXyzRot;
    Vec3<T> od = otherXyzRot - targetXyzRot;
    T dMag     = d.dot(d);
    T odMag    = od.dot(od);

    if (odMag < dMag)
    {
	xyzRot = otherXyzRot;
    }
}

template <class T>
void
Euler<T>::makeNear (const Euler<T> &target)
{
    Vec3<T> xyzRot = toXYZVector();
    Vec3<T> targetXyz;
    if (order() != target.order())
    {
        Euler<T> targetSameOrder = Euler<T>(target, order());
        targetXyz = targetSameOrder.toXYZVector();
    }
    else
    {
        targetXyz = target.toXYZVector();
    }

    nearestRotation(xyzRot, targetXyz, order());

    setXYZVector(xyzRot);
}

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
#pragma warning(default:4244)
#endif

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT


#endif // INCLUDED_IMATHEULER_H
