///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IMATHSPHERE_H
#define INCLUDED_IMATHSPHERE_H

//-------------------------------------
//
//	A 3D sphere class template
//
//-------------------------------------

#include "ImathVec.h"
#include "ImathBox.h"
#include "ImathLine.h"

namespace Imath {

template <class T>
class Sphere3
{
  public:

    Vec3<T>	center;
    T           radius;

    //---------------
    //	Constructors
    //---------------

    Sphere3() : center(0,0,0), radius(0) {}
    Sphere3(const Vec3<T> &c, T r) : center(c), radius(r) {}

    //-------------------------------------------------------------------
    //	Utilities:
    //
    //	s.circumscribe(b)	sets center and radius of sphere s
    //				so that the s tightly encloses box b.
    //
    //	s.intersectT (l, t)	If sphere s and line l intersect, then
    //				intersectT() computes the smallest t,
    //				t >= 0, so that l(t) is a point on the
    //				sphere.  intersectT() then returns true.
    //
    //				If s and l do not intersect, intersectT()
    //				returns false.
    //
    //	s.intersect (l, i)	If sphere s and line l intersect, then
    //				intersect() calls s.intersectT(l,t) and
    //				computes i = l(t).
    //
    //				If s and l do not intersect, intersect()
    //				returns false.
    //
    //-------------------------------------------------------------------

    void circumscribe(const Box<Vec3<T> > &box);
    bool intersect(const Line3<T> &l, Vec3<T> &intersection) const;
    bool intersectT(const Line3<T> &l, T &t) const;
};


//--------------------
// Convenient typedefs
//--------------------

typedef Sphere3<float> Sphere3f;
typedef Sphere3<double> Sphere3d;


//---------------
// Implementation
//---------------

template <class T>
void Sphere3<T>::circumscribe(const Box<Vec3<T> > &box)
{
    center = T(0.5) * (box.min + box.max);
    radius = (box.max - center).length();
}


template <class T>
bool Sphere3<T>::intersectT(const Line3<T> &line, T &t) const
{
    bool doesIntersect = true;

    Vec3<T> v = line.pos - center;
    T B = T(2.0) * (line.dir ^ v);
    T C = (v ^ v) - (radius * radius);

    // compute discriminant
    // if negative, there is no intersection

    T discr = B*B - T(4.0)*C;

    if (discr < 0.0)
    {
	// line and Sphere3 do not intersect

	doesIntersect = false;
    }
    else
    {
	// t0: (-B - sqrt(B^2 - 4AC)) / 2A  (A = 1)

	T sqroot = Math<T>::sqrt(discr);
	t = (-B - sqroot) * T(0.5);

	if (t < 0.0)
	{
	    // no intersection, try t1: (-B + sqrt(B^2 - 4AC)) / 2A  (A = 1)

	    t = (-B + sqroot) * T(0.5);
	}

	if (t < 0.0)
	    doesIntersect = false;
    }

    return doesIntersect;
}


template <class T>
bool Sphere3<T>::intersect(const Line3<T> &line, Vec3<T> &intersection) const
{
    T t;

    if (intersectT (line, t))
    {
	intersection = line(t);
	return true;
    }
    else
    {
	return false;
    }
}


} //namespace Imath

#endif
