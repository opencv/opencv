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



#ifndef INCLUDED_IMATHLINEALGO_H
#define INCLUDED_IMATHLINEALGO_H

//------------------------------------------------------------------
//
//	This file contains algorithms applied to or in conjunction
//	with lines (Imath::Line). These algorithms may require
//	more headers to compile. The assumption made is that these
//	functions are called much less often than the basic line
//	functions or these functions require more support classes
//
//	Contains:
//
//	bool closestPoints(const Line<T>& line1,
//			   const Line<T>& line2,
//			   Vec3<T>& point1,
//			   Vec3<T>& point2)
//
//	bool intersect( const Line3<T> &line,
//			const Vec3<T> &v0,
//			const Vec3<T> &v1,
//			const Vec3<T> &v2,
//			Vec3<T> &pt,
//			Vec3<T> &barycentric,
//			bool &front)
//
//      V3f
//      closestVertex(const Vec3<T> &v0,
//                    const Vec3<T> &v1,
//                    const Vec3<T> &v2,
//                    const Line3<T> &l)
//
//      V3f
//      nearestPointOnTriangle(const Vec3<T> &v0,
//                             const Vec3<T> &v1,
//                             const Vec3<T> &v2,
//                             const Line3<T> &l)
//
//	V3f
//	rotatePoint(const Vec3<T> p, Line3<T> l, float angle)
//
//------------------------------------------------------------------

#include "ImathLine.h"
#include "ImathVecAlgo.h"

namespace Imath {


template <class T>
bool closestPoints(const Line3<T>& line1,
		   const Line3<T>& line2,
		   Vec3<T>& point1,
		   Vec3<T>& point2)
{
    //
    //	Compute the closest points on two lines. This was originally
    //	lifted from inventor. This function assumes that the line
    //	directions are normalized. The original math has been collapsed.
    //

    T A = line1.dir ^ line2.dir;

    if ( A == 1 ) return false;

    T denom = A * A - 1;

    T B = (line1.dir ^ line1.pos) - (line1.dir ^ line2.pos);
    T C = (line2.dir ^ line1.pos) - (line2.dir ^ line2.pos);

    point1 = line1(( B - A * C ) / denom);
    point2 = line2(( B * A - C ) / denom);

    return true;
}



template <class T>
bool intersect( const Line3<T> &line,
		const Vec3<T> &v0,
		const Vec3<T> &v1,
		const Vec3<T> &v2,
		Vec3<T> &pt,
		Vec3<T> &barycentric,
		bool &front)
{
    //    Intersect the line with a triangle.
    //    1. find plane of triangle
    //    2. find intersection point of ray and plane
    //    3. pick plane to project point and triangle into
    //    4. check each edge of triangle to see if point is inside it

    //
    // XXX TODO - this routine is way too long
    //		- the value of EPSILON is dubious
    //		- there should be versions of this
    //		  routine that do not calculate the
    //            barycentric coordinates or the
    //		  front flag

    const float EPSILON	= 1e-6;

    T	d, t, d01, d12, d20, vd0, vd1, vd2, ax, ay, az, sense;
    Vec3<T>	v01, v12, v20, c;
    int		axis0, axis1;

    // calculate plane for polygon
    v01 = v1 - v0;
    v12 = v2 - v1;

    // c is un-normalized normal
    c = v12.cross(v01);

    d = c.length();
    if(d < EPSILON)
	return false;	// cant hit a triangle with no area
    c = c * (1. / d);

    // calculate distance to plane along ray

    d = line.dir.dot(c);
    if (d < EPSILON && d > -EPSILON)
	return false;	// line is parallel to plane containing triangle

    t = (v0 - line.pos).dot(c) / d;

    if(t < 0)
	return false;

    // calculate intersection point
    pt = line.pos + t * line.dir;

    // is point inside triangle? Project to 2d to find out
    // use the plane that has the largest absolute value
    // component in the normal
    ax = c[0] < 0 ? -c[0] : c[0];
    ay = c[1] < 0 ? -c[1] : c[1];
    az = c[2] < 0 ? -c[2] : c[2];

    if(ax > ay && ax > az) 
    { 
        // project on x=0 plane

	axis0 = 1;
	axis1 = 2;
	sense = c[0] < 0 ? -1 : 1;
    }
    else if(ay > az) 
    {
	axis0 = 2;
	axis1 = 0;
	sense = c[1] < 0 ? -1 : 1;
    }
    else 
    {
	axis0 = 0;
	axis1 = 1;
	sense = c[2] < 0 ? -1 : 1;
    }

    // distance from v0-v1 must be less than distance from v2 to v0-v1
    d01 = sense * ((pt[axis0] - v0[axis0]) * v01[axis1]
	         - (pt[axis1] - v0[axis1]) * v01[axis0]);

    if(d01 < 0) return false;

    vd2 = sense * ((v2[axis0] - v0[axis0]) * v01[axis1]
	         - (v2[axis1] - v0[axis1]) * v01[axis0]);

    if(d01 > vd2) return false;

    // distance from v1-v2 must be less than distance from v1 to v2-v0
    d12 = sense * ((pt[axis0] - v1[axis0]) * v12[axis1]
	         - (pt[axis1] - v1[axis1]) * v12[axis0]);

    if(d12 < 0) return false;

    vd0 = sense * ((v0[axis0] - v1[axis0]) * v12[axis1]
	         - (v0[axis1] - v1[axis1]) * v12[axis0]);

    if(d12 > vd0) return false;

    // calculate v20, and do check on final side of triangle
    v20 = v0 - v2;
    d20 = sense * ((pt[axis0] - v2[axis0]) * v20[axis1]
                 - (pt[axis1] - v2[axis1]) * v20[axis0]);

    if(d20 < 0) return false;

    vd1 = sense * ((v1[axis0] - v2[axis0]) * v20[axis1]
	         - (v1[axis1] - v2[axis1]) * v20[axis0]);

    if(d20 > vd1) return false;

    // vd0, vd1, and vd2 will always be non-zero for a triangle
    // that has non-zero area (we return before this for
    // zero area triangles)
    barycentric = Vec3<T>(d12 / vd0, d20 / vd1, d01 / vd2);
    front = line.dir.dot(c) < 0;

    return true;
}

template <class T>
Vec3<T>
closestVertex(const Vec3<T> &v0,
              const Vec3<T> &v1,
              const Vec3<T> &v2,
              const Line3<T> &l)
{
    Vec3<T> nearest = v0;
    T neardot       = (v0 - l.closestPointTo(v0)).length2();
    
    T tmp           = (v1 - l.closestPointTo(v1)).length2();

    if (tmp < neardot)
    {
        neardot = tmp;
        nearest = v1;
    }

    tmp = (v2 - l.closestPointTo(v2)).length2();
    if (tmp < neardot)
    {
        neardot = tmp;
        nearest = v2;
    }

    return nearest;
}

template <class T>
Vec3<T>
nearestPointOnTriangle(const Vec3<T> &v0,
                       const Vec3<T> &v1,
                       const Vec3<T> &v2,
                       const Line3<T> &l)
{
    Vec3<T> pt, barycentric;
    bool front;

    if (intersect (l, v0, v1, v2, pt, barycentric, front))
	return pt;

    //
    // The line did not intersect the triangle, so to be picky, you should
    // find the closest edge that it passed over/under, but chances are that
    // 1) another triangle will be closer
    // 2) the app does not need this much precision for a ray that does not
    //    intersect the triangle
    // 3) the expense of the calculation is not worth it since this is the
    //    common case
    //
    // XXX TODO  This is bogus -- nearestPointOnTriangle() should do
    //		 what its name implies; it should return a point
    //           on an edge if some edge is closer to the line than
    //		 any vertex.  If the application does not want the
    //		 extra calculations, it should be possible to specify
    //		 that; it is not up to this nearestPointOnTriangle()
    //		 to make the decision.

    return closestVertex(v0, v1, v2, l);
}

template <class T>
Vec3<T>
rotatePoint(const Vec3<T> p, Line3<T> l, T angle)
{
    //
    // Rotate the point p around the line l by the given angle.
    //

    //
    // Form a coordinate frame with <x,y,a>. The rotation is the in xy
    // plane.
    //

    Vec3<T> q = l.closestPointTo(p);
    Vec3<T> x = p - q;
    T radius = x.length();

    x.normalize();
    Vec3<T> y = (x % l.dir).normalize();

    T cosangle = Math<T>::cos(angle);
    T sinangle = Math<T>::sin(angle);

    Vec3<T> r = q + x * radius * cosangle + y * radius * sinangle; 

    return r;
}


} // namespace Imath

#endif
