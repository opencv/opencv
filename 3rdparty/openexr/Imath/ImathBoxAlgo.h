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



#ifndef INCLUDED_IMATHBOXALGO_H
#define INCLUDED_IMATHBOXALGO_H


//---------------------------------------------------------------------------
//
//	This file contains algorithms applied to or in conjunction
//	with bounding boxes (Imath::Box). These algorithms require
//	more headers to compile. The assumption made is that these
//	functions are called much less often than the basic box
//	functions or these functions require more support classes.
//
//	Contains:
//
//	T clip<T>(const T& in, const Box<T>& box)
//
//	Vec3<T> closestPointOnBox(const Vec3<T>&, const Box<Vec3<T>>& )
//
//	Vec3<T> closestPointInBox(const Vec3<T>&, const Box<Vec3<T>>& )
//
//	Box< Vec3<S> > transform(const Box<Vec3<S>>&, const Matrix44<T>&)
//	Box< Vec3<S> > affineTransform(const Box<Vec3<S>>&, const Matrix44<T>&)
//
//	void transform(const Box<Vec3<S>>&, const Matrix44<T>&, Box<V3ec3<S>>&)
//	void affineTransform(const Box<Vec3<S>>&,
//                           const Matrix44<T>&,
//                           Box<V3ec3<S>>&)
//
//	bool findEntryAndExitPoints(const Line<T> &line,
//				    const Box< Vec3<T> > &box,
//				    Vec3<T> &enterPoint,
//				    Vec3<T> &exitPoint)
//
//	bool intersects(const Box<Vec3<T>> &box, 
//			const Line3<T> &ray, 
//			Vec3<T> intersectionPoint)
//
//	bool intersects(const Box<Vec3<T>> &box, const Line3<T> &ray)
//
//---------------------------------------------------------------------------

#include "ImathBox.h"
#include "ImathMatrix.h"
#include "ImathLineAlgo.h"
#include "ImathPlane.h"
#include "ImathNamespace.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER


template <class T>
inline T
clip (const T &p, const Box<T> &box)
{
    //
    // Clip the coordinates of a point, p, against a box.
    // The result, q, is the closest point to p that is inside the box.
    //

    T q;

    for (int i = 0; i < int (box.min.dimensions()); i++)
    {
	if (p[i] < box.min[i])
	    q[i] = box.min[i];
	else if (p[i] > box.max[i])
	    q[i] = box.max[i];
	else
	    q[i] = p[i];
    }

    return q;
}


template <class T>
inline T
closestPointInBox (const T &p, const Box<T> &box)
{
    return clip (p, box);
}


template <class T>
Vec3<T>
closestPointOnBox (const Vec3<T> &p, const Box< Vec3<T> > &box)
{
    //
    // Find the point, q, on the surface of
    // the box, that is closest to point p.
    //
    // If the box is empty, return p.
    //

    if (box.isEmpty())
	return p;

    Vec3<T> q = closestPointInBox (p, box);

    if (q == p)
    {
	Vec3<T> d1 = p - box.min;
	Vec3<T> d2 = box.max - p;

	Vec3<T> d ((d1.x < d2.x)? d1.x: d2.x,
		   (d1.y < d2.y)? d1.y: d2.y,
		   (d1.z < d2.z)? d1.z: d2.z);

	if (d.x < d.y && d.x < d.z)
	{
	    q.x = (d1.x < d2.x)? box.min.x: box.max.x;
	}
	else if (d.y < d.z)
	{
	    q.y = (d1.y < d2.y)? box.min.y: box.max.y;
	}
	else
	{
	    q.z = (d1.z < d2.z)? box.min.z: box.max.z;
	}
    }

    return q;
}


template <class S, class T>
Box< Vec3<S> >
transform (const Box< Vec3<S> > &box, const Matrix44<T> &m)
{
    //
    // Transform a 3D box by a matrix, and compute a new box that
    // tightly encloses the transformed box.
    //
    // If m is an affine transform, then we use James Arvo's fast
    // method as described in "Graphics Gems", Academic Press, 1990,
    // pp. 548-550.
    //

    //
    // A transformed empty box is still empty, and a transformed infinite box
    // is still infinite
    //

    if (box.isEmpty() || box.isInfinite())
	return box;

    //
    // If the last column of m is (0 0 0 1) then m is an affine
    // transform, and we use the fast Graphics Gems trick.
    //

    if (m[0][3] == 0 && m[1][3] == 0 && m[2][3] == 0 && m[3][3] == 1)
    {
	Box< Vec3<S> > newBox;

	for (int i = 0; i < 3; i++) 
        {
	    newBox.min[i] = newBox.max[i] = (S) m[3][i];

	    for (int j = 0; j < 3; j++) 
            {
		S a, b;

		a = (S) m[j][i] * box.min[j];
		b = (S) m[j][i] * box.max[j];

		if (a < b) 
                {
		    newBox.min[i] += a;
		    newBox.max[i] += b;
		}
		else 
                {
		    newBox.min[i] += b;
		    newBox.max[i] += a;
		}
	    }
	}

	return newBox;
    }

    //
    // M is a projection matrix.  Do things the naive way:
    // Transform the eight corners of the box, and find an
    // axis-parallel box that encloses the transformed corners.
    //

    Vec3<S> points[8];

    points[0][0] = points[1][0] = points[2][0] = points[3][0] = box.min[0];
    points[4][0] = points[5][0] = points[6][0] = points[7][0] = box.max[0];

    points[0][1] = points[1][1] = points[4][1] = points[5][1] = box.min[1];
    points[2][1] = points[3][1] = points[6][1] = points[7][1] = box.max[1];

    points[0][2] = points[2][2] = points[4][2] = points[6][2] = box.min[2];
    points[1][2] = points[3][2] = points[5][2] = points[7][2] = box.max[2];

    Box< Vec3<S> > newBox;

    for (int i = 0; i < 8; i++) 
	newBox.extendBy (points[i] * m);

    return newBox;
}

template <class S, class T>
void
transform (const Box< Vec3<S> > &box,
           const Matrix44<T>    &m,
           Box< Vec3<S> >       &result)
{
    //
    // Transform a 3D box by a matrix, and compute a new box that
    // tightly encloses the transformed box.
    //
    // If m is an affine transform, then we use James Arvo's fast
    // method as described in "Graphics Gems", Academic Press, 1990,
    // pp. 548-550.
    //

    //
    // A transformed empty box is still empty, and a transformed infinite
    // box is still infinite
    //

    if (box.isEmpty() || box.isInfinite())
    {
	return;
    }

    //
    // If the last column of m is (0 0 0 1) then m is an affine
    // transform, and we use the fast Graphics Gems trick.
    //

    if (m[0][3] == 0 && m[1][3] == 0 && m[2][3] == 0 && m[3][3] == 1)
    {
	for (int i = 0; i < 3; i++) 
        {
	    result.min[i] = result.max[i] = (S) m[3][i];

	    for (int j = 0; j < 3; j++) 
            {
		S a, b;

		a = (S) m[j][i] * box.min[j];
		b = (S) m[j][i] * box.max[j];

		if (a < b) 
                {
		    result.min[i] += a;
		    result.max[i] += b;
		}
		else 
                {
		    result.min[i] += b;
		    result.max[i] += a;
		}
	    }
	}

	return;
    }

    //
    // M is a projection matrix.  Do things the naive way:
    // Transform the eight corners of the box, and find an
    // axis-parallel box that encloses the transformed corners.
    //

    Vec3<S> points[8];

    points[0][0] = points[1][0] = points[2][0] = points[3][0] = box.min[0];
    points[4][0] = points[5][0] = points[6][0] = points[7][0] = box.max[0];

    points[0][1] = points[1][1] = points[4][1] = points[5][1] = box.min[1];
    points[2][1] = points[3][1] = points[6][1] = points[7][1] = box.max[1];

    points[0][2] = points[2][2] = points[4][2] = points[6][2] = box.min[2];
    points[1][2] = points[3][2] = points[5][2] = points[7][2] = box.max[2];

    for (int i = 0; i < 8; i++) 
	result.extendBy (points[i] * m);
}


template <class S, class T>
Box< Vec3<S> >
affineTransform (const Box< Vec3<S> > &box, const Matrix44<T> &m)
{
    //
    // Transform a 3D box by a matrix whose rightmost column
    // is (0 0 0 1), and compute a new box that tightly encloses
    // the transformed box.
    //
    // As in the transform() function, above, we use James Arvo's
    // fast method.
    //

    if (box.isEmpty() || box.isInfinite())
    {
	//
	// A transformed empty or infinite box is still empty or infinite
	//

	return box;
    }

    Box< Vec3<S> > newBox;

    for (int i = 0; i < 3; i++) 
    {
	newBox.min[i] = newBox.max[i] = (S) m[3][i];

	for (int j = 0; j < 3; j++) 
	{
	    S a, b;

	    a = (S) m[j][i] * box.min[j];
	    b = (S) m[j][i] * box.max[j];

	    if (a < b) 
	    {
		newBox.min[i] += a;
		newBox.max[i] += b;
	    }
	    else 
	    {
		newBox.min[i] += b;
		newBox.max[i] += a;
	    }
	}
    }

    return newBox;
}

template <class S, class T>
void
affineTransform (const Box< Vec3<S> > &box,
                 const Matrix44<T>    &m,
                 Box<Vec3<S> >        &result)
{
    //
    // Transform a 3D box by a matrix whose rightmost column
    // is (0 0 0 1), and compute a new box that tightly encloses
    // the transformed box.
    //
    // As in the transform() function, above, we use James Arvo's
    // fast method.
    //

    if (box.isEmpty())
    {
	//
	// A transformed empty box is still empty
	//
        result.makeEmpty();
	return;
    }

    if (box.isInfinite())
    {
	//
	// A transformed infinite box is still infinite
	//
        result.makeInfinite();
	return;
    }

    for (int i = 0; i < 3; i++) 
    {
	result.min[i] = result.max[i] = (S) m[3][i];

	for (int j = 0; j < 3; j++) 
	{
	    S a, b;

	    a = (S) m[j][i] * box.min[j];
	    b = (S) m[j][i] * box.max[j];

	    if (a < b) 
	    {
		result.min[i] += a;
		result.max[i] += b;
	    }
	    else 
	    {
		result.min[i] += b;
		result.max[i] += a;
	    }
	}
    }
}


template <class T>
bool
findEntryAndExitPoints (const Line3<T> &r,
			const Box<Vec3<T> > &b,
			Vec3<T> &entry,
			Vec3<T> &exit)
{
    //
    // Compute the points where a ray, r, enters and exits a box, b:
    //
    // findEntryAndExitPoints() returns
    //
    //     - true if the ray starts inside the box or if the
    //       ray starts outside and intersects the box
    //
    //	   - false otherwise (that is, if the ray does not
    //       intersect the box)
    //
    // The entry and exit points are
    //
    //     - points on two of the faces of the box when
    //       findEntryAndExitPoints() returns true
    //       (The entry end exit points may be on either
    //       side of the ray's origin)
    //
    //     - undefined when findEntryAndExitPoints()
    //       returns false
    //

    if (b.isEmpty())
    {
	//
	// No ray intersects an empty box
	//

	return false;
    }

    //
    // The following description assumes that the ray's origin is outside
    // the box, but the code below works even if the origin is inside the
    // box:
    //
    // Between one and three "frontfacing" sides of the box are oriented
    // towards the ray's origin, and between one and three "backfacing"
    // sides are oriented away from the ray's origin.
    // We intersect the ray with the planes that contain the sides of the
    // box, and compare the distances between the ray's origin and the
    // ray-plane intersections.  The ray intersects the box if the most
    // distant frontfacing intersection is nearer than the nearest
    // backfacing intersection.  If the ray does intersect the box, then
    // the most distant frontfacing ray-plane intersection is the entry
    // point and the nearest backfacing ray-plane intersection is the
    // exit point.
    //

    const T TMAX = limits<T>::max();

    T tFrontMax = -TMAX;
    T tBackMin = TMAX;

    //
    // Minimum and maximum X sides.
    //

    if (r.dir.x >= 0)
    {
	T d1 = b.max.x - r.pos.x;
	T d2 = b.min.x - r.pos.x;

	if (r.dir.x > 1 ||
	    (abs (d1) < TMAX * r.dir.x &&
	     abs (d2) < TMAX * r.dir.x))
	{
	    T t1 = d1 / r.dir.x;
	    T t2 = d2 / r.dir.x;

	    if (tBackMin > t1)
	    {
		tBackMin = t1;

		exit.x = b.max.x; 
		exit.y = clamp (r.pos.y + t1 * r.dir.y, b.min.y, b.max.y);
		exit.z = clamp (r.pos.z + t1 * r.dir.z, b.min.z, b.max.z);
	    }

	    if (tFrontMax < t2)
	    {
		tFrontMax = t2;

		entry.x = b.min.x; 
		entry.y = clamp (r.pos.y + t2 * r.dir.y, b.min.y, b.max.y);
		entry.z = clamp (r.pos.z + t2 * r.dir.z, b.min.z, b.max.z);
	    }
	}
	else if (r.pos.x < b.min.x || r.pos.x > b.max.x)
	{
	    return false;
	}
    }
    else // r.dir.x < 0
    {
	T d1 = b.min.x - r.pos.x;
	T d2 = b.max.x - r.pos.x;

	if (r.dir.x < -1 ||
	    (abs (d1) < -TMAX * r.dir.x &&
	     abs (d2) < -TMAX * r.dir.x))
	{
	    T t1 = d1 / r.dir.x;
	    T t2 = d2 / r.dir.x;

	    if (tBackMin > t1)
	    {
		tBackMin = t1;

		exit.x = b.min.x; 
		exit.y = clamp (r.pos.y + t1 * r.dir.y, b.min.y, b.max.y);
		exit.z = clamp (r.pos.z + t1 * r.dir.z, b.min.z, b.max.z);
	    }

	    if (tFrontMax < t2)
	    {
		tFrontMax = t2;

		entry.x = b.max.x; 
		entry.y = clamp (r.pos.y + t2 * r.dir.y, b.min.y, b.max.y);
		entry.z = clamp (r.pos.z + t2 * r.dir.z, b.min.z, b.max.z);
	    }
	}
	else if (r.pos.x < b.min.x || r.pos.x > b.max.x)
	{
	    return false;
	}
    }

    //
    // Minimum and maximum Y sides.
    //

    if (r.dir.y >= 0)
    {
	T d1 = b.max.y - r.pos.y;
	T d2 = b.min.y - r.pos.y;

	if (r.dir.y > 1 ||
	    (abs (d1) < TMAX * r.dir.y &&
	     abs (d2) < TMAX * r.dir.y))
	{
	    T t1 = d1 / r.dir.y;
	    T t2 = d2 / r.dir.y;

	    if (tBackMin > t1)
	    {
		tBackMin = t1;

		exit.x = clamp (r.pos.x + t1 * r.dir.x, b.min.x, b.max.x);
		exit.y = b.max.y; 
		exit.z = clamp (r.pos.z + t1 * r.dir.z, b.min.z, b.max.z);
	    }

	    if (tFrontMax < t2)
	    {
		tFrontMax = t2;

		entry.x = clamp (r.pos.x + t2 * r.dir.x, b.min.x, b.max.x);
		entry.y = b.min.y; 
		entry.z = clamp (r.pos.z + t2 * r.dir.z, b.min.z, b.max.z);
	    }
	}
	else if (r.pos.y < b.min.y || r.pos.y > b.max.y)
	{
	    return false;
	}
    }
    else // r.dir.y < 0
    {
	T d1 = b.min.y - r.pos.y;
	T d2 = b.max.y - r.pos.y;

	if (r.dir.y < -1 ||
	    (abs (d1) < -TMAX * r.dir.y &&
	     abs (d2) < -TMAX * r.dir.y))
	{
	    T t1 = d1 / r.dir.y;
	    T t2 = d2 / r.dir.y;

	    if (tBackMin > t1)
	    {
		tBackMin = t1;

		exit.x = clamp (r.pos.x + t1 * r.dir.x, b.min.x, b.max.x);
		exit.y = b.min.y; 
		exit.z = clamp (r.pos.z + t1 * r.dir.z, b.min.z, b.max.z);
	    }

	    if (tFrontMax < t2)
	    {
		tFrontMax = t2;

		entry.x = clamp (r.pos.x + t2 * r.dir.x, b.min.x, b.max.x);
		entry.y = b.max.y; 
		entry.z = clamp (r.pos.z + t2 * r.dir.z, b.min.z, b.max.z);
	    }
	}
	else if (r.pos.y < b.min.y || r.pos.y > b.max.y)
	{
	    return false;
	}
    }

    //
    // Minimum and maximum Z sides.
    //

    if (r.dir.z >= 0)
    {
	T d1 = b.max.z - r.pos.z;
	T d2 = b.min.z - r.pos.z;

	if (r.dir.z > 1 ||
	    (abs (d1) < TMAX * r.dir.z &&
	     abs (d2) < TMAX * r.dir.z))
	{
	    T t1 = d1 / r.dir.z;
	    T t2 = d2 / r.dir.z;

	    if (tBackMin > t1)
	    {
		tBackMin = t1;

		exit.x = clamp (r.pos.x + t1 * r.dir.x, b.min.x, b.max.x);
		exit.y = clamp (r.pos.y + t1 * r.dir.y, b.min.y, b.max.y);
		exit.z = b.max.z; 
	    }

	    if (tFrontMax < t2)
	    {
		tFrontMax = t2;

		entry.x = clamp (r.pos.x + t2 * r.dir.x, b.min.x, b.max.x);
		entry.y = clamp (r.pos.y + t2 * r.dir.y, b.min.y, b.max.y);
		entry.z = b.min.z; 
	    }
	}
	else if (r.pos.z < b.min.z || r.pos.z > b.max.z)
	{
	    return false;
	}
    }
    else // r.dir.z < 0
    {
	T d1 = b.min.z - r.pos.z;
	T d2 = b.max.z - r.pos.z;

	if (r.dir.z < -1 ||
	    (abs (d1) < -TMAX * r.dir.z &&
	     abs (d2) < -TMAX * r.dir.z))
	{
	    T t1 = d1 / r.dir.z;
	    T t2 = d2 / r.dir.z;

	    if (tBackMin > t1)
	    {
		tBackMin = t1;

		exit.x = clamp (r.pos.x + t1 * r.dir.x, b.min.x, b.max.x);
		exit.y = clamp (r.pos.y + t1 * r.dir.y, b.min.y, b.max.y);
		exit.z = b.min.z; 
	    }

	    if (tFrontMax < t2)
	    {
		tFrontMax = t2;

		entry.x = clamp (r.pos.x + t2 * r.dir.x, b.min.x, b.max.x);
		entry.y = clamp (r.pos.y + t2 * r.dir.y, b.min.y, b.max.y);
		entry.z = b.max.z; 
	    }
	}
	else if (r.pos.z < b.min.z || r.pos.z > b.max.z)
	{
	    return false;
	}
    }

    return tFrontMax <= tBackMin;
}


template<class T>
bool
intersects (const Box< Vec3<T> > &b, const Line3<T> &r, Vec3<T> &ip)
{
    //
    // Intersect a ray, r, with a box, b, and compute the intersection
    // point, ip:
    //
    // intersect() returns
    //
    //     - true if the ray starts inside the box or if the
    //       ray starts outside and intersects the box
    //
    //     - false if the ray starts outside the box and intersects it,
    //       but the intersection is behind the ray's origin.
    //
    //     - false if the ray starts outside and does not intersect it
    //
    // The intersection point is
    //
    //     - the ray's origin if the ray starts inside the box
    //
    //     - a point on one of the faces of the box if the ray
    //       starts outside the box
    //
    //     - undefined when intersect() returns false
    //

    if (b.isEmpty())
    {
	//
	// No ray intersects an empty box
	//

	return false;
    }

    if (b.intersects (r.pos))
    {
	//
	// The ray starts inside the box
	//

	ip = r.pos;
	return true;
    }

    //
    // The ray starts outside the box.  Between one and three "frontfacing"
    // sides of the box are oriented towards the ray, and between one and
    // three "backfacing" sides are oriented away from the ray.
    // We intersect the ray with the planes that contain the sides of the
    // box, and compare the distances between ray's origin and the ray-plane
    // intersections.
    // The ray intersects the box if the most distant frontfacing intersection
    // is nearer than the nearest backfacing intersection.  If the ray does
    // intersect the box, then the most distant frontfacing ray-plane
    // intersection is the ray-box intersection.
    //

    const T TMAX = limits<T>::max();

    T tFrontMax = -1;
    T tBackMin = TMAX;

    //
    // Minimum and maximum X sides.
    //

    if (r.dir.x > 0)
    {
	if (r.pos.x > b.max.x)
	    return false;

	T d = b.max.x - r.pos.x;

	if (r.dir.x > 1 || d < TMAX * r.dir.x)
	{
	    T t = d / r.dir.x;

	    if (tBackMin > t)
		tBackMin = t;
	}

	if (r.pos.x <= b.min.x)
	{
	    T d = b.min.x - r.pos.x;
	    T t = (r.dir.x > 1 || d < TMAX * r.dir.x)? d / r.dir.x: TMAX;

	    if (tFrontMax < t)
	    {
		tFrontMax = t;

		ip.x = b.min.x; 
		ip.y = clamp (r.pos.y + t * r.dir.y, b.min.y, b.max.y);
		ip.z = clamp (r.pos.z + t * r.dir.z, b.min.z, b.max.z);
	    }
	}
    }
    else if (r.dir.x < 0)
    {
	if (r.pos.x < b.min.x)
	    return false;

	T d = b.min.x - r.pos.x;

	if (r.dir.x < -1 || d > TMAX * r.dir.x)
	{
	    T t = d / r.dir.x;

	    if (tBackMin > t)
		tBackMin = t;
	}

	if (r.pos.x >= b.max.x)
	{
	    T d = b.max.x - r.pos.x;
	    T t = (r.dir.x < -1 || d > TMAX * r.dir.x)? d / r.dir.x: TMAX;

	    if (tFrontMax < t)
	    {
		tFrontMax = t;

		ip.x = b.max.x; 
		ip.y = clamp (r.pos.y + t * r.dir.y, b.min.y, b.max.y);
		ip.z = clamp (r.pos.z + t * r.dir.z, b.min.z, b.max.z);
	    }
	}
    }
    else // r.dir.x == 0
    {
	if (r.pos.x < b.min.x || r.pos.x > b.max.x)
	    return false;
    }

    //
    // Minimum and maximum Y sides.
    //

    if (r.dir.y > 0)
    {
	if (r.pos.y > b.max.y)
	    return false;

	T d = b.max.y - r.pos.y;

	if (r.dir.y > 1 || d < TMAX * r.dir.y)
	{
	    T t = d / r.dir.y;

	    if (tBackMin > t)
		tBackMin = t;
	}

	if (r.pos.y <= b.min.y)
	{
	    T d = b.min.y - r.pos.y;
	    T t = (r.dir.y > 1 || d < TMAX * r.dir.y)? d / r.dir.y: TMAX;

	    if (tFrontMax < t)
	    {
		tFrontMax = t;

		ip.x = clamp (r.pos.x + t * r.dir.x, b.min.x, b.max.x);
		ip.y = b.min.y; 
		ip.z = clamp (r.pos.z + t * r.dir.z, b.min.z, b.max.z);
	    }
	}
    }
    else if (r.dir.y < 0)
    {
	if (r.pos.y < b.min.y)
	    return false;

	T d = b.min.y - r.pos.y;

	if (r.dir.y < -1 || d > TMAX * r.dir.y)
	{
	    T t = d / r.dir.y;

	    if (tBackMin > t)
		tBackMin = t;
	}

	if (r.pos.y >= b.max.y)
	{
	    T d = b.max.y - r.pos.y;
	    T t = (r.dir.y < -1 || d > TMAX * r.dir.y)? d / r.dir.y: TMAX;
	    
	    if (tFrontMax < t)
	    {
		tFrontMax = t;

		ip.x = clamp (r.pos.x + t * r.dir.x, b.min.x, b.max.x);
		ip.y = b.max.y; 
		ip.z = clamp (r.pos.z + t * r.dir.z, b.min.z, b.max.z);
	    }
	}
    }
    else // r.dir.y == 0
    {
	if (r.pos.y < b.min.y || r.pos.y > b.max.y)
	    return false;
    }

    //
    // Minimum and maximum Z sides.
    //

    if (r.dir.z > 0)
    {
	if (r.pos.z > b.max.z)
	    return false;

	T d = b.max.z - r.pos.z;

	if (r.dir.z > 1 || d < TMAX * r.dir.z)
	{
	    T t = d / r.dir.z;

	    if (tBackMin > t)
		tBackMin = t;
	}

	if (r.pos.z <= b.min.z)
	{
	    T d = b.min.z - r.pos.z;
	    T t = (r.dir.z > 1 || d < TMAX * r.dir.z)? d / r.dir.z: TMAX;
	    
	    if (tFrontMax < t)
	    {
		tFrontMax = t;

		ip.x = clamp (r.pos.x + t * r.dir.x, b.min.x, b.max.x);
		ip.y = clamp (r.pos.y + t * r.dir.y, b.min.y, b.max.y);
		ip.z = b.min.z; 
	    }
	}
    }
    else if (r.dir.z < 0)
    {
	if (r.pos.z < b.min.z)
	    return false;

	T d = b.min.z - r.pos.z;

	if (r.dir.z < -1 || d > TMAX * r.dir.z)
	{
	    T t = d / r.dir.z;

	    if (tBackMin > t)
		tBackMin = t;
	}

	if (r.pos.z >= b.max.z)
	{
	    T d = b.max.z - r.pos.z;
	    T t = (r.dir.z < -1 || d > TMAX * r.dir.z)? d / r.dir.z: TMAX;
	    
	    if (tFrontMax < t)
	    {
		tFrontMax = t;

		ip.x = clamp (r.pos.x + t * r.dir.x, b.min.x, b.max.x);
		ip.y = clamp (r.pos.y + t * r.dir.y, b.min.y, b.max.y);
		ip.z = b.max.z; 
	    }
	}
    }
    else // r.dir.z == 0
    {
	if (r.pos.z < b.min.z || r.pos.z > b.max.z)
	    return false;
    }

    return tFrontMax <= tBackMin;
}


template<class T>
bool
intersects (const Box< Vec3<T> > &box, const Line3<T> &ray)
{
    Vec3<T> ignored;
    return intersects (box, ray, ignored);
}


IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHBOXALGO_H
