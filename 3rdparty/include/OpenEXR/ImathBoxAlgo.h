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
//	void transform(Box<Vec3<T>>&, const Matrix44<T>&)
//
//	bool findEntryAndExitPoints(const Line<T> &line,
//				    const Box< Vec3<T> > &box,
//				    Vec3<T> &enterPoint,
//				    Vec3<T> &exitPoint)
//
//	bool intersects(const Box<Vec3<T>> &box, 
//			const Line3<T> &line, 
//			Vec3<T> result)
//
//	bool intersects(const Box<Vec3<T>> &box, const Line3<T> &line)
//
//---------------------------------------------------------------------------

#include "ImathBox.h"
#include "ImathMatrix.h"
#include "ImathLineAlgo.h"
#include "ImathPlane.h"

namespace Imath {


template <class T>
inline T clip(const T& in, const Box<T>& box)
{
    //
    //	Clip a point so that it lies inside the given bbox
    //

    T out;

    for (int i=0; i<(int)box.min.dimensions(); i++)
    {
	if (in[i] < box.min[i]) out[i] = box.min[i];
	else if (in[i] > box.max[i]) out[i] = box.max[i];
	else out[i] = in[i];
    }

    return out;
}


//
// Return p if p is inside the box.
//
 
template <class T>
Vec3<T> 
closestPointInBox(const Vec3<T>& p, const Box< Vec3<T> >& box )
{
    Imath::V3f b;

    if (p.x < box.min.x)
	b.x = box.min.x;
    else if (p.x > box.max.x)
	b.x = box.max.x;
    else
	b.x = p.x;

    if (p.y < box.min.y)
	b.y = box.min.y;
    else if (p.y > box.max.y)
	b.y = box.max.y;
    else
	b.y = p.y;

    if (p.z < box.min.z)
	b.z = box.min.z;
    else if (p.z > box.max.z)
	b.z = box.max.z;
    else
	b.z = p.z;

    return b;
}

template <class T>
Vec3<T> closestPointOnBox(const Vec3<T>& pt, const Box< Vec3<T> >& box )
{
    //
    //	This sucker is specialized to work with a Vec3f and a box
    //	made of Vec3fs. 
    //

    Vec3<T> result;
    
    // trivial cases first
    if (box.isEmpty())
	return pt;
    else if (pt == box.center()) 
    {
	// middle of z side
	result[0] = (box.max[0] + box.min[0])/2.0;
	result[1] = (box.max[1] + box.min[1])/2.0;
	result[2] = box.max[2];
    }
    else 
    {
	// Find the closest point on a unit box (from -1 to 1),
	// then scale up.

	// Find the vector from center to the point, then scale
	// to a unit box.
	Vec3<T> vec = pt - box.center();
	T sizeX = box.max[0]-box.min[0];
	T sizeY = box.max[1]-box.min[1];
	T sizeZ = box.max[2]-box.min[2];

	T halfX = sizeX/2.0;
	T halfY = sizeY/2.0;
	T halfZ = sizeZ/2.0;
	if (halfX > 0.0)
	    vec[0] /= halfX;
	if (halfY > 0.0)
	    vec[1] /= halfY;
	if (halfZ > 0.0)
	    vec[2] /= halfZ;

	// Side to snap side that has greatest magnitude in the vector.
	Vec3<T> mag;
	mag[0] = fabs(vec[0]);
	mag[1] = fabs(vec[1]);
	mag[2] = fabs(vec[2]);

	result = mag;

	// Check if beyond corners
	if (result[0] > 1.0)
	    result[0] = 1.0;
	if (result[1] > 1.0)
	    result[1] = 1.0;
	if (result[2] > 1.0)
	    result[2] = 1.0;

	// snap to appropriate side	    
	if ((mag[0] > mag[1]) && (mag[0] >  mag[2])) 
        {
	    result[0] = 1.0;
	}
	else if ((mag[1] > mag[0]) && (mag[1] >  mag[2])) 
        {
	    result[1] = 1.0;
	}
	else if ((mag[2] > mag[0]) && (mag[2] >  mag[1])) 
        {
	    result[2] = 1.0;
	}
	else if ((mag[0] == mag[1]) && (mag[0] == mag[2])) 
        {
	    // corner
	    result = Vec3<T>(1,1,1);
	}
	else if (mag[0] == mag[1]) 
        {
	    // edge parallel with z
	    result[0] = 1.0;
	    result[1] = 1.0;
	}
	else if (mag[0] == mag[2]) 
        {
	    // edge parallel with y
	    result[0] = 1.0;
	    result[2] = 1.0;
	}
	else if (mag[1] == mag[2]) 
        {
	    // edge parallel with x
	    result[1] = 1.0;
	    result[2] = 1.0;
	}

	// Now make everything point the right way
	for (int i=0; i < 3; i++)
        {
	    if (vec[i] < 0.0)
		result[i] = -result[i];
        }

	// scale back up and move to center
	result[0] *= halfX;
	result[1] *= halfY;
	result[2] *= halfZ;

	result += box.center();
    }
    return result;
}

template <class S, class T>
Box< Vec3<S> >
transform(const Box< Vec3<S> >& box, const Matrix44<T>& m)
{
    // Transforms Box3f by matrix, enlarging Box3f to contain result.
    // Clever method courtesy of Graphics Gems, pp. 548-550
    //
    // This works for projection matrices as well as simple affine
    // transformations.  Coordinates of the box are rehomogenized if there
    // is a projection matrix

    // a transformed empty box is still empty
    if (box.isEmpty())
	return box;

    // If the last column is close enuf to ( 0 0 0 1 ) then we use the
    // fast, affine version.  The tricky affine method could maybe be
    // extended to deal with the projection case as well, but its not
    // worth it right now.

    if (m[0][3] * m[0][3] + m[1][3] * m[1][3] + m[2][3] * m[2][3]
	+ (1.0 - m[3][3]) * (1.0 - m[3][3]) < 0.00001) 
    {
	// Affine version, use the Graphics Gems hack
	int		i, j;
	Box< Vec3<S> >  newBox;

	for (i = 0; i < 3; i++) 
        {
	    newBox.min[i] = newBox.max[i] = (S) m[3][i];

	    for (j = 0; j < 3; j++) 
            {
		float a, b;

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

    // This is a projection matrix.  Do things the naive way.
    Vec3<S> points[8];

    /* Set up the eight points at the corners of the extent */
    points[0][0] = points[1][0] = points[2][0] = points[3][0] = box.min[0];
    points[4][0] = points[5][0] = points[6][0] = points[7][0] = box.max[0];

    points[0][1] = points[1][1] = points[4][1] = points[5][1] = box.min[1];
    points[2][1] = points[3][1] = points[6][1] = points[7][1] = box.max[1];

    points[0][2] = points[2][2] = points[4][2] = points[6][2] = box.min[2];
    points[1][2] = points[3][2] = points[5][2] = points[7][2] = box.max[2];

    Box< Vec3<S> > newBox;
    for (int i = 0; i < 8; i++) 
	newBox.extendBy(points[i] * m);

    return newBox;
}

template <class T>
Box< Vec3<T> >
affineTransform(const Box< Vec3<T> > &bbox, const Matrix44<T> &M)
{
    float       min0, max0, min1, max1, min2, max2, a, b;
    float       min0new, max0new, min1new, max1new, min2new, max2new;

    min0 = bbox.min[0];
    max0 = bbox.max[0];
    min1 = bbox.min[1];
    max1 = bbox.max[1];
    min2 = bbox.min[2];
    max2 = bbox.max[2];

    min0new = max0new = M[3][0];
    a = M[0][0] * min0;
    b = M[0][0] * max0;
    if (a < b) {
        min0new += a;
        max0new += b;
    } else {
        min0new += b;
        max0new += a;
    }
    a = M[1][0] * min1;
    b = M[1][0] * max1;
    if (a < b) {
        min0new += a;
        max0new += b;
    } else {
        min0new += b;
        max0new += a;
    }
    a = M[2][0] * min2;
    b = M[2][0] * max2;
    if (a < b) {
        min0new += a;
        max0new += b;
    } else {
        min0new += b;
        max0new += a;
    }

    min1new = max1new = M[3][1];
    a = M[0][1] * min0;
    b = M[0][1] * max0;
    if (a < b) {
        min1new += a;
        max1new += b;
    } else {
        min1new += b;
        max1new += a;
    }
    a = M[1][1] * min1;
    b = M[1][1] * max1;
    if (a < b) {
        min1new += a;
        max1new += b;
    } else {
        min1new += b;
        max1new += a;
    }
    a = M[2][1] * min2;
    b = M[2][1] * max2;
    if (a < b) {
        min1new += a;
        max1new += b;
    } else {
        min1new += b;
        max1new += a;
    }

    min2new = max2new = M[3][2];
    a = M[0][2] * min0;
    b = M[0][2] * max0;
    if (a < b) {
        min2new += a;
        max2new += b;
    } else {
        min2new += b;
        max2new += a;
    }
    a = M[1][2] * min1;
    b = M[1][2] * max1;
    if (a < b) {
        min2new += a;
        max2new += b;
    } else {
        min2new += b;
        max2new += a;
    }
    a = M[2][2] * min2;
    b = M[2][2] * max2;
    if (a < b) {
        min2new += a;
        max2new += b;
    } else {
        min2new += b;
        max2new += a;
    }

    Box< Vec3<T> > xbbox;

    xbbox.min[0] = min0new;
    xbbox.max[0] = max0new;
    xbbox.min[1] = min1new;
    xbbox.max[1] = max1new;
    xbbox.min[2] = min2new;
    xbbox.max[2] = max2new;

    return xbbox;
}


template <class T>
bool findEntryAndExitPoints(const Line3<T>& line,
			    const Box<Vec3<T> >& box,
			    Vec3<T> &enterPoint,
			    Vec3<T> &exitPoint)
{
    if ( box.isEmpty() ) return false;
    if ( line.distanceTo(box.center()) > box.size().length()/2. ) return false;

    Vec3<T>	points[8], inter, bary;
    Plane3<T>	plane;
    int		i, v0, v1, v2;
    bool	front = false, valid, validIntersection = false;

    // set up the eight coords of the corners of the box
    for(i = 0; i < 8; i++) 
    {
	points[i].setValue( i & 01 ? box.min[0] : box.max[0],
			    i & 02 ? box.min[1] : box.max[1],
			    i & 04 ? box.min[2] : box.max[2]);
    }

    // intersect the 12 triangles.
    for(i = 0; i < 12; i++) 
    {
	switch(i) 
        {
	case  0: v0 = 2; v1 = 1; v2 = 0; break;		// +z
	case  1: v0 = 2; v1 = 3; v2 = 1; break;

	case  2: v0 = 4; v1 = 5; v2 = 6; break;		// -z
	case  3: v0 = 6; v1 = 5; v2 = 7; break;

	case  4: v0 = 0; v1 = 6; v2 = 2; break;		// -x
	case  5: v0 = 0; v1 = 4; v2 = 6; break;

	case  6: v0 = 1; v1 = 3; v2 = 7; break;		// +x
	case  7: v0 = 1; v1 = 7; v2 = 5; break;

	case  8: v0 = 1; v1 = 4; v2 = 0; break;		// -y
	case  9: v0 = 1; v1 = 5; v2 = 4; break;

	case 10: v0 = 2; v1 = 7; v2 = 3; break;		// +y
	case 11: v0 = 2; v1 = 6; v2 = 7; break;
	}
	if((valid=intersect (line, points[v0], points[v1], points[v2],
                             inter, bary, front)) == true) 
        {
	    if(front == true) 
            {
		enterPoint = inter;
		validIntersection = valid;
	    }
	    else 
            {
		exitPoint = inter;
		validIntersection = valid;
	    }
	}
    }
    return validIntersection;
}

template<class T>
bool intersects(const Box< Vec3<T> > &box, 
		const Line3<T> &line,
		Vec3<T> &result)
{
    /* 
       Fast Ray-Box Intersection
       by Andrew Woo
       from "Graphics Gems", Academic Press, 1990
    */

    const int right	= 0;
    const int left	= 1;
    const int middle	= 2;

    const Vec3<T> &minB = box.min;
    const Vec3<T> &maxB = box.max;
    const Vec3<T> &origin = line.pos;
    const Vec3<T> &dir = line.dir;

    bool inside = true;
    char quadrant[3];
    int whichPlane;
    float maxT[3];
    float candidatePlane[3];

    /* Find candidate planes; this loop can be avoided if
   	rays cast all from the eye(assume perpsective view) */
    for (int i=0; i<3; i++)
    {
	if(origin[i] < minB[i]) 
	{
	    quadrant[i] = left;
	    candidatePlane[i] = minB[i];
	    inside = false;
	}
	else if (origin[i] > maxB[i]) 
	{
	    quadrant[i] = right;
	    candidatePlane[i] = maxB[i];
	    inside = false;
	}
	else	
	{
	    quadrant[i] = middle;
	}
    }

    /* Ray origin inside bounding box */
    if ( inside )	
    {
	result = origin;
	return true;
    }


	/* Calculate T distances to candidate planes */
    for (int i = 0; i < 3; i++)
    {
	if (quadrant[i] != middle && dir[i] !=0.)
	{
	    maxT[i] = (candidatePlane[i]-origin[i]) / dir[i];
	}
	else
	{
	    maxT[i] = -1.;
	}
    }

    /* Get largest of the maxT's for final choice of intersection */
    whichPlane = 0;

    for (int i = 1; i < 3; i++)
    {
	if (maxT[whichPlane] < maxT[i])
	{
	    whichPlane = i;
	}
    }

    /* Check final candidate actually inside box */
    if (maxT[whichPlane] < 0.) return false;

    for (int i = 0; i < 3; i++)
    {
	if (whichPlane != i) 
	{
	    result[i] = origin[i] + maxT[whichPlane] *dir[i];

	    if ((quadrant[i] == right && result[i] < minB[i]) ||
		(quadrant[i] == left && result[i] > maxB[i]))
	    {
		return false;	/* outside box */
	    }
	}
	else 
	{
	    result[i] = candidatePlane[i];
	}
    }

    return true;
}

template<class T>
bool intersects(const Box< Vec3<T> > &box, const Line3<T> &line)
{
    Vec3<T> ignored;
    return intersects(box,line,ignored);
}


} // namespace Imath

#endif
