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



#ifndef INCLUDED_IMATHPLANE_H
#define INCLUDED_IMATHPLANE_H

//----------------------------------------------------------------------
//
//	template class Plane3
//
//	The Imath::Plane3<> class represents a half space, so the
//	normal may point either towards or away from origin.  The
//	plane P can be represented by Imath::Plane3 as either p or -p
//	corresponding to the two half-spaces on either side of the
//	plane. Any function which computes a distance will return
//	either negative or positive values for the distance indicating
//	which half-space the point is in. Note that reflection, and
//	intersection functions will operate as expected.
//
//----------------------------------------------------------------------

#include "ImathVec.h"
#include "ImathLine.h"

namespace Imath {


template <class T>
class Plane3
{
  public:

    Vec3<T>			normal;
    T				distance;

    Plane3() {}
    Plane3(const Vec3<T> &normal, T distance);
    Plane3(const Vec3<T> &point, const Vec3<T> &normal);
    Plane3(const Vec3<T> &point1,
       const Vec3<T> &point2,
       const Vec3<T> &point3);

    //----------------------
    //	Various set methods
    //----------------------

    void                        set(const Vec3<T> &normal,
                    T distance);

    void                        set(const Vec3<T> &point,
                    const Vec3<T> &normal);

    void                        set(const Vec3<T> &point1,
                    const Vec3<T> &point2,
                    const Vec3<T> &point3 );

    //----------------------
    //	Utilities
    //----------------------

    bool                        intersect(const Line3<T> &line,
                                          Vec3<T> &intersection) const;

    bool                        intersectT(const Line3<T> &line,
                       T &parameter) const;

    T				distanceTo(const Vec3<T> &) const;

    Vec3<T>                     reflectPoint(const Vec3<T> &) const;
    Vec3<T>                     reflectVector(const Vec3<T> &) const;
};


//--------------------
// Convenient typedefs
//--------------------

typedef Plane3<float> Plane3f;
typedef Plane3<double> Plane3d;


//---------------
// Implementation
//---------------

template <class T>
inline Plane3<T>::Plane3(const Vec3<T> &p0,
             const Vec3<T> &p1,
             const Vec3<T> &p2)
{
    set(p0,p1,p2);
}

template <class T>
inline Plane3<T>::Plane3(const Vec3<T> &n, T d)
{
    set(n, d);
}

template <class T>
inline Plane3<T>::Plane3(const Vec3<T> &p, const Vec3<T> &n)
{
    set(p, n);
}

template <class T>
inline void Plane3<T>::set(const Vec3<T>& point1,
               const Vec3<T>& point2,
               const Vec3<T>& point3)
{
    normal = (point2 - point1) % (point3 - point1);
    normal.normalize();
    distance = normal ^ point1;
}

template <class T>
inline void Plane3<T>::set(const Vec3<T>& point, const Vec3<T>& n)
{
    normal = n;
    normal.normalize();
    distance = normal ^ point;
}

template <class T>
inline void Plane3<T>::set(const Vec3<T>& n, T d)
{
    normal = n;
    normal.normalize();
    distance = d;
}

template <class T>
inline T Plane3<T>::distanceTo(const Vec3<T> &point) const
{
    return (point ^ normal) - distance;
}

template <class T>
inline Vec3<T> Plane3<T>::reflectPoint(const Vec3<T> &point) const
{
    return normal * distanceTo(point) * -2.0 + point;
}


template <class T>
inline Vec3<T> Plane3<T>::reflectVector(const Vec3<T> &v) const
{
    return normal * (normal ^ v)  * 2.0 - v;
}


template <class T>
inline bool Plane3<T>::intersect(const Line3<T>& line, Vec3<T>& point) const
{
    T d = normal ^ line.dir;
    if ( d == 0.0 ) return false;
    T t = - ((normal ^ line.pos) - distance) /  d;
    point = line(t);
    return true;
}

template <class T>
inline bool Plane3<T>::intersectT(const Line3<T>& line, T &t) const
{
    T d = normal ^ line.dir;
    if ( d == 0.0 ) return false;
    t = - ((normal ^ line.pos) - distance) /  d;
    return true;
}

template<class T>
std::ostream &operator<< (std::ostream &o, const Plane3<T> &plane)
{
    return o << "(" << plane.normal << ", " << plane.distance
         << ")";
}

template<class T>
Plane3<T> operator* (const Plane3<T> &plane, const Matrix44<T> &M)
{
    //                        T
    //	                    -1
    //	Could also compute M    but that would suck.
    //

    Vec3<T> dir1   = Vec3<T> (1, 0, 0) % plane.normal;
    T dir1Len      = dir1 ^ dir1;

    Vec3<T> tmp    = Vec3<T> (0, 1, 0) % plane.normal;
    T tmpLen       = tmp ^ tmp;

    if (tmpLen > dir1Len)
    {
    dir1      = tmp;
    dir1Len   = tmpLen;
    }

    tmp            = Vec3<T> (0, 0, 1) % plane.normal;
    tmpLen         = tmp ^ tmp;

    if (tmpLen > dir1Len)
    {
    dir1      = tmp;
    }

    Vec3<T> dir2   = dir1 % plane.normal;
    Vec3<T> point  = plane.distance * plane.normal;

    return Plane3<T> ( point         * M,
              (point + dir2) * M,
              (point + dir1) * M );
}

template<class T>
Plane3<T> operator- (const Plane3<T> &plane)
{
    return Plane3<T>(-plane.normal,-plane.distance);
}


} // namespace Imath

#endif
