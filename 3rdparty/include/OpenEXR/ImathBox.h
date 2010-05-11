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


#ifndef INCLUDED_IMATHBOX_H
#define INCLUDED_IMATHBOX_H

//-------------------------------------------------------------------
//
//	class Imath::Box<class T>
//	--------------------------------
//
//	This class imposes the following requirements on its 
//	parameter class:
//	
//	1) The class T must implement these operators:
//			+ - < > <= >= = 
//	   with the signature (T,T) and the expected 
//	   return values for a numeric type. 
//
//	2) The class T must implement operator=
//	   with the signature (T,float and/or double)
//
//	3) The class T must have a constructor which takes
//	   a float (and/or double) for use in initializing the box.
//
//	4) The class T must have a function T::dimensions()
//	   which returns the number of dimensions in the class
//	   (since its assumed its a vector) -- preferably, this
//	   returns a constant expression.
//
//-------------------------------------------------------------------

#include "ImathVec.h"

namespace Imath {


template <class T>	
class Box
{
  public:

    //-------------------------
    //  Data Members are public
    //-------------------------

    T				min;
    T				max;

    //-----------------------------------------------------
    //	Constructors - an "empty" box is created by default
    //-----------------------------------------------------

    Box(); 
    Box(const T& point);
    Box(const T& minT, const T& maxT);

    //--------------------
    //  Operators:  ==, !=
    //--------------------
    
    bool                        operator == (const Box<T> &src) const;
    bool                        operator != (const Box<T> &src) const;

    //------------------
    //	Box manipulation
    //------------------

    void			makeEmpty();
    void			extendBy(const T& point);
    void			extendBy(const Box<T>& box);

    //---------------------------------------------------
    //	Query functions - these compute results each time
    //---------------------------------------------------

    T				size() const;
    T				center() const;
    bool			intersects(const T &point) const;
    bool			intersects(const Box<T> &box) const;

    unsigned int		majorAxis() const;

    //----------------
    //	Classification
    //----------------

    bool			isEmpty() const;
    bool			hasVolume() const;
};


//--------------------
// Convenient typedefs
//--------------------


typedef Box <V2s> Box2s;
typedef Box <V2i> Box2i;
typedef Box <V2f> Box2f;
typedef Box <V2d> Box2d;
typedef Box <V3s> Box3s;
typedef Box <V3i> Box3i;
typedef Box <V3f> Box3f;
typedef Box <V3d> Box3d;


//----------------
//  Implementation
//----------------


template <class T>
inline Box<T>::Box()
{
    makeEmpty();
}

template <class T>
inline Box<T>::Box(const T& point)
{
    min = point;
    max = point;
}

template <class T>
inline Box<T>::Box(const T& minV, const T& maxV)
{
    min = minV;
    max = maxV;
}

template <class T>
inline bool
Box<T>::operator == (const Box<T> &src) const
{
    return (min == src.min && max == src.max);
}

template <class T>
inline bool
Box<T>::operator != (const Box<T> &src) const
{
    return (min != src.min || max != src.max);
}

template <class T>
inline void Box<T>::makeEmpty()
{
    min = T(T::baseTypeMax());
    max = T(T::baseTypeMin());
}

template <class T>
inline void Box<T>::extendBy(const T& point)
{
    for (unsigned int i=0; i<min.dimensions(); i++)
    {
	if ( point[i] < min[i] ) min[i] = point[i];
	if ( point[i] > max[i] ) max[i] = point[i];
    }
}

template <class T>
inline void Box<T>::extendBy(const Box<T>& box)
{
    for (unsigned int i=0; i<min.dimensions(); i++)
    {
	if ( box.min[i] < min[i] ) min[i] = box.min[i];
	if ( box.max[i] > max[i] ) max[i] = box.max[i];
    }
}

template <class T>
inline bool Box<T>::intersects(const T& point) const
{
    for (unsigned int i=0; i<min.dimensions(); i++)
    {
        if (point[i] < min[i] || point[i] > max[i]) return false;
    }
    return true;
}

template <class T>
inline bool Box<T>::intersects(const Box<T>& box) const
{
    for (unsigned int i=0; i<min.dimensions(); i++)
    {
        if (box.max[i] < min[i] || box.min[i] > max[i]) return false;
    }
    return true;
}

template <class T> 
inline T Box<T>::size() const 
{ 
    if (isEmpty())
	return T (0);

    return max-min;
}

template <class T> 
inline T Box<T>::center() const 
{ 
    return (max+min)/2;
}

template <class T>
inline bool Box<T>::isEmpty() const
{
    for (unsigned int i=0; i<min.dimensions(); i++)
    {
        if (max[i] < min[i]) return true;
    }
    return false;
}

template <class T>
inline bool Box<T>::hasVolume() const
{
    for (unsigned int i=0; i<min.dimensions(); i++)
    {
        if (max[i] <= min[i]) return false;
    }
    return true;
}

template<class T>
inline unsigned int Box<T>::majorAxis() const
{
    unsigned int major = 0;
    T s = size();

    for (unsigned int i=1; i<min.dimensions(); i++)
    {
	if ( s[i] > s[major] ) major = i;
    }

    return major;
}

} // namespace Imath

#endif
