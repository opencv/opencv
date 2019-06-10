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



#ifndef INCLUDED_IMATHINTERVAL_H
#define INCLUDED_IMATHINTERVAL_H


//-------------------------------------------------------------------
//
//	class Imath::Interval<class T>
//	--------------------------------
//
//	An Interval has a min and a max and some miscellaneous
//	functions. It is basically a Box<T> that allows T to be
//	a scalar.
//
//-------------------------------------------------------------------

#include "ImathVec.h"
#include "ImathNamespace.h"

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER


template <class T>	
class Interval
{
  public:

    //-------------------------
    //  Data Members are public
    //-------------------------

    T				min;
    T				max;

    //-----------------------------------------------------
    //	Constructors - an "empty" Interval is created by default
    //-----------------------------------------------------

    Interval(); 
    Interval(const T& point);
    Interval(const T& minT, const T& maxT);

    //--------------------------------
    //  Operators:  we get != from STL
    //--------------------------------
    
    bool                        operator == (const Interval<T> &src) const;

    //------------------
    //	Interval manipulation
    //------------------

    void			makeEmpty();
    void			extendBy(const T& point);
    void			extendBy(const Interval<T>& interval);

    //---------------------------------------------------
    //	Query functions - these compute results each time
    //---------------------------------------------------

    T				size() const;
    T				center() const;
    bool			intersects(const T &point) const;
    bool			intersects(const Interval<T> &interval) const;

    //----------------
    //	Classification
    //----------------

    bool			hasVolume() const;
    bool			isEmpty() const;
};


//--------------------
// Convenient typedefs
//--------------------


typedef Interval <float>  Intervalf;
typedef Interval <double> Intervald;
typedef Interval <short>  Intervals;
typedef Interval <int>    Intervali;

//----------------
//  Implementation
//----------------


template <class T>
inline Interval<T>::Interval()
{
    makeEmpty();
}

template <class T>
inline Interval<T>::Interval(const T& point)
{
    min = point;
    max = point;
}

template <class T>
inline Interval<T>::Interval(const T& minV, const T& maxV)
{
    min = minV;
    max = maxV;
}

template <class T>
inline bool
Interval<T>::operator == (const Interval<T> &src) const
{
    return (min == src.min && max == src.max);
}

template <class T>
inline void
Interval<T>::makeEmpty()
{
    min = limits<T>::max();
    max = limits<T>::min();
}

template <class T>
inline void
Interval<T>::extendBy(const T& point)
{
    if ( point < min )
	min = point;
    
    if ( point > max )
	max = point;
}

template <class T>
inline void
Interval<T>::extendBy(const Interval<T>& interval)
{
    if ( interval.min < min )
	min = interval.min;

    if ( interval.max > max )
	max = interval.max;
}

template <class T>
inline bool
Interval<T>::intersects(const T& point) const
{
    return point >= min && point <= max;
}

template <class T>
inline bool
Interval<T>::intersects(const Interval<T>& interval) const
{
    return interval.max >= min && interval.min <= max;
}

template <class T> 
inline T
Interval<T>::size() const 
{ 
    return max-min;
}

template <class T> 
inline T
Interval<T>::center() const 
{ 
    return (max+min)/2;
}

template <class T>
inline bool
Interval<T>::isEmpty() const
{
    return max < min;
}

template <class T>
inline bool Interval<T>::hasVolume() const
{
    return max > min;
}


IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHINTERVAL_H
