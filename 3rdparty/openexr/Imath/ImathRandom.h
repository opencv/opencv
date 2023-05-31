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


#ifndef INCLUDED_IMATHRANDOM_H
#define INCLUDED_IMATHRANDOM_H

//-----------------------------------------------------------------------------
//
//	Generators for uniformly distributed pseudo-random numbers and
//	functions that use those generators to generate numbers with
//	non-uniform distributions:
//
//		class Rand32
//		class Rand48
//		solidSphereRand()
//		hollowSphereRand()
//		gaussRand()
//		gaussSphereRand()
//
//	Note: class Rand48() calls erand48() and nrand48(), which are not
//	available on all operating systems.  For compatibility we include
//	our own versions of erand48() and nrand48().  Our functions have
//	been reverse-engineered from the corresponding Unix/Linux man page.
//
//-----------------------------------------------------------------------------

#include "ImathNamespace.h"
#include "ImathExport.h"

#include <stdlib.h>
#include <math.h>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

//-----------------------------------------------
// Fast random-number generator that generates
// a uniformly distributed sequence with a period
// length of 2^32.
//-----------------------------------------------

class IMATH_EXPORT Rand32
{
  public:

    //------------
    // Constructor
    //------------

    Rand32 (unsigned long int seed = 0);
    

    //--------------------------------
    // Re-initialize with a given seed
    //--------------------------------

    void		init (unsigned long int seed);


    //----------------------------------------------------------
    // Get the next value in the sequence (range: [false, true])
    //----------------------------------------------------------

    bool		nextb ();


    //---------------------------------------------------------------
    // Get the next value in the sequence (range: [0 ... 0xffffffff])
    //---------------------------------------------------------------

    unsigned long int	nexti ();


    //------------------------------------------------------
    // Get the next value in the sequence (range: [0 ... 1[)
    //------------------------------------------------------

    float		nextf ();


    //-------------------------------------------------------------------
    // Get the next value in the sequence (range [rangeMin ... rangeMax[)
    //-------------------------------------------------------------------

    float		nextf (float rangeMin, float rangeMax);


  private:

    void		next ();

    unsigned long int	_state;
};


//--------------------------------------------------------
// Random-number generator based on the C Standard Library
// functions erand48(), nrand48() & company; generates a
// uniformly distributed sequence.
//--------------------------------------------------------

class Rand48
{
  public:

    //------------
    // Constructor
    //------------

    Rand48 (unsigned long int seed = 0);
    

    //--------------------------------
    // Re-initialize with a given seed
    //--------------------------------

    void		init (unsigned long int seed);


    //----------------------------------------------------------
    // Get the next value in the sequence (range: [false, true])
    //----------------------------------------------------------

    bool		nextb ();


    //---------------------------------------------------------------
    // Get the next value in the sequence (range: [0 ... 0x7fffffff])
    //---------------------------------------------------------------

    long int		nexti ();


    //------------------------------------------------------
    // Get the next value in the sequence (range: [0 ... 1[)
    //------------------------------------------------------

    double		nextf ();


    //-------------------------------------------------------------------
    // Get the next value in the sequence (range [rangeMin ... rangeMax[)
    //-------------------------------------------------------------------

    double		nextf (double rangeMin, double rangeMax);


  private:

    unsigned short int	_state[3];
};


//------------------------------------------------------------
// Return random points uniformly distributed in a sphere with
// radius 1 around the origin (distance from origin <= 1).
//------------------------------------------------------------

template <class Vec, class Rand>
Vec		
solidSphereRand (Rand &rand);


//-------------------------------------------------------------
// Return random points uniformly distributed on the surface of
// a sphere with radius 1 around the origin.
//-------------------------------------------------------------

template <class Vec, class Rand>
Vec		
hollowSphereRand (Rand &rand);


//-----------------------------------------------
// Return random numbers with a normal (Gaussian)
// distribution with zero mean and unit variance.
//-----------------------------------------------

template <class Rand>
float
gaussRand (Rand &rand);


//----------------------------------------------------
// Return random points whose distance from the origin
// has a normal (Gaussian) distribution with zero mean
// and unit variance.
//----------------------------------------------------

template <class Vec, class Rand>
Vec
gaussSphereRand (Rand &rand);


//---------------------------------
// erand48(), nrand48() and friends
//---------------------------------

IMATH_EXPORT double     erand48 (unsigned short state[3]);
IMATH_EXPORT double     drand48 ();
IMATH_EXPORT long int   nrand48 (unsigned short state[3]);
IMATH_EXPORT long int   lrand48 ();
IMATH_EXPORT void       srand48 (long int seed);


//---------------
// Implementation
//---------------


inline void
Rand32::init (unsigned long int seed)
{
    _state = (seed * 0xa5a573a5L) ^ 0x5a5a5a5aL;
}


inline
Rand32::Rand32 (unsigned long int seed)
{
    init (seed);
}


inline void
Rand32::next ()
{
    _state = 1664525L * _state + 1013904223L;
}


inline bool
Rand32::nextb ()
{
    next ();
    // Return the 31st (most significant) bit, by and-ing with 2 ^ 31.
    return !!(_state & 2147483648UL);
}


inline unsigned long int
Rand32::nexti ()
{
    next ();
    return _state & 0xffffffff;
}


inline float
Rand32::nextf (float rangeMin, float rangeMax)
{
    float f = nextf();
    return rangeMin * (1 - f) + rangeMax * f;
}


inline void
Rand48::init (unsigned long int seed)
{
    seed = (seed * 0xa5a573a5L) ^ 0x5a5a5a5aL;

    _state[0] = (unsigned short int) (seed & 0xFFFF);
    _state[1] = (unsigned short int) ((seed >> 16) & 0xFFFF);
    _state[2] = (unsigned short int) (seed & 0xFFFF);   
}


inline 
Rand48::Rand48 (unsigned long int seed)
{
    init (seed);
}


inline bool
Rand48::nextb ()
{
    return nrand48 (_state) & 1;
}


inline long int
Rand48::nexti ()
{
    return nrand48 (_state);
}


inline double
Rand48::nextf ()
{
    return erand48 (_state);
}


inline double
Rand48::nextf (double rangeMin, double rangeMax)
{
    double f = nextf();
    return rangeMin * (1 - f) + rangeMax * f;
}


template <class Vec, class Rand>
Vec
solidSphereRand (Rand &rand)
{
    Vec v;

    do
    {
	for (unsigned int i = 0; i < Vec::dimensions(); i++)
	    v[i] = (typename Vec::BaseType) rand.nextf (-1, 1);
    }
    while (v.length2() > 1);

    return v;
}


template <class Vec, class Rand>
Vec
hollowSphereRand (Rand &rand)
{
    Vec v;
    typename Vec::BaseType length;

    do
    {
	for (unsigned int i = 0; i < Vec::dimensions(); i++)
	    v[i] = (typename Vec::BaseType) rand.nextf (-1, 1);

	length = v.length();
    }
    while (length > 1 || length == 0);

    return v / length;
}


template <class Rand>
float
gaussRand (Rand &rand)
{
    float x;		// Note: to avoid numerical problems with very small
    float y;		// numbers, we make these variables singe-precision
    float length2;	// floats, but later we call the double-precision log()
			// and sqrt() functions instead of logf() and sqrtf().
    do
    {
	x = float (rand.nextf (-1, 1));
	y = float (rand.nextf (-1, 1));
	length2 = x * x + y * y;
    }
    while (length2 >= 1 || length2 == 0);

    return x * sqrt (-2 * log (double (length2)) / length2);
}


template <class Vec, class Rand>
Vec
gaussSphereRand (Rand &rand)
{
    return hollowSphereRand <Vec> (rand) * gaussRand (rand);
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHRANDOM_H
