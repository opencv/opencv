//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Generators for uniformly distributed pseudo-random numbers and
// functions that use those generators to generate numbers with
// non-uniform distributions
//
// Note: class Rand48() calls erand48() and nrand48(), which are not
// available on all operating systems.  For compatibility we include
// our own versions of erand48() and nrand48().  Our functions
// have been reverse-engineered from the corresponding Unix/Linux
// man page.
//

#ifndef INCLUDED_IMATHRANDOM_H
#define INCLUDED_IMATHRANDOM_H

#include "ImathExport.h"
#include "ImathNamespace.h"

#include <math.h>
#include <stdlib.h>

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

/// Fast random-number generator that generates
/// a uniformly distributed sequence with a period
/// length of 2^32.
class Rand32
{
  public:

    /// Constructor, given a seed
    IMATH_HOSTDEVICE Rand32 (unsigned long int seed = 0);

    /// Re-initialize with a given seed
    IMATH_HOSTDEVICE void init (unsigned long int seed);

    /// Get the next value in the sequence (range: [false, true])
    IMATH_HOSTDEVICE bool nextb();

    /// Get the next value in the sequence (range: [0 ... 0xffffffff])
    IMATH_HOSTDEVICE unsigned long int nexti();

    /// Get the next value in the sequence (range: [0 ... 1[)
    IMATH_HOSTDEVICE IMATH_EXPORT float nextf ();

    /// Get the next value in the sequence (range [rangeMin ... rangeMax[)
    IMATH_HOSTDEVICE float nextf (float rangeMin, float rangeMax);

  private:
    IMATH_HOSTDEVICE void next();

    unsigned long int _state;
};

/// Random-number generator based on the C Standard Library
/// functions erand48(), nrand48() & company; generates a
/// uniformly distributed sequence.
class Rand48
{
  public:

    /// Constructor
    IMATH_HOSTDEVICE Rand48 (unsigned long int seed = 0);

    /// Re-initialize with a given seed
    IMATH_HOSTDEVICE void init (unsigned long int seed);

    /// Get the next value in the sequence (range: [false, true])
    IMATH_HOSTDEVICE bool nextb();

    /// Get the next value in the sequence (range: [0 ... 0x7fffffff])
    IMATH_HOSTDEVICE long int nexti();

    /// Get the next value in the sequence (range: [0 ... 1[)
    IMATH_HOSTDEVICE double nextf();

    /// Get the next value in the sequence (range [rangeMin ... rangeMax[)
    IMATH_HOSTDEVICE double nextf (double rangeMin, double rangeMax);

  private:
    unsigned short int _state[3];
};

/// Return random points uniformly distributed in a sphere with
/// radius 1 around the origin (distance from origin <= 1).
template <class Vec, class Rand> IMATH_HOSTDEVICE Vec solidSphereRand (Rand& rand);

/// Return random points uniformly distributed on the surface of
/// a sphere with radius 1 around the origin.
template <class Vec, class Rand> IMATH_HOSTDEVICE Vec hollowSphereRand (Rand& rand);

/// Return random numbers with a normal (Gaussian)
/// distribution with zero mean and unit variance.
template <class Rand> IMATH_HOSTDEVICE float gaussRand (Rand& rand);

/// Return random points whose distance from the origin
/// has a normal (Gaussian) distribution with zero mean
/// and unit variance.
template <class Vec, class Rand> IMATH_HOSTDEVICE Vec gaussSphereRand (Rand& rand);

//---------------------------------
// erand48(), nrand48() and friends
//---------------------------------

/// @cond Doxygen_Suppress
IMATH_HOSTDEVICE IMATH_EXPORT double erand48 (unsigned short state[3]);
IMATH_HOSTDEVICE IMATH_EXPORT double drand48();
IMATH_HOSTDEVICE IMATH_EXPORT long int nrand48 (unsigned short state[3]);
IMATH_HOSTDEVICE IMATH_EXPORT long int lrand48();
IMATH_HOSTDEVICE IMATH_EXPORT void srand48 (long int seed);
/// @endcond

//---------------
// Implementation
//---------------

IMATH_HOSTDEVICE inline void
Rand32::init (unsigned long int seed)
{
    _state = (seed * 0xa5a573a5L) ^ 0x5a5a5a5aL;
}

IMATH_HOSTDEVICE inline Rand32::Rand32 (unsigned long int seed)
{
    init (seed);
}

IMATH_HOSTDEVICE inline void
Rand32::next()
{
    _state = 1664525L * _state + 1013904223L;
}

IMATH_HOSTDEVICE inline bool
Rand32::nextb()
{
    next();
    // Return the 31st (most significant) bit, by and-ing with 2 ^ 31.
    return !!(_state & 2147483648UL);
}

IMATH_HOSTDEVICE inline unsigned long int
Rand32::nexti()
{
    next();
    return _state & 0xffffffff;
}

IMATH_HOSTDEVICE inline float
Rand32::nextf (float rangeMin, float rangeMax)
{
    float f = nextf();
    return rangeMin * (1 - f) + rangeMax * f;
}

IMATH_HOSTDEVICE inline void
Rand48::init (unsigned long int seed)
{
    seed = (seed * 0xa5a573a5L) ^ 0x5a5a5a5aL;

    _state[0] = (unsigned short int) (seed & 0xFFFF);
    _state[1] = (unsigned short int) ((seed >> 16) & 0xFFFF);
    _state[2] = (unsigned short int) (seed & 0xFFFF);
}

IMATH_HOSTDEVICE inline Rand48::Rand48 (unsigned long int seed)
{
    init (seed);
}

IMATH_HOSTDEVICE inline bool
Rand48::nextb()
{
    return nrand48 (_state) & 1;
}

IMATH_HOSTDEVICE inline long int
Rand48::nexti()
{
    return nrand48 (_state);
}

IMATH_HOSTDEVICE inline double
Rand48::nextf()
{
    return erand48 (_state);
}

IMATH_HOSTDEVICE inline double
Rand48::nextf (double rangeMin, double rangeMax)
{
    double f = nextf();
    return rangeMin * (1 - f) + rangeMax * f;
}

template <class Vec, class Rand>
IMATH_HOSTDEVICE Vec
solidSphereRand (Rand& rand)
{
    Vec v;

    do
    {
        for (unsigned int i = 0; i < Vec::dimensions(); i++)
            v[i] = (typename Vec::BaseType) rand.nextf (-1, 1);
    } while (v.length2() > 1);

    return v;
}

template <class Vec, class Rand>
IMATH_HOSTDEVICE Vec
hollowSphereRand (Rand& rand)
{
    Vec v;
    typename Vec::BaseType length;

    do
    {
        for (unsigned int i = 0; i < Vec::dimensions(); i++)
            v[i] = (typename Vec::BaseType) rand.nextf (-1, 1);

        length = v.length();
    } while (length > 1 || length == 0);

    return v / length;
}

template <class Rand>
IMATH_HOSTDEVICE float
gaussRand (Rand& rand)
{
    float x;       // Note: to avoid numerical problems with very small
    float y;       // numbers, we make these variables singe-precision
    float length2; // floats, but later we call the double-precision log()
                   // and sqrt() functions instead of logf() and sqrtf().
    do
    {
        x       = float (rand.nextf (-1, 1));
        y       = float (rand.nextf (-1, 1));
        length2 = x * x + y * y;
    } while (length2 >= 1 || length2 == 0);

    return x * sqrt (-2 * log (double (length2)) / length2);
}

template <class Vec, class Rand>
IMATH_HOSTDEVICE Vec
gaussSphereRand (Rand& rand)
{
    return hollowSphereRand<Vec> (rand) * gaussRand (rand);
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHRANDOM_H
