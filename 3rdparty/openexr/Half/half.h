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

// Primary authors:
//     Florian Kainz <kainz@ilm.com>
//     Rod Bogart <rgb@ilm.com>

//---------------------------------------------------------------------------
//
//	half -- a 16-bit floating point number class:
//
//	Type half can represent positive and negative numbers whose
//	magnitude is between roughly 6.1e-5 and 6.5e+4 with a relative
//	error of 9.8e-4; numbers smaller than 6.1e-5 can be represented
//	with an absolute error of 6.0e-8.  All integers from -2048 to
//	+2048 can be represented exactly.
//
//	Type half behaves (almost) like the built-in C++ floating point
//	types.  In arithmetic expressions, half, float and double can be
//	mixed freely.  Here are a few examples:
//
//	    half a (3.5);
//	    float b (a + sqrt (a));
//	    a += b;
//	    b += a;
//	    b = a + 7;
//
//	Conversions from half to float are lossless; all half numbers
//	are exactly representable as floats.
//
//	Conversions from float to half may not preserve a float's value
//	exactly.  If a float is not representable as a half, then the
//	float value is rounded to the nearest representable half.  If a
//	float value is exactly in the middle between the two closest
//	representable half values, then the float value is rounded to
//	the closest half whose least significant bit is zero.
//
//	Overflows during float-to-half conversions cause arithmetic
//	exceptions.  An overflow occurs when the float value to be
//	converted is too large to be represented as a half, or if the
//	float value is an infinity or a NAN.
//
//	The implementation of type half makes the following assumptions
//	about the implementation of the built-in C++ types:
//
//	    float is an IEEE 754 single-precision number
//	    sizeof (float) == 4
//	    sizeof (unsigned int) == sizeof (float)
//	    alignof (unsigned int) == alignof (float)
//	    sizeof (unsigned short) == 2
//
//---------------------------------------------------------------------------

#ifndef _HALF_H_
#define _HALF_H_

#include "halfExport.h"    // for definition of HALF_EXPORT
#include <iostream>

class half
{
  public:

    //-------------
    // Constructors
    //-------------

    half ();			// no initialization
    half (float f);


    //--------------------
    // Conversion to float
    //--------------------

    operator		float () const;


    //------------
    // Unary minus
    //------------

    half		operator - () const;


    //-----------
    // Assignment
    //-----------

    half &		operator = (half  h);
    half &		operator = (float f);

    half &		operator += (half  h);
    half &		operator += (float f);

    half &		operator -= (half  h);
    half &		operator -= (float f);

    half &		operator *= (half  h);
    half &		operator *= (float f);

    half &		operator /= (half  h);
    half &		operator /= (float f);


    //---------------------------------------------------------
    // Round to n-bit precision (n should be between 0 and 10).
    // After rounding, the significand's 10-n least significant
    // bits will be zero.
    //---------------------------------------------------------

    half		round (unsigned int n) const;


    //--------------------------------------------------------------------
    // Classification:
    //
    //	h.isFinite()		returns true if h is a normalized number,
    //				a denormalized number or zero
    //
    //	h.isNormalized()	returns true if h is a normalized number
    //
    //	h.isDenormalized()	returns true if h is a denormalized number
    //
    //	h.isZero()		returns true if h is zero
    //
    //	h.isNan()		returns true if h is a NAN
    //
    //	h.isInfinity()		returns true if h is a positive
    //				or a negative infinity
    //
    //	h.isNegative()		returns true if the sign bit of h
    //				is set (negative)
    //--------------------------------------------------------------------

    bool		isFinite () const;
    bool		isNormalized () const;
    bool		isDenormalized () const;
    bool		isZero () const;
    bool		isNan () const;
    bool		isInfinity () const;
    bool		isNegative () const;


    //--------------------------------------------
    // Special values
    //
    //	posInf()	returns +infinity
    //
    //	negInf()	returns -infinity
    //
    //	qNan()		returns a NAN with the bit
    //			pattern 0111111111111111
    //
    //	sNan()		returns a NAN with the bit
    //			pattern 0111110111111111
    //--------------------------------------------

    static half		posInf ();
    static half		negInf ();
    static half		qNan ();
    static half		sNan ();


    //--------------------------------------
    // Access to the internal representation
    //--------------------------------------

    HALF_EXPORT unsigned short	bits () const;
    HALF_EXPORT void		setBits (unsigned short bits);


  public:

    union uif
    {
	unsigned int	i;
	float		f;
    };

  private:

    HALF_EXPORT static short                  convert (int i);
    HALF_EXPORT static float                  overflow ();

    unsigned short                            _h;

    HALF_EXPORT static const uif              _toFloat[1 << 16];
    HALF_EXPORT static const unsigned short   _eLut[1 << 9];
};



//-----------
// Stream I/O
//-----------

HALF_EXPORT std::ostream &      operator << (std::ostream &os, half  h);
HALF_EXPORT std::istream &      operator >> (std::istream &is, half &h);


//----------
// Debugging
//----------

HALF_EXPORT void        printBits   (std::ostream &os, half  h);
HALF_EXPORT void        printBits   (std::ostream &os, float f);
HALF_EXPORT void        printBits   (char  c[19], half  h);
HALF_EXPORT void        printBits   (char  c[35], float f);


//-------------------------------------------------------------------------
// Limits
//
// Visual C++ will complain if HALF_MIN, HALF_NRM_MIN etc. are not float
// constants, but at least one other compiler (gcc 2.96) produces incorrect
// results if they are.
//-------------------------------------------------------------------------

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER

  #define HALF_MIN	5.96046448e-08f	// Smallest positive half

  #define HALF_NRM_MIN	6.10351562e-05f	// Smallest positive normalized half

  #define HALF_MAX	65504.0f	// Largest positive half

  #define HALF_EPSILON	0.00097656f	// Smallest positive e for which
					// half (1.0 + e) != half (1.0)
#else

  #define HALF_MIN	5.96046448e-08	// Smallest positive half

  #define HALF_NRM_MIN	6.10351562e-05	// Smallest positive normalized half

  #define HALF_MAX	65504.0		// Largest positive half

  #define HALF_EPSILON	0.00097656	// Smallest positive e for which
					// half (1.0 + e) != half (1.0)
#endif


#define HALF_MANT_DIG	11		// Number of digits in mantissa
					// (significand + hidden leading 1)

#define HALF_DIG	2		// Number of base 10 digits that
					// can be represented without change

#define HALF_DECIMAL_DIG	5	// Number of base-10 digits that are
					// necessary to uniquely represent all
					// distinct values

#define HALF_RADIX	2		// Base of the exponent

#define HALF_MIN_EXP	-13		// Minimum negative integer such that
					// HALF_RADIX raised to the power of
					// one less than that integer is a
					// normalized half

#define HALF_MAX_EXP	16		// Maximum positive integer such that
					// HALF_RADIX raised to the power of
					// one less than that integer is a
					// normalized half

#define HALF_MIN_10_EXP	-4		// Minimum positive integer such
					// that 10 raised to that power is
					// a normalized half

#define HALF_MAX_10_EXP	4		// Maximum positive integer such
					// that 10 raised to that power is
					// a normalized half


//---------------------------------------------------------------------------
//
// Implementation --
//
// Representation of a float:
//
//	We assume that a float, f, is an IEEE 754 single-precision
//	floating point number, whose bits are arranged as follows:
//
//	    31 (msb)
//	    | 
//	    | 30     23
//	    | |      | 
//	    | |      | 22                    0 (lsb)
//	    | |      | |                     |
//	    X XXXXXXXX XXXXXXXXXXXXXXXXXXXXXXX
//
//	    s e        m
//
//	S is the sign-bit, e is the exponent and m is the significand.
//
//	If e is between 1 and 254, f is a normalized number:
//
//	            s    e-127
//	    f = (-1)  * 2      * 1.m
//
//	If e is 0, and m is not zero, f is a denormalized number:
//
//	            s    -126
//	    f = (-1)  * 2      * 0.m
//
//	If e and m are both zero, f is zero:
//
//	    f = 0.0
//
//	If e is 255, f is an "infinity" or "not a number" (NAN),
//	depending on whether m is zero or not.
//
//	Examples:
//
//	    0 00000000 00000000000000000000000 = 0.0
//	    0 01111110 00000000000000000000000 = 0.5
//	    0 01111111 00000000000000000000000 = 1.0
//	    0 10000000 00000000000000000000000 = 2.0
//	    0 10000000 10000000000000000000000 = 3.0
//	    1 10000101 11110000010000000000000 = -124.0625
//	    0 11111111 00000000000000000000000 = +infinity
//	    1 11111111 00000000000000000000000 = -infinity
//	    0 11111111 10000000000000000000000 = NAN
//	    1 11111111 11111111111111111111111 = NAN
//
// Representation of a half:
//
//	Here is the bit-layout for a half number, h:
//
//	    15 (msb)
//	    | 
//	    | 14  10
//	    | |   |
//	    | |   | 9        0 (lsb)
//	    | |   | |        |
//	    X XXXXX XXXXXXXXXX
//
//	    s e     m
//
//	S is the sign-bit, e is the exponent and m is the significand.
//
//	If e is between 1 and 30, h is a normalized number:
//
//	            s    e-15
//	    h = (-1)  * 2     * 1.m
//
//	If e is 0, and m is not zero, h is a denormalized number:
//
//	            S    -14
//	    h = (-1)  * 2     * 0.m
//
//	If e and m are both zero, h is zero:
//
//	    h = 0.0
//
//	If e is 31, h is an "infinity" or "not a number" (NAN),
//	depending on whether m is zero or not.
//
//	Examples:
//
//	    0 00000 0000000000 = 0.0
//	    0 01110 0000000000 = 0.5
//	    0 01111 0000000000 = 1.0
//	    0 10000 0000000000 = 2.0
//	    0 10000 1000000000 = 3.0
//	    1 10101 1111000001 = -124.0625
//	    0 11111 0000000000 = +infinity
//	    1 11111 0000000000 = -infinity
//	    0 11111 1000000000 = NAN
//	    1 11111 1111111111 = NAN
//
// Conversion:
//
//	Converting from a float to a half requires some non-trivial bit
//	manipulations.  In some cases, this makes conversion relatively
//	slow, but the most common case is accelerated via table lookups.
//
//	Converting back from a half to a float is easier because we don't
//	have to do any rounding.  In addition, there are only 65536
//	different half numbers; we can convert each of those numbers once
//	and store the results in a table.  Later, all conversions can be
//	done using only simple table lookups.
//
//---------------------------------------------------------------------------


//--------------------
// Simple constructors
//--------------------

inline
half::half ()
{
    // no initialization
}


//----------------------------
// Half-from-float constructor
//----------------------------

inline
half::half (float f)
{
    uif x;

    x.f = f;

    if (f == 0)
    {
	//
	// Common special case - zero.
	// Preserve the zero's sign bit.
	//

	_h = (x.i >> 16);
    }
    else
    {
	//
	// We extract the combined sign and exponent, e, from our
	// floating-point number, f.  Then we convert e to the sign
	// and exponent of the half number via a table lookup.
	//
	// For the most common case, where a normalized half is produced,
	// the table lookup returns a non-zero value; in this case, all
	// we have to do is round f's significand to 10 bits and combine
	// the result with e.
	//
	// For all other cases (overflow, zeroes, denormalized numbers
	// resulting from underflow, infinities and NANs), the table
	// lookup returns zero, and we call a longer, non-inline function
	// to do the float-to-half conversion.
	//

	int e = (x.i >> 23) & 0x000001ff;

	e = _eLut[e];

	if (e)
	{
	    //
	    // Simple case - round the significand, m, to 10
	    // bits and combine it with the sign and exponent.
	    //

	    int m = x.i & 0x007fffff;
	    _h = (unsigned short)(e + ((m + 0x00000fff + ((m >> 13) & 1)) >> 13));
	}
	else
	{
	    //
	    // Difficult case - call a function.
	    //

	    _h = convert (x.i);
	}
    }
}


//------------------------------------------
// Half-to-float conversion via table lookup
//------------------------------------------

inline
half::operator float () const
{
    return _toFloat[_h].f;
}


//-------------------------
// Round to n-bit precision
//-------------------------

inline half
half::round (unsigned int n) const
{
    //
    // Parameter check.
    //

    if (n >= 10)
	return *this;

    //
    // Disassemble h into the sign, s,
    // and the combined exponent and significand, e.
    //

    unsigned short s = _h & 0x8000;
    unsigned short e = _h & 0x7fff;

    //
    // Round the exponent and significand to the nearest value
    // where ones occur only in the (10-n) most significant bits.
    // Note that the exponent adjusts automatically if rounding
    // up causes the significand to overflow.
    //

    e >>= 9 - n;
    e  += e & 1;
    e <<= 9 - n;

    //
    // Check for exponent overflow.
    //

    if (e >= 0x7c00)
    {
	//
	// Overflow occurred -- truncate instead of rounding.
	//

	e = _h;
	e >>= 10 - n;
	e <<= 10 - n;
    }

    //
    // Put the original sign bit back.
    //

    half h;
    h._h = s | e;

    return h;
}


//-----------------------
// Other inline functions
//-----------------------

inline half	
half::operator - () const
{
    half h;
    h._h = _h ^ 0x8000;
    return h;
}


inline half &
half::operator = (half h)
{
    _h = h._h;
    return *this;
}


inline half &
half::operator = (float f)
{
    *this = half (f);
    return *this;
}


inline half &
half::operator += (half h)
{
    *this = half (float (*this) + float (h));
    return *this;
}


inline half &
half::operator += (float f)
{
    *this = half (float (*this) + f);
    return *this;
}


inline half &
half::operator -= (half h)
{
    *this = half (float (*this) - float (h));
    return *this;
}


inline half &
half::operator -= (float f)
{
    *this = half (float (*this) - f);
    return *this;
}


inline half &
half::operator *= (half h)
{
    *this = half (float (*this) * float (h));
    return *this;
}


inline half &
half::operator *= (float f)
{
    *this = half (float (*this) * f);
    return *this;
}


inline half &
half::operator /= (half h)
{
    *this = half (float (*this) / float (h));
    return *this;
}


inline half &
half::operator /= (float f)
{
    *this = half (float (*this) / f);
    return *this;
}


inline bool	
half::isFinite () const
{
    unsigned short e = (_h >> 10) & 0x001f;
    return e < 31;
}


inline bool
half::isNormalized () const
{
    unsigned short e = (_h >> 10) & 0x001f;
    return e > 0 && e < 31;
}


inline bool
half::isDenormalized () const
{
    unsigned short e = (_h >> 10) & 0x001f;
    unsigned short m =  _h & 0x3ff;
    return e == 0 && m != 0;
}


inline bool
half::isZero () const
{
    return (_h & 0x7fff) == 0;
}


inline bool
half::isNan () const
{
    unsigned short e = (_h >> 10) & 0x001f;
    unsigned short m =  _h & 0x3ff;
    return e == 31 && m != 0;
}


inline bool
half::isInfinity () const
{
    unsigned short e = (_h >> 10) & 0x001f;
    unsigned short m =  _h & 0x3ff;
    return e == 31 && m == 0;
}


inline bool	
half::isNegative () const
{
    return (_h & 0x8000) != 0;
}


inline half
half::posInf ()
{
    half h;
    h._h = 0x7c00;
    return h;
}


inline half
half::negInf ()
{
    half h;
    h._h = 0xfc00;
    return h;
}


inline half
half::qNan ()
{
    half h;
    h._h = 0x7fff;
    return h;
}


inline half
half::sNan ()
{
    half h;
    h._h = 0x7dff;
    return h;
}


inline unsigned short
half::bits () const
{
    return _h;
}


inline void
half::setBits (unsigned short bits)
{
    _h = bits;
}

#endif
