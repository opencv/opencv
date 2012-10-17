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



#ifndef INCLUDED_IMATHFUN_H
#define INCLUDED_IMATHFUN_H

//-----------------------------------------------------------------------------
//
//	Miscellaneous utility functions
//
//-----------------------------------------------------------------------------

#include "ImathLimits.h"
#include "ImathInt64.h"

namespace Imath {

template <class T>
inline T
abs (T a)
{
    return (a > T(0)) ? a : -a;
}


template <class T>
inline int
sign (T a)
{
    return (a > T(0))? 1 : ((a < T(0)) ? -1 : 0);
}


template <class T, class Q>
inline T
lerp (T a, T b, Q t)
{
    return (T) (a * (1 - t) + b * t);
}


template <class T, class Q>
inline T
ulerp (T a, T b, Q t)
{
    return (T) ((a > b)? (a - (a - b) * t): (a + (b - a) * t));
}


template <class T>
inline T
lerpfactor(T m, T a, T b)
{
    //
    // Return how far m is between a and b, that is return t such that
    // if:
    //     t = lerpfactor(m, a, b);
    // then:
    //     m = lerp(a, b, t);
    //
    // If a==b, return 0.
    //

    T d = b - a;
    T n = m - a;

    if (abs(d) > T(1) || abs(n) < limits<T>::max() * abs(d))
    return n / d;

    return T(0);
}


template <class T>
inline T
clamp (T a, T l, T h)
{
    return (a < l)? l : ((a > h)? h : a);
}


template <class T>
inline int
cmp (T a, T b)
{
    return Imath::sign (a - b);
}


template <class T>
inline int
cmpt (T a, T b, T t)
{
    return (Imath::abs (a - b) <= t)? 0 : cmp (a, b);
}


template <class T>
inline bool
iszero (T a, T t)
{
    return (Imath::abs (a) <= t) ? 1 : 0;
}


template <class T1, class T2, class T3>
inline bool
equal (T1 a, T2 b, T3 t)
{
    return Imath::abs (a - b) <= t;
}

template <class T>
inline int
floor (T x)
{
    return (x >= 0)? int (x): -(int (-x) + (-x > int (-x)));
}


template <class T>
inline int
ceil (T x)
{
    return -floor (-x);
}

template <class T>
inline int
trunc (T x)
{
    return (x >= 0) ? int(x) : -int(-x);
}


//
// Integer division and remainder where the
// remainder of x/y has the same sign as x:
//
//	divs(x,y) == (abs(x) / abs(y)) * (sign(x) * sign(y))
//	mods(x,y) == x - y * divs(x,y)
//

inline int
divs (int x, int y)
{
    return (x >= 0)? ((y >= 0)?  ( x / y): -( x / -y)):
             ((y >= 0)? -(-x / y):  (-x / -y));
}


inline int
mods (int x, int y)
{
    return (x >= 0)? ((y >= 0)?  ( x % y):  ( x % -y)):
             ((y >= 0)? -(-x % y): -(-x % -y));
}


//
// Integer division and remainder where the
// remainder of x/y is always positive:
//
//	divp(x,y) == floor (double(x) / double (y))
//	modp(x,y) == x - y * divp(x,y)
//

inline int
divp (int x, int y)
{
    return (x >= 0)? ((y >= 0)?  (     x  / y): -(      x  / -y)):
             ((y >= 0)? -((y-1-x) / y):  ((-y-1-x) / -y));
}


inline int
modp (int x, int y)
{
    return x - y * divp (x, y);
}

//----------------------------------------------------------
// Successor and predecessor for floating-point numbers:
//
// succf(f)     returns float(f+e), where e is the smallest
//              positive number such that float(f+e) != f.
//
// predf(f)     returns float(f-e), where e is the smallest
//              positive number such that float(f-e) != f.
//
// succd(d)     returns double(d+e), where e is the smallest
//              positive number such that double(d+e) != d.
//
// predd(d)     returns double(d-e), where e is the smallest
//              positive number such that double(d-e) != d.
//
// Exceptions:  If the input value is an infinity or a nan,
//              succf(), predf(), succd(), and predd() all
//              return the input value without changing it.
//
//----------------------------------------------------------

float succf (float f);
float predf (float f);

double succd (double d);
double predd (double d);

//
// Return true if the number is not a NaN or Infinity.
//

inline bool
finitef (float f)
{
    union {float f; int i;} u;
    u.f = f;

    return (u.i & 0x7f800000) != 0x7f800000;
}

inline bool
finited (double d)
{
    union {double d; Int64 i;} u;
    u.d = d;

    return (u.i & 0x7ff0000000000000LL) != 0x7ff0000000000000LL;
}


} // namespace Imath

#endif
