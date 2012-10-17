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



#ifndef INCLUDED_IMATHROOTS_H
#define INCLUDED_IMATHROOTS_H

//---------------------------------------------------------------------
//
//	Functions to solve linear, quadratic or cubic equations
//
//---------------------------------------------------------------------

#include <ImathMath.h>
#include <complex>

namespace Imath {

//--------------------------------------------------------------------------
// Find the real solutions of a linear, quadratic or cubic equation:
//
//   	function				   equation solved
//
//   solveLinear (a, b, x)		                      a * x + b == 0
//   solveQuadratic (a, b, c, x)	            a * x*x + b * x + c == 0
//   solveNormalizedCubic (r, s, t, x)	    x*x*x + r * x*x + s * x + t == 0
//   solveCubic (a, b, c, d, x)		a * x*x*x + b * x*x + c * x + d == 0
//
// Return value:
//
//	 3	three real solutions, stored in x[0], x[1] and x[2]
//	 2	two real solutions, stored in x[0] and x[1]
//	 1	one real solution, stored in x[1]
//	 0	no real solutions
//	-1	all real numbers are solutions
//
// Notes:
//
//    * It is possible that an equation has real solutions, but that the
//	solutions (or some intermediate result) are not representable.
//	In this case, either some of the solutions returned are invalid
//	(nan or infinity), or, if floating-point exceptions have been
//	enabled with Iex::mathExcOn(), an Iex::MathExc exception is
//	thrown.
//
//    * Cubic equations are solved using Cardano's Formula; even though
//	only real solutions are produced, some intermediate results are
//	complex (std::complex<T>).
//
//--------------------------------------------------------------------------

template <class T> int	solveLinear (T a, T b, T &x);
template <class T> int	solveQuadratic (T a, T b, T c, T x[2]);
template <class T> int	solveNormalizedCubic (T r, T s, T t, T x[3]);
template <class T> int	solveCubic (T a, T b, T c, T d, T x[3]);


//---------------
// Implementation
//---------------

template <class T>
int
solveLinear (T a, T b, T &x)
{
    if (a != 0)
    {
    x = -b / a;
    return 1;
    }
    else if (b != 0)
    {
    return 0;
    }
    else
    {
    return -1;
    }
}


template <class T>
int
solveQuadratic (T a, T b, T c, T x[2])
{
    if (a == 0)
    {
    return solveLinear (b, c, x[0]);
    }
    else
    {
    T D = b * b - 4 * a * c;

    if (D > 0)
    {
        T s = Math<T>::sqrt (D);
        T q = -(b + (b > 0 ? 1 : -1) * s) / T(2);

        x[0] = q / a;
        x[1] = c / q;
        return 2;
    }
    if (D == 0)
    {
        x[0] = -b / (2 * a);
        return 1;
    }
    else
    {
        return 0;
    }
    }
}


template <class T>
int
solveNormalizedCubic (T r, T s, T t, T x[3])
{
    T p  = (3 * s - r * r) / 3;
    T q  = 2 * r * r * r / 27 - r * s / 3 + t;
    T p3 = p / 3;
    T q2 = q / 2;
    T D  = p3 * p3 * p3 + q2 * q2;

    if (D == 0 && p3 == 0)
    {
    x[0] = -r / 3;
    x[1] = -r / 3;
    x[2] = -r / 3;
    return 1;
    }

    std::complex<T> u = std::pow (-q / 2 + std::sqrt (std::complex<T> (D)),
                  T (1) / T (3));

    std::complex<T> v = -p / (T (3) * u);

    const T sqrt3 = T (1.73205080756887729352744634150587); // enough digits
                                // for long double
    std::complex<T> y0 (u + v);

    std::complex<T> y1 (-(u + v) / T (2) +
             (u - v) / T (2) * std::complex<T> (0, sqrt3));

    std::complex<T> y2 (-(u + v) / T (2) -
             (u - v) / T (2) * std::complex<T> (0, sqrt3));

    if (D > 0)
    {
    x[0] = y0.real() - r / 3;
    return 1;
    }
    else if (D == 0)
    {
    x[0] = y0.real() - r / 3;
    x[1] = y1.real() - r / 3;
    return 2;
    }
    else
    {
    x[0] = y0.real() - r / 3;
    x[1] = y1.real() - r / 3;
    x[2] = y2.real() - r / 3;
    return 3;
    }
}


template <class T>
int
solveCubic (T a, T b, T c, T d, T x[3])
{
    if (a == 0)
    {
    return solveQuadratic (b, c, d, x);
    }
    else
    {
    return solveNormalizedCubic (b / a, c / a, d / a, x);
    }
}


} // namespace Imath

#endif
