//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Functions to solve linear, quadratic or cubic equations
//
// Note: It is possible that an equation has real solutions, but that
// the solutions (or some intermediate result) are not representable.
// In this case, either some of the solutions returned are invalid
// (nan or infinity), or, if floating-point exceptions have been
// enabled, an exception is thrown.
//

#ifndef INCLUDED_IMATHROOTS_H
#define INCLUDED_IMATHROOTS_H

#include "ImathMath.h"
#include "ImathNamespace.h"
#include <complex>

/// @cond Doxygen_Suppress

#ifdef __CUDACC__
#    include <thrust/complex.h>
#    define COMPLEX_NAMESPACE thrust
#else
#    define COMPLEX_NAMESPACE std
#endif

/// @endcond

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

///
/// Solve for x in the linear equation:
///
///     a * x + b == 0
///
/// @return 1 if the equation has a solution, 0 if there is no
/// solution, and -1 if all real numbers are solutions.
template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 int solveLinear (T a, T b, T& x);

///
/// Solve for x in the quadratic equation:
///
///     a * x*x + b * x + c == 0
///
/// @return 2 if the equation has two solutions, 1 if the equation has
/// a single solution, 0 if there is no solution, and -1 if all real
/// numbers are solutions.
template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 int solveQuadratic (T a, T b, T c, T x[2]);
template <class T>

///
/// Solve for x in the normalized cubic equation:
///
///     x*x*x + r * x*x + s * x + t == 0
///
/// The equation is solved using Cardano's Formula; even though only
/// real solutions are produced, some intermediate results are complex
/// (std::complex<T>).
///
/// @return 0 if there is no solution, and -1 if all real
/// numbers are solutions, otherwise return the number of solutions.
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 int solveNormalizedCubic (T r, T s, T t, T x[3]);

///
/// Solve for x in the cubic equation:
///
///     a * x*x*x + b * x*x + c * x + d == 0
///
/// The equation is solved using Cardano's Formula; even though only
/// real solutions are produced, some intermediate results are complex
/// (std::complex<T>).
///
/// @return 0 if there is no solution, and -1 if all real
/// numbers are solutions, otherwise return the number of solutions.
template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 int solveCubic (T a, T b, T c, T d, T x[3]);

//---------------
// Implementation
//---------------

template <class T>
IMATH_CONSTEXPR14 int
solveLinear (T a, T b, T& x)
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
IMATH_CONSTEXPR14 int
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
            T s = std::sqrt (D);
            T q = -(b + (b > 0 ? 1 : -1) * s) / T (2);

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
IMATH_CONSTEXPR14 int
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

    if (D > 0)
    {
        auto real_root = [] (T a, T x) -> T {
            T sign = std::copysign(T(1), a);
            return sign * std::pow (sign * a, T (1) / x);
        };

        T u = real_root (-q / 2 + std::sqrt (D), 3);
        T v = -p / (T (3) * u);

        x[0] = u + v - r / 3;
        return 1;
    }

    namespace CN     = COMPLEX_NAMESPACE;
    CN::complex<T> u = CN::pow (-q / 2 + CN::sqrt (CN::complex<T> (D)), T (1) / T (3));
    CN::complex<T> v = -p / (T (3) * u);

    const T sqrt3 = T (1.73205080756887729352744634150587); // enough digits
                                                            // for long double
    CN::complex<T> y0 (u + v);
    CN::complex<T> y1 (-(u + v) / T (2) + (u - v) / T (2) * CN::complex<T> (0, sqrt3));
    CN::complex<T> y2 (-(u + v) / T (2) - (u - v) / T (2) * CN::complex<T> (0, sqrt3));

    if (D == 0)
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
IMATH_CONSTEXPR14 int
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

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHROOTS_H
