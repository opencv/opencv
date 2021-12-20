//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_CHECKED_ARITHMETIC_H
#define INCLUDED_IMF_CHECKED_ARITHMETIC_H

//-----------------------------------------------------------------------------
//
//	Integer arithmetic operations that throw exceptions
//      on overflow, underflow or division by zero.
//
//-----------------------------------------------------------------------------

#include "ImfNamespace.h"

#include "IexMathExc.h"

#include <limits>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

template <bool b> struct StaticAssertionFailed;
template <> struct StaticAssertionFailed <true> {};

#define IMF_STATIC_ASSERT(x) \
    do {StaticAssertionFailed <x> staticAssertionFailed; ((void) staticAssertionFailed);} while (false)


template <class T>
inline T
uiMult (T a, T b)
{
    //
    // Unsigned integer multiplication
    //

    IMF_STATIC_ASSERT (!std::numeric_limits<T>::is_signed &&
                        std::numeric_limits<T>::is_integer);

    if (a > 0 && b > std::numeric_limits<T>::max() / a)
        throw IEX_NAMESPACE::OverflowExc ("Integer multiplication overflow.");

    return a * b;
}


template <class T>
inline T
uiDiv (T a, T b)
{
    //
    // Unsigned integer division
    //

    IMF_STATIC_ASSERT (!std::numeric_limits<T>::is_signed &&
                        std::numeric_limits<T>::is_integer);

    if (b == 0)
        throw IEX_NAMESPACE::DivzeroExc ("Integer division by zero.");

    return a / b;
}


template <class T>
inline T
uiAdd (T a, T b)
{
    //
    // Unsigned integer addition
    //

    IMF_STATIC_ASSERT (!std::numeric_limits<T>::is_signed &&
                        std::numeric_limits<T>::is_integer);

    if (a > std::numeric_limits<T>::max() - b)
        throw IEX_NAMESPACE::OverflowExc ("Integer addition overflow.");

    return a + b;
}


template <class T>
inline T
uiSub (T a, T b)
{
    //
    // Unsigned integer subtraction
    //

    IMF_STATIC_ASSERT (!std::numeric_limits<T>::is_signed &&
                        std::numeric_limits<T>::is_integer);

    if (a < b)
        throw IEX_NAMESPACE::UnderflowExc ("Integer subtraction underflow.");

    return a - b;
}


template <class T>
inline size_t
checkArraySize (T n, size_t s)
{
    //
    // Verify that the size, in bytes, of an array with n elements
    // of size s can be computed without overflowing:
    //
    // If computing
    //
    //      size_t (n) * s
    //
    // would overflow, then throw an IEX_NAMESPACE::OverflowExc exception.
    // Otherwise return
    //
    //      size_t (n).
    //

    IMF_STATIC_ASSERT (!std::numeric_limits<T>::is_signed &&
                        std::numeric_limits<T>::is_integer);

    IMF_STATIC_ASSERT (sizeof (T) <= sizeof (size_t));

    if (size_t (n) > std::numeric_limits<size_t>::max() / s)
        throw IEX_NAMESPACE::OverflowExc ("Integer multiplication overflow.");

    return size_t (n);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif
