///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2009, Industrial Light & Magic, a division of Lucas
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

#ifndef INCLUDED_IMF_CHECKED_ARITHMETIC_H
#define INCLUDED_IMF_CHECKED_ARITHMETIC_H

//-----------------------------------------------------------------------------
//
//	Integer arithmetic operations that throw exceptions
//      on overflow, underflow or division by zero.
//
//-----------------------------------------------------------------------------

#include <limits>
#include <IexMathExc.h>

namespace Imf {

template <bool b> struct StaticAssertionFailed;
template <> struct StaticAssertionFailed <true> {};

#define IMF_STATIC_ASSERT(x) \
    do {StaticAssertionFailed <x> staticAssertionFailed;} while (false)


template <class T>
T
uiMult (T a, T b)
{
    //
    // Unsigned integer multiplication
    //

    IMF_STATIC_ASSERT (!std::numeric_limits<T>::is_signed &&
                        std::numeric_limits<T>::is_integer);

    if (a > 0 && b > std::numeric_limits<T>::max() / a)
        throw Iex::OverflowExc ("Integer multiplication overflow.");

    return a * b;
}


template <class T>
T
uiDiv (T a, T b)
{
    //
    // Unsigned integer division
    //

    IMF_STATIC_ASSERT (!std::numeric_limits<T>::is_signed &&
                        std::numeric_limits<T>::is_integer);

    if (b == 0)
        throw Iex::DivzeroExc ("Integer division by zero.");

    return a / b;
}


template <class T>
T
uiAdd (T a, T b)
{
    //
    // Unsigned integer addition
    //

    IMF_STATIC_ASSERT (!std::numeric_limits<T>::is_signed &&
                        std::numeric_limits<T>::is_integer);

    if (a > std::numeric_limits<T>::max() - b)
        throw Iex::OverflowExc ("Integer addition overflow.");

    return a + b;
}


template <class T>
T
uiSub (T a, T b)
{
    //
    // Unsigned integer subtraction
    //

    IMF_STATIC_ASSERT (!std::numeric_limits<T>::is_signed &&
                        std::numeric_limits<T>::is_integer);

    if (a < b)
        throw Iex::UnderflowExc ("Integer subtraction underflow.");

    return a - b;
}


template <class T>
size_t
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
    // would overflow, then throw an Iex::OverflowExc exception.
    // Otherwise return
    //
    //      size_t (n).
    //

    IMF_STATIC_ASSERT (!std::numeric_limits<T>::is_signed &&
                        std::numeric_limits<T>::is_integer);

    IMF_STATIC_ASSERT (sizeof (T) <= sizeof (size_t));

    if (size_t (n) > std::numeric_limits<size_t>::max() / s)
        throw Iex::OverflowExc ("Integer multiplication overflow.");

    return size_t (n);
}


} // namespace Imf

#endif
