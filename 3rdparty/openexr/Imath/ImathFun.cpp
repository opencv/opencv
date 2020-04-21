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


#include "ImathFun.h"

IMATH_INTERNAL_NAMESPACE_SOURCE_ENTER


float
succf (float f)
{
    union {float f; int i;} u;
    u.f = f;

    if ((u.i & 0x7f800000) == 0x7f800000)
    {
        // Nan or infinity; don't change value.
    }
    else if (u.i == 0x00000000 || u.i == 0x80000000)
    {
        // Plus or minus zero.

        u.i = 0x00000001;
    }
    else if (u.i > 0)
    {
        // Positive float, normalized or denormalized.
        // Incrementing the largest positive float
        // produces +infinity.

        ++u.i;
    }
    else
    {
        // Negative normalized or denormalized float.

        --u.i;
    }

    return u.f;
}


float
predf (float f)
{
    union {float f; int i;} u;
    u.f = f;

    if ((u.i & 0x7f800000) == 0x7f800000)
    {
        // Nan or infinity; don't change value.
    }
    else if (u.i == 0x00000000 || u.i == 0x80000000)
    {
        // Plus or minus zero.

        u.i = 0x80000001;
    }
    else if (u.i > 0)
    {
        // Positive float, normalized or denormalized.

        --u.i;
    }
    else
    {
        // Negative normalized or denormalized float.
        // Decrementing the largest negative float
        // produces -infinity.

        ++u.i;
    }

    return u.f;
}


double
succd (double d)
{
    union {double d; Int64 i;} u;
    u.d = d;

    if ((u.i & 0x7ff0000000000000LL) == 0x7ff0000000000000LL)
    {
        // Nan or infinity; don't change value.
    }
    else if (u.i == 0x0000000000000000LL || u.i == 0x8000000000000000LL)
    {
        // Plus or minus zero.

        u.i = 0x0000000000000001LL;
    }
    else if (u.d > 0)
    {
        // Positive double, normalized or denormalized.
        // Incrementing the largest positive double
        // produces +infinity.

        ++u.i;
    }
    else
    {
        // Negative normalized or denormalized double.

        --u.i;
    }

    return u.d;
}


double
predd (double d)
{
    union {double d; Int64 i;} u;
    u.d = d;

    if ((u.i & 0x7ff0000000000000LL) == 0x7ff0000000000000LL)
    {
        // Nan or infinity; don't change value.
    }
    else if (u.i == 0x0000000000000000LL || u.i == 0x8000000000000000LL)
    {
        // Plus or minus zero.

        u.i = 0x8000000000000001LL;
    }
    else if (u.d > 0)
    {
        // Positive double, normalized or denormalized.

        --u.i;
    }
    else
    {
        // Negative normalized or denormalized double.
        // Decrementing the largest negative double
        // produces -infinity.

        ++u.i;
    }

    return u.d;
}


IMATH_INTERNAL_NAMESPACE_SOURCE_EXIT
