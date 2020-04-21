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


//-----------------------------------------------------------------------------
//
//	Routines for converting between pixel data types,
//	with well-defined behavior for exceptional cases.
//
//-----------------------------------------------------------------------------

#include "ImfConvert.h"
#include "ImfNamespace.h"

#include <limits.h>


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

namespace {

inline bool
isNegative (float f)
{
    union {float f; int i;} u;
    u.f = f;
    return (u.i & 0x80000000) != 0;
}


inline bool
isNan (float f)
{
    union {float f; int i;} u;
    u.f = f;
    return (u.i & 0x7fffffff) > 0x7f800000;
}


inline bool
isInfinity (float f)
{
    union {float f; int i;} u;
    u.f = f;
    return (u.i & 0x7fffffff) == 0x7f800000;
}


inline bool
isFinite (float f)
{
    union {float f; int i;} u;
    u.f = f;
    return (u.i & 0x7f800000) != 0x7f800000;
}

} // namespace


unsigned int
halfToUint (half h)
{
    if (h.isNegative() || h.isNan())
	return 0;

    if (h.isInfinity())
	return UINT_MAX;

    return (unsigned int) h;
}


unsigned int
floatToUint (float f)
{
    if (isNegative (f) || isNan (f))
	return 0;

    if (isInfinity (f) || f > UINT_MAX)
	return UINT_MAX;

    return (unsigned int) f;
}


half	
uintToHalf (unsigned int ui)
{
    if (ui >  HALF_MAX)
	return half::posInf();

    return half ((float) ui);
}


half	
floatToHalf (float f)
{
    if (isFinite (f))
    {
	if (f >  HALF_MAX)
	    return half::posInf();

	if (f < -HALF_MAX)
	    return half::negInf();
    }

    return half (f);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
