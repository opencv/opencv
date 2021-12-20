//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	Routines for converting between pixel data types,
//	with well-defined behavior for exceptional cases.
//
//-----------------------------------------------------------------------------

#include "ImfConvert.h"
#include "ImfNamespace.h"

#include "halfLimits.h"
#include <limits>

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
	return std::numeric_limits <unsigned int>::max();

    return static_cast <unsigned int> (h);
}


unsigned int
floatToUint (float f)
{
    if (isNegative (f) || isNan (f))
	return 0;

    if (isInfinity (f) ||
        f > static_cast <float> (std::numeric_limits <unsigned int>::max()))
	return std::numeric_limits<unsigned int>::max();

    return static_cast <unsigned int> (f);
}


half	
uintToHalf (unsigned int ui)
{
    if (ui >  std::numeric_limits<half>::max())
	return half::posInf();

    return half ((float) ui);
}


half	
floatToHalf (float f)
{
    if (isFinite (f))
    {
	if (f >  std::numeric_limits<half>::max())
	    return half::posInf();

	if (f < std::numeric_limits<half>::lowest())
	    return half::negInf();
    }

    return half (f);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
