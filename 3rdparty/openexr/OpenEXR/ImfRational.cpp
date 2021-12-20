//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//-----------------------------------------------------------------------------
//
//	Rational numbers
//
//	The double-to-Rational conversion code below
//	was contributed to OpenEXR by Greg Ward.
//
//-----------------------------------------------------------------------------

#include <ImfRational.h>
#include <cmath>

using namespace std;
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

namespace {

double
frac (double x, double e)
{
    return x - floor (x + e);
}


double
square (double x)
{
    return x * x;
}


double
denom (double x, double e)
{
    if (e > frac (x, e))
    {
        return 1;
    }
    else
    {
	double r = frac (1 / x, e);
	
        if (e > r)
        {
            return floor (1 / x + e);
        }
        else
        {
            return denom (frac (1 / r, e), e / square (x * r)) +
                   floor (1 / x + e) * denom (frac (1 / x, e), e / square (x));
        }
    }
}

} // namespace


Rational::Rational (double x)
{
    int sign;

    if (x >= 0)
    {
	sign = 1;	// positive
    }
    else if (x < 0)
    {
	sign = -1;	// negative
	x = -x;
    }
    else
    {
	n = 0;		// NaN
	d = 0;
	return;
    }

    if (x >= (1U << 31) - 0.5)
    {
	n = sign;	// infinity
	d = 0;
	return;
    }

    double e = (x < 1? 1: x) / (1U << 30);
    d = (unsigned int) denom (x, e);
    n = sign * (int) floor (x * d + 0.5);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
