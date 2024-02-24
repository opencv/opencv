//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

#include "ImathFun.h"

IMATH_INTERNAL_NAMESPACE_SOURCE_ENTER

float
succf (float f) IMATH_NOEXCEPT
{
    union
    {
        float f;
        uint32_t i;
    } u;
    u.f = f;

    if (isinf(f) || isnan (f))
    {
        // Nan or infinity; don't change value.
    }
    else if (u.i == 0x00000000 || u.i == 0x80000000)
    {
        // Plus or minus zero.

        u.i = 0x00000001;
    }
    else if (u.f > 0)
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
predf (float f) IMATH_NOEXCEPT
{
    union
    {
        float f;
        uint32_t i;
    } u;
    u.f = f;

    if (isinf(f) || isnan (f))
    {
        // Nan or infinity; don't change value.
    }
    else if (u.i == 0x00000000 || u.i == 0x80000000)
    {
        // Plus or minus zero.

        u.i = 0x80000001;
    }
    else if (u.f > 0)
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
succd (double d) IMATH_NOEXCEPT
{
    union
    {
        double d;
        uint64_t i;
    } u;
    u.d = d;

    if (isinf(d) || isnan (d))
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
predd (double d) IMATH_NOEXCEPT
{
    union
    {
        double d;
        uint64_t i;
    } u;
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
