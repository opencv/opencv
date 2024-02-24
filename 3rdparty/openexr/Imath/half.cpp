//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// Primary original authors:
//     Florian Kainz <kainz@ilm.com>
//     Rod Bogart <rgb@ilm.com>
//

//---------------------------------------------------------------------------
//
//	class half --
//	implementation of non-inline members
//
//---------------------------------------------------------------------------

#include "half.h"
#include <assert.h>

using namespace std;

#if defined(IMATH_DLL)
#    define EXPORT_CONST __declspec(dllexport)
#else
#    define EXPORT_CONST
#endif

//-------------------------------------------------------------
// Lookup tables for half-to-float and float-to-half conversion
//-------------------------------------------------------------

// clang-format off

#if !defined(IMATH_HALF_NO_LOOKUP_TABLE)
// Omit the table entirely if IMATH_HALF_NO_LOOKUP_TABLE is
// defined. Half-to-float conversion must be accomplished either by
// F16C instructions or the bit-shift algorithm.
const imath_half_uif_t imath_half_to_float_table_data[1 << 16] =
#include "toFloat.h"

extern "C" {
EXPORT_CONST const imath_half_uif_t *imath_half_to_float_table = imath_half_to_float_table_data;
} // extern "C"

#endif

// clang-format on

//---------------------
// Stream I/O operators
//---------------------

IMATH_EXPORT ostream&
operator<< (ostream& os, half h)
{
    os << float (h);
    return os;
}

IMATH_EXPORT istream&
operator>> (istream& is, half& h)
{
    float f;
    is >> f;
    h = half (f);
    return is;
}

//---------------------------------------
// Functions to print the bit-layout of
// floats and halfs, mostly for debugging
//---------------------------------------

IMATH_EXPORT void
printBits (ostream& os, half h)
{
    unsigned short b = h.bits();

    for (int i = 15; i >= 0; i--)
    {
        os << (((b >> i) & 1) ? '1' : '0');

        if (i == 15 || i == 10)
            os << ' ';
    }
}

IMATH_EXPORT void
printBits (ostream& os, float f)
{
    half::uif x;
    x.f = f;

    for (int i = 31; i >= 0; i--)
    {
        os << (((x.i >> i) & 1) ? '1' : '0');

        if (i == 31 || i == 23)
            os << ' ';
    }
}

IMATH_EXPORT void
printBits (char c[19], half h)
{
    unsigned short b = h.bits();

    for (int i = 15, j = 0; i >= 0; i--, j++)
    {
        c[j] = (((b >> i) & 1) ? '1' : '0');

        if (i == 15 || i == 10)
            c[++j] = ' ';
    }

    c[18] = 0;
}

IMATH_EXPORT void
printBits (char c[35], float f)
{
    half::uif x;
    x.f = f;

    for (int i = 31, j = 0; i >= 0; i--, j++)
    {
        c[j] = (((x.i >> i) & 1) ? '1' : '0');

        if (i == 31 || i == 23)
            c[++j] = ' ';
    }

    c[34] = 0;
}
