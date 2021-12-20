//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//---------------------------------------------------------------------------
//
//	toFloat
//
//	A program to generate the lookup table for half-to-float
//	conversion needed by class half.
//	The program loops over all 65536 possible half numbers,
//	converts each of them to a float, and prints the result.
//
//---------------------------------------------------------------------------

#include <iomanip>
#include <iostream>

using namespace std;

//---------------------------------------------------
// Interpret an unsigned short bit pattern as a half,
// and convert that half to the corresponding float's
// bit pattern.
//---------------------------------------------------

unsigned int
halfToFloat (unsigned short y)
{

    int s = (y >> 15) & 0x00000001;
    int e = (y >> 10) & 0x0000001f;
    int m = y & 0x000003ff;

    if (e == 0)
    {
        if (m == 0)
        {
            //
            // Plus or minus zero
            //

            return s << 31;
        }
        else
        {
            //
            // Denormalized number -- renormalize it
            //

            while (!(m & 0x00000400))
            {
                m <<= 1;
                e -= 1;
            }

            e += 1;
            m &= ~0x00000400;
        }
    }
    else if (e == 31)
    {
        if (m == 0)
        {
            //
            // Positive or negative infinity
            //

            return (s << 31) | 0x7f800000;
        }
        else
        {
            //
            // Nan -- preserve sign and significand bits
            //

            return (s << 31) | 0x7f800000 | (m << 13);
        }
    }

    //
    // Normalized number
    //

    e = e + (127 - 15);
    m = m << 13;

    //
    // Assemble s, e and m.
    //

    return (s << 31) | (e << 23) | m;
}

//---------------------------------------------
// Main - prints the half-to-float lookup table
//---------------------------------------------

int
main()
{
    cout.precision (9);
    cout.setf (ios_base::hex, ios_base::basefield);

    cout << "//\n"
            "// This is an automatically generated file.\n"
            "// Do not edit.\n"
            "//\n\n";

    cout << "{\n    ";

    const int iMax = (1 << 16);

    for (int i = 0; i < iMax; i++)
    {
        cout << "{0x" << setfill ('0') << setw (8) << halfToFloat (i) << "}, ";

        if (i % 4 == 3)
        {
            cout << "\n";

            if (i < iMax - 1)
                cout << "    ";
        }
    }

    cout << "};\n";
    return 0;
}
