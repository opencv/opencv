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


#include <iostream>
#include <iomanip>

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
    int m =  y        & 0x000003ff;

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
        e -=  1;
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
main ()
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
