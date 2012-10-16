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



#include <iostream>
#include <iomanip>

using namespace std;

//-----------------------------------------------------
// Compute a lookup table for float-to-half conversion.
//
// When indexed with the combined sign and exponent of
// a float, the table either returns the combined sign
// and exponent of the corresponding half, or zero if
// the corresponding half may not be normalized (zero,
// denormalized, overflow).
//-----------------------------------------------------

void
initELut (unsigned short eLut[])
{
    for (int i = 0; i < 0x100; i++)
    {
    int e = (i & 0x0ff) - (127 - 15);

    if (e <= 0 || e >= 30)
    {
        //
        // Special case
        //

        eLut[i]         = 0;
        eLut[i | 0x100] = 0;
    }
    else
    {
        //
        // Common case - normalized half, no exponent overflow possible
        //

        eLut[i]         =  (e << 10);
        eLut[i | 0x100] = ((e << 10) | 0x8000);
    }
    }
}


//------------------------------------------------------------
// Main - prints the sign-and-exponent conversion lookup table
//------------------------------------------------------------

int
main ()
{
    const int tableSize = 1 << 9;
    unsigned short eLut[tableSize];
    initELut (eLut);

    cout << "//\n"
        "// This is an automatically generated file.\n"
        "// Do not edit.\n"
        "//\n\n";

    cout << "{\n    ";

    for (int i = 0; i < tableSize; i++)
    {
    cout << setw (5) << eLut[i] << ", ";

    if (i % 8 == 7)
    {
        cout << "\n";

        if (i < tableSize - 1)
        cout << "    ";
    }
    }

    cout << "};\n";
    return 0;
}
