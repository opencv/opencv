///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2006, Industrial Light & Magic, a division of Lucas
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
//	b44ExpLogTable
//
//	A program to generate lookup tables for
//
//		y = exp (x / 8)
//
//	and
//		x = 8 * log (x);
//
//	where x and y are 16-bit floating-point numbers
//
//	The tables are used by class B44Compressor.
//
//---------------------------------------------------------------------------

#include <half.h>
#include <math.h>
#include <iostream>
#include <iomanip>

using namespace std;

//---------------------------------------------
// Main - prints the half-to-float lookup table
//---------------------------------------------

int
main ()
{
#ifndef HAVE_IOS_BASE
    cout.setf (ios::hex, ios::basefield);
#else
    cout.setf (ios_base::hex, ios_base::basefield);
#endif

    cout << "//\n"
	    "// This is an automatically generated file.\n"
	    "// Do not edit.\n"
	    "//\n\n";

    const int iMax = (1 << 16);

    cout << "const unsigned short expTable[] =\n"
	    "{\n"
	    "    ";

    for (int i = 0; i < iMax; i++)
    {
	half h;
	h.setBits (i);

	if (!h.isFinite())
	    h = 0;
	else if (h >= 8 * log (HALF_MAX))
	    h = HALF_MAX;
	else
	    h = exp (h / 8);

	cout << "0x" << setfill ('0') << setw (4) << h.bits() << ", ";

	if (i % 8 == 7)
	{
	    cout << "\n";

	    if (i < iMax - 1)
		cout << "    ";
	}
    }

    cout << "};\n\n";

    cout << "const unsigned short logTable[] =\n"
	    "{\n"
	    "    ";

    for (int i = 0; i < iMax; i++)
    {
	half h;
	h.setBits (i);

	if (!h.isFinite() || h < 0)
	    h = 0;
	else
	    h = 8 * log (h);

	cout << "0x" << setfill ('0') << setw (4) << h.bits() << ", ";

	if (i % 8 == 7)
	{
	    cout << "\n";

	    if (i < iMax - 1)
		cout << "    ";
	}
    }

    cout << "};\n";

    return 0;
}
