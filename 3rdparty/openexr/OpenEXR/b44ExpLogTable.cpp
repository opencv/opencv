//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

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
