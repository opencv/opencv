//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_RATIONAL_H
#define INCLUDED_IMF_RATIONAL_H

#include "ImfExport.h"
#include "ImfNamespace.h"

//-----------------------------------------------------------------------------
//
//	Rational numbers
//
//	A rational number is represented as pair of integers, n and d.
//	The value of of the rational number is
// 
//		n/d			for d > 0
//		positive infinity	for n > 0, d == 0
//		negative infinity	for n < 0, d == 0
//		not a number (NaN)	for n == 0, d == 0
//
//-----------------------------------------------------------------------------

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


class IMF_EXPORT_TYPE Rational
{
  public:

    int			n;		// numerator
    unsigned int	d;		// denominator


    //----------------------------------------
    // Default constructor, sets value to zero
    //----------------------------------------

    Rational (): n (0), d (1) {}


    //-------------------------------------
    // Constructor, explicitly sets n and d
    //-------------------------------------

    Rational (int n, int d): n (n), d (d) {}


    //----------------------------
    // Constructor, approximates x
    //----------------------------

    IMF_EXPORT
    explicit Rational (double x);


    //---------------------------------
    // Approximate conversion to double
    //---------------------------------

    operator double () const {return double (n) / double (d);}
};


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
