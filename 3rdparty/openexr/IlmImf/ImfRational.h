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


#ifndef INCLUDED_IMF_RATIONAL_H
#define INCLUDED_IMF_RATIONAL_H

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

namespace Imf {

class Rational
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

    explicit Rational (double x);


    //---------------------------------
    // Approximate conversion to double
    //---------------------------------

    operator double () const {return double (n) / double (d);}
};

} // namespace Imf

#endif
