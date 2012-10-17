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


#ifndef INCLUDED_IMF_CONVERT_H
#define INCLUDED_IMF_CONVERT_H

//-----------------------------------------------------------------------------
//
//	Routines for converting between pixel data types,
//	with well-defined behavior for exceptional cases,
//	without depending on how hardware and operating
//	system handle integer overflows and floating-point
//	exceptions.
//
//-----------------------------------------------------------------------------

#include "half.h"


namespace Imf {

//---------------------------------------------------------
// Conversion from half or float to unsigned int:
//
//	input			result
//	---------------------------------------------------
//
//	finite, >= 0		input, cast to unsigned int
//				(rounds towards zero)
//
//	finite, < 0		0
//
//	NaN			0
//
//	+infinity		UINT_MAX
//
//	-infinity		0
//
//---------------------------------------------------------

unsigned int	halfToUint (half h);
unsigned int	floatToUint (float f);


//---------------------------------------------------------
// Conversion from unsigned int or float to half:
//
// 	input			result
//	---------------------------------------------------
//
// 	finite,			closest possible half
// 	magnitude <= HALF_MAX
//
// 	finite, > HALF_MAX	+infinity
//
// 	finite, < -HALF_MAX	-infinity
//
// 	NaN			NaN
//
// 	+infinity		+infinity
//
// 	-infinity		-infinity
//
//---------------------------------------------------------

half		uintToHalf (unsigned int ui);
half		floatToHalf (float f);


} // namespace Imf

#endif
