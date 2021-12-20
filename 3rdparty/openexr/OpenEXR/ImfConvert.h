//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

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

#include "ImfExport.h"
#include "ImfNamespace.h"

#include <half.h>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER


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

IMF_EXPORT unsigned int	halfToUint (half h);
IMF_EXPORT unsigned int	floatToUint (float f);


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

IMF_EXPORT half		uintToHalf (unsigned int ui);
IMF_EXPORT half		floatToHalf (float f);


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif
