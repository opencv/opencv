/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2002 Michael David Adams.
 * All rights reserved.
 */

/* __START_OF_JASPER_LICENSE__
 * 
 * JasPer License Version 2.0
 * 
 * Copyright (c) 2001-2006 Michael David Adams
 * Copyright (c) 1999-2000 Image Power, Inc.
 * Copyright (c) 1999-2000 The University of British Columbia
 * 
 * All rights reserved.
 * 
 * Permission is hereby granted, free of charge, to any person (the
 * "User") obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 * 
 * 1.  The above copyright notices and this permission notice (which
 * includes the disclaimer below) shall be included in all copies or
 * substantial portions of the Software.
 * 
 * 2.  The name of a copyright holder shall not be used to endorse or
 * promote products derived from the Software without specific prior
 * written permission.
 * 
 * THIS DISCLAIMER OF WARRANTY CONSTITUTES AN ESSENTIAL PART OF THIS
 * LICENSE.  NO USE OF THE SOFTWARE IS AUTHORIZED HEREUNDER EXCEPT UNDER
 * THIS DISCLAIMER.  THE SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS
 * "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
 * PARTICULAR PURPOSE AND NONINFRINGEMENT OF THIRD PARTY RIGHTS.  IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, OR ANY SPECIAL
 * INDIRECT OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING
 * FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
 * NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
 * WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.  NO ASSURANCES ARE
 * PROVIDED BY THE COPYRIGHT HOLDERS THAT THE SOFTWARE DOES NOT INFRINGE
 * THE PATENT OR OTHER INTELLECTUAL PROPERTY RIGHTS OF ANY OTHER ENTITY.
 * EACH COPYRIGHT HOLDER DISCLAIMS ANY LIABILITY TO THE USER FOR CLAIMS
 * BROUGHT BY ANY OTHER ENTITY BASED ON INFRINGEMENT OF INTELLECTUAL
 * PROPERTY RIGHTS OR OTHERWISE.  AS A CONDITION TO EXERCISING THE RIGHTS
 * GRANTED HEREUNDER, EACH USER HEREBY ASSUMES SOLE RESPONSIBILITY TO SECURE
 * ANY OTHER INTELLECTUAL PROPERTY RIGHTS NEEDED, IF ANY.  THE SOFTWARE
 * IS NOT FAULT-TOLERANT AND IS NOT INTENDED FOR USE IN MISSION-CRITICAL
 * SYSTEMS, SUCH AS THOSE USED IN THE OPERATION OF NUCLEAR FACILITIES,
 * AIRCRAFT NAVIGATION OR COMMUNICATION SYSTEMS, AIR TRAFFIC CONTROL
 * SYSTEMS, DIRECT LIFE SUPPORT MACHINES, OR WEAPONS SYSTEMS, IN WHICH
 * THE FAILURE OF THE SOFTWARE OR SYSTEM COULD LEAD DIRECTLY TO DEATH,
 * PERSONAL INJURY, OR SEVERE PHYSICAL OR ENVIRONMENTAL DAMAGE ("HIGH
 * RISK ACTIVITIES").  THE COPYRIGHT HOLDERS SPECIFICALLY DISCLAIM ANY
 * EXPRESS OR IMPLIED WARRANTY OF FITNESS FOR HIGH RISK ACTIVITIES.
 * 
 * __END_OF_JASPER_LICENSE__
 */

/*
 * Fixed-Point Number Class
 *
 * $Id: jpc_fix.h,v 1.2 2008-05-26 09:40:52 vp153 Exp $
 */

#ifndef JPC_FIX_H
#define JPC_FIX_H

/******************************************************************************\
* Includes.
\******************************************************************************/

#include "jasper/jas_types.h"
#include "jasper/jas_fix.h"

/******************************************************************************\
* Basic parameters of the fixed-point type.
\******************************************************************************/

/* The integral type used to represent a fixed-point number.  This
  type must be capable of representing values from -(2^31) to 2^31-1
  (inclusive). */
typedef int_fast32_t jpc_fix_t;

/* The integral type used to respresent higher-precision intermediate results.
  This type should be capable of representing values from -(2^63) to 2^63-1
  (inclusive). */
typedef int_fast64_t jpc_fix_big_t;

/* The number of bits used for the fractional part of a fixed-point number. */
#define JPC_FIX_FRACBITS	13

/******************************************************************************\
* Instantiations of the generic fixed-point number macros for the
* parameters given above.  (Too bad C does not support templates, eh?)
* The purpose of these macros is self-evident if one examines the
* corresponding macros in the jasper/jas_fix.h header file.
\******************************************************************************/

#define	JPC_FIX_ZERO	JAS_FIX_ZERO(jpc_fix_t, JPC_FIX_FRACBITS)
#define	JPC_FIX_ONE		JAS_FIX_ONE(jpc_fix_t, JPC_FIX_FRACBITS)
#define	JPC_FIX_HALF	JAS_FIX_HALF(jpc_fix_t, JPC_FIX_FRACBITS)

#define jpc_inttofix(x)	JAS_INTTOFIX(jpc_fix_t, JPC_FIX_FRACBITS, x)
#define jpc_fixtoint(x)	JAS_FIXTOINT(jpc_fix_t, JPC_FIX_FRACBITS, x)
#define jpc_fixtodbl(x)	JAS_FIXTODBL(jpc_fix_t, JPC_FIX_FRACBITS, x)
#define jpc_dbltofix(x)	JAS_DBLTOFIX(jpc_fix_t, JPC_FIX_FRACBITS, x)

#define	jpc_fix_add(x, y)	JAS_FIX_ADD(jpc_fix_t, JPC_FIX_FRACBITS, x, y)
#define	jpc_fix_sub(x, y)	JAS_FIX_SUB(jpc_fix_t, JPC_FIX_FRACBITS, x, y)
#define	jpc_fix_mul(x, y) \
	JAS_FIX_MUL(jpc_fix_t, JPC_FIX_FRACBITS, jpc_fix_big_t, x, y)
#define	jpc_fix_mulbyint(x, y) \
	JAS_FIX_MULBYINT(jpc_fix_t, JPC_FIX_FRACBITS, x, y)
#define	jpc_fix_div(x, y) \
	JAS_FIX_DIV(jpc_fix_t, JPC_FIX_FRACBITS, jpc_fix_big_t, x, y)
#define	jpc_fix_neg(x)		JAS_FIX_NEG(jpc_fix_t, JPC_FIX_FRACBITS, x)
#define	jpc_fix_asl(x, n)	JAS_FIX_ASL(jpc_fix_t, JPC_FIX_FRACBITS, x, n)
#define	jpc_fix_asr(x, n)	JAS_FIX_ASR(jpc_fix_t, JPC_FIX_FRACBITS, x, n)

#define jpc_fix_pluseq(x, y)	JAS_FIX_PLUSEQ(jpc_fix_t, JPC_FIX_FRACBITS, x, y)
#define jpc_fix_minuseq(x, y)	JAS_FIX_MINUSEQ(jpc_fix_t, JPC_FIX_FRACBITS, x, y)
#define	jpc_fix_muleq(x, y)	\
	JAS_FIX_MULEQ(jpc_fix_t, JPC_FIX_FRACBITS, jpc_fix_big_t, x, y)

#define	jpc_fix_abs(x)		JAS_FIX_ABS(jpc_fix_t, JPC_FIX_FRACBITS, x)
#define	jpc_fix_isint(x)	JAS_FIX_ISINT(jpc_fix_t, JPC_FIX_FRACBITS, x)
#define jpc_fix_sgn(x)		JAS_FIX_SGN(jpc_fix_t, JPC_FIX_FRACBITS, x)
#define	jpc_fix_round(x)	JAS_FIX_ROUND(jpc_fix_t, JPC_FIX_FRACBITS, x)
#define	jpc_fix_floor(x)	JAS_FIX_FLOOR(jpc_fix_t, JPC_FIX_FRACBITS, x)
#define jpc_fix_trunc(x)	JAS_FIX_TRUNC(jpc_fix_t, JPC_FIX_FRACBITS, x)

/******************************************************************************\
* Extra macros for convenience.
\******************************************************************************/

/* Compute the sum of three fixed-point numbers. */
#define jpc_fix_add3(x, y, z)	jpc_fix_add(jpc_fix_add(x, y), z)

#endif
