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
 * $Id: jas_fix.h,v 1.2 2008-05-26 09:41:51 vp153 Exp $
 */

#ifndef JAS_FIX_H
#define JAS_FIX_H

/******************************************************************************\
* Includes.
\******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <jasper/jas_config.h>
#include <jasper/jas_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************\
* Constants.
\******************************************************************************/

/* The representation of the value zero. */
#define	JAS_FIX_ZERO(fix_t, fracbits) \
	JAS_CAST(fix_t, 0)

/* The representation of the value one. */
#define	JAS_FIX_ONE(fix_t, fracbits) \
	(JAS_CAST(fix_t, 1) << (fracbits))

/* The representation of the value one half. */
#define	JAS_FIX_HALF(fix_t, fracbits) \
	(JAS_CAST(fix_t, 1) << ((fracbits) - 1))

/******************************************************************************\
* Conversion operations.
\******************************************************************************/

/* Convert an int to a fixed-point number. */
#define JAS_INTTOFIX(fix_t, fracbits, x) \
	JAS_CAST(fix_t, (x) << (fracbits))

/* Convert a fixed-point number to an int. */
#define JAS_FIXTOINT(fix_t, fracbits, x) \
	JAS_CAST(int, (x) >> (fracbits))

/* Convert a fixed-point number to a double. */
#define JAS_FIXTODBL(fix_t, fracbits, x) \
	(JAS_CAST(double, x) / (JAS_CAST(fix_t, 1) << (fracbits)))

/* Convert a double to a fixed-point number. */
#define JAS_DBLTOFIX(fix_t, fracbits, x) \
	JAS_CAST(fix_t, ((x) * JAS_CAST(double, JAS_CAST(fix_t, 1) << (fracbits))))

/******************************************************************************\
* Basic arithmetic operations.
* All other arithmetic operations are synthesized from these basic operations.
* There are three macros for each type of arithmetic operation.
* One macro always performs overflow/underflow checking, one never performs
* overflow/underflow checking, and one is generic with its behavior
* depending on compile-time flags.
* Only the generic macros should be invoked directly by application code.
\******************************************************************************/

/* Calculate the sum of two fixed-point numbers. */
#if !defined(DEBUG_OVERFLOW)
#define JAS_FIX_ADD			JAS_FIX_ADD_FAST
#else
#define JAS_FIX_ADD			JAS_FIX_ADD_OFLOW
#endif

/* Calculate the sum of two fixed-point numbers without overflow checking. */
#define	JAS_FIX_ADD_FAST(fix_t, fracbits, x, y)	((x) + (y))

/* Calculate the sum of two fixed-point numbers with overflow checking. */
#define	JAS_FIX_ADD_OFLOW(fix_t, fracbits, x, y) \
	((x) >= 0) ? \
	  (((y) >= 0) ? ((x) + (y) >= 0 || JAS_FIX_OFLOW(), (x) + (y)) : \
	  ((x) + (y))) : \
	  (((y) >= 0) ? ((x) + (y)) : ((x) + (y) < 0 || JAS_FIX_OFLOW(), \
	  (x) + (y)))

/* Calculate the product of two fixed-point numbers. */
#if !defined(DEBUG_OVERFLOW)
#define JAS_FIX_MUL			JAS_FIX_MUL_FAST
#else
#define JAS_FIX_MUL			JAS_FIX_MUL_OFLOW
#endif

/* Calculate the product of two fixed-point numbers without overflow
  checking. */
#define	JAS_FIX_MUL_FAST(fix_t, fracbits, bigfix_t, x, y) \
	JAS_CAST(fix_t, (JAS_CAST(bigfix_t, x) * JAS_CAST(bigfix_t, y)) >> \
	  (fracbits))

/* Calculate the product of two fixed-point numbers with overflow
  checking. */
#define JAS_FIX_MUL_OFLOW(fix_t, fracbits, bigfix_t, x, y) \
	((JAS_CAST(bigfix_t, x) * JAS_CAST(bigfix_t, y) >> (fracbits)) == \
	  JAS_CAST(fix_t, (JAS_CAST(bigfix_t, x) * JAS_CAST(bigfix_t, y) >> \
	  (fracbits))) ? \
	  JAS_CAST(fix_t, (JAS_CAST(bigfix_t, x) * JAS_CAST(bigfix_t, y) >> \
	  (fracbits))) : JAS_FIX_OFLOW())

/* Calculate the product of a fixed-point number and an int. */
#if !defined(DEBUG_OVERFLOW)
#define	JAS_FIX_MULBYINT	JAS_FIX_MULBYINT_FAST
#else
#define	JAS_FIX_MULBYINT	JAS_FIX_MULBYINT_OFLOW
#endif

/* Calculate the product of a fixed-point number and an int without overflow
  checking. */
#define	JAS_FIX_MULBYINT_FAST(fix_t, fracbits, x, y) \
	JAS_CAST(fix_t, ((x) * (y)))

/* Calculate the product of a fixed-point number and an int with overflow
  checking. */
#define	JAS_FIX_MULBYINT_OFLOW(fix_t, fracbits, x, y) \
	JAS_FIX_MULBYINT_FAST(fix_t, fracbits, x, y)

/* Calculate the quotient of two fixed-point numbers. */
#if !defined(DEBUG_OVERFLOW)
#define JAS_FIX_DIV			JAS_FIX_DIV_FAST
#else
#define JAS_FIX_DIV			JAS_FIX_DIV_UFLOW
#endif

/* Calculate the quotient of two fixed-point numbers without underflow
  checking. */
#define	JAS_FIX_DIV_FAST(fix_t, fracbits, bigfix_t, x, y) \
	JAS_CAST(fix_t, (JAS_CAST(bigfix_t, x) << (fracbits)) / (y))

/* Calculate the quotient of two fixed-point numbers with underflow
  checking. */
#define JAS_FIX_DIV_UFLOW(fix_t, fracbits, bigfix_t, x, y) \
	JAS_FIX_DIV_FAST(fix_t, fracbits, bigfix_t, x, y)

/* Negate a fixed-point number. */
#if !defined(DEBUG_OVERFLOW)
#define	JAS_FIX_NEG			JAS_FIX_NEG_FAST
#else
#define	JAS_FIX_NEG			JAS_FIX_NEG_OFLOW
#endif

/* Negate a fixed-point number without overflow checking. */
#define	JAS_FIX_NEG_FAST(fix_t, fracbits, x) \
	(-(x))

/* Negate a fixed-point number with overflow checking. */
/* Yes, overflow is actually possible for two's complement representations,
  although highly unlikely to occur. */
#define	JAS_FIX_NEG_OFLOW(fix_t, fracbits, x) \
	(((x) < 0) ? (-(x) > 0 || JAS_FIX_OFLOW(), -(x)) : (-(x)))

/* Perform an arithmetic shift left of a fixed-point number. */
#if !defined(DEBUG_OVERFLOW)
#define	JAS_FIX_ASL			JAS_FIX_ASL_FAST
#else
#define	JAS_FIX_ASL			JAS_FIX_ASL_OFLOW
#endif

/* Perform an arithmetic shift left of a fixed-point number without overflow
  checking. */
#define	JAS_FIX_ASL_FAST(fix_t, fracbits, x, n) \
	((x) << (n))

/* Perform an arithmetic shift left of a fixed-point number with overflow
  checking. */
#define	JAS_FIX_ASL_OFLOW(fix_t, fracbits, x, n) \
	((((x) << (n)) >> (n)) == (x) || JAS_FIX_OFLOW(), (x) << (n))

/* Perform an arithmetic shift right of a fixed-point number. */
#if !defined(DEBUG_OVERFLOW)
#define	JAS_FIX_ASR			JAS_FIX_ASR_FAST
#else
#define	JAS_FIX_ASR			JAS_FIX_ASR_UFLOW
#endif

/* Perform an arithmetic shift right of a fixed-point number without underflow
  checking. */
#define	JAS_FIX_ASR_FAST(fix_t, fracbits, x, n) \
	((x) >> (n))

/* Perform an arithmetic shift right of a fixed-point number with underflow
  checking. */
#define	JAS_FIX_ASR_UFLOW(fix_t, fracbits, x, n) \
	JAS_FIX_ASR_FAST(fix_t, fracbits, x, n)

/******************************************************************************\
* Other basic arithmetic operations.
\******************************************************************************/

/* Calculate the difference between two fixed-point numbers. */
#define JAS_FIX_SUB(fix_t, fracbits, x, y) \
	JAS_FIX_ADD(fix_t, fracbits, x, JAS_FIX_NEG(fix_t, fracbits, y))

/* Add one fixed-point number to another. */
#define JAS_FIX_PLUSEQ(fix_t, fracbits, x, y) \
	((x) = JAS_FIX_ADD(fix_t, fracbits, x, y))

/* Subtract one fixed-point number from another. */
#define JAS_FIX_MINUSEQ(fix_t, fracbits, x, y) \
	((x) = JAS_FIX_SUB(fix_t, fracbits, x, y))

/* Multiply one fixed-point number by another. */
#define	JAS_FIX_MULEQ(fix_t, fracbits, bigfix_t, x, y) \
	((x) = JAS_FIX_MUL(fix_t, fracbits, bigfix_t, x, y))

/******************************************************************************\
* Miscellaneous operations.
\******************************************************************************/

/* Calculate the absolute value of a fixed-point number. */
#define	JAS_FIX_ABS(fix_t, fracbits, x) \
	(((x) >= 0) ? (x) : (JAS_FIX_NEG(fix_t, fracbits, x)))

/* Is a fixed-point number an integer? */
#define	JAS_FIX_ISINT(fix_t, fracbits, x) \
	(JAS_FIX_FLOOR(fix_t, fracbits, x) == (x))

/* Get the sign of a fixed-point number. */
#define JAS_FIX_SGN(fix_t, fracbits, x) \
	((x) >= 0 ? 1 : (-1))

/******************************************************************************\
* Relational operations.
\******************************************************************************/

/* Compare two fixed-point numbers. */
#define JAS_FIX_CMP(fix_t, fracbits, x, y) \
	((x) > (y) ? 1 : (((x) == (y)) ? 0 : (-1)))

/* Less than. */
#define	JAS_FIX_LT(fix_t, fracbits, x, y) \
	((x) < (y))

/* Less than or equal. */
#define	JAS_FIX_LTE(fix_t, fracbits, x, y) \
	((x) <= (y))

/* Greater than. */
#define	JAS_FIX_GT(fix_t, fracbits, x, y) \
	((x) > (y))

/* Greater than or equal. */
#define	JAS_FIX_GTE(fix_t, fracbits, x, y) \
	((x) >= (y))

/******************************************************************************\
* Rounding functions.
\******************************************************************************/

/* Round a fixed-point number to the nearest integer. */
#define	JAS_FIX_ROUND(fix_t, fracbits, x) \
	(((x) < 0) ? JAS_FIX_FLOOR(fix_t, fracbits, JAS_FIX_ADD(fix_t, fracbits, \
	  (x), JAS_FIX_HALF(fix_t, fracbits))) : \
	  JAS_FIX_NEG(fix_t, fracbits, JAS_FIX_FLOOR(fix_t, fracbits, \
	  JAS_FIX_ADD(fix_t, fracbits, (-(x)), JAS_FIX_HALF(fix_t, fracbits)))))

/* Round a fixed-point number to the nearest integer in the direction of
  negative infinity (i.e., the floor function). */
#define	JAS_FIX_FLOOR(fix_t, fracbits, x) \
	((x) & (~((JAS_CAST(fix_t, 1) << (fracbits)) - 1)))

/* Round a fixed-point number to the nearest integer in the direction
  of zero. */
#define JAS_FIX_TRUNC(fix_t, fracbits, x) \
	(((x) >= 0) ? JAS_FIX_FLOOR(fix_t, fracbits, x) : \
	  JAS_FIX_CEIL(fix_t, fracbits, x))

/******************************************************************************\
* The below macros are for internal library use only.  Do not invoke them
* directly in application code.
\******************************************************************************/

/* Handle overflow. */
#define	JAS_FIX_OFLOW() \
	jas_eprintf("overflow error: file %s, line %d\n", __FILE__, __LINE__)

/* Handle underflow. */
#define	JAS_FIX_UFLOW() \
	jas_eprintf("underflow error: file %s, line %d\n", __FILE__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif
