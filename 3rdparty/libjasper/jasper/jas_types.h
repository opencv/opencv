/*
 * Copyright (c) 1999-2000 Image Power, Inc. and the University of
 *   British Columbia.
 * Copyright (c) 2001-2003 Michael David Adams.
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
 * Primitive Types
 *
 * $Id: jas_types.h,v 1.2 2008-05-26 09:41:51 vp153 Exp $
 */

#ifndef JAS_TYPES_H
#define JAS_TYPES_H

#include <jasper/jas_config.h>

#if !defined(JAS_CONFIGURE)

#if defined(WIN32) || defined(HAVE_WINDOWS_H)
/*
   We are dealing with Microsoft Windows and most likely Microsoft
   Visual C (MSVC).  (Heaven help us.)  Sadly, MSVC does not correctly
   define some of the standard types specified in ISO/IEC 9899:1999.
   In particular, it does not define the "long long" and "unsigned long
   long" types.  So, we work around this problem by using the "INT64"
   and "UINT64" types that are defined in the header file "windows.h".
 */
#include <windows.h>
#undef longlong
#define	longlong	INT64
#undef ulonglong
#define	ulonglong	UINT64
#endif

#endif

#if defined(HAVE_STDLIB_H)
#undef false
#undef true
#include <stdlib.h>
#endif
#if defined(HAVE_STDDEF_H)
#include <stddef.h>
#endif
#if defined(HAVE_SYS_TYPES_H)
#include <sys/types.h>
#endif

#ifndef __cplusplus
#if defined(HAVE_STDBOOL_H)
/*
 * The C language implementation does correctly provide the standard header
 * file "stdbool.h".
 */
#include <stdbool.h>
#else

/*
 * The C language implementation does not provide the standard header file
 * "stdbool.h" as required by ISO/IEC 9899:1999.  Try to compensate for this
 * braindamage below.
 */
#if !defined(bool)
#define	bool	int
#endif
#if !defined(true)
#define true	1
#endif
#if !defined(false)
#define	false	0
#endif
#endif

#endif

#if defined(HAVE_STDINT_H)
/*
 * The C language implementation does correctly provide the standard header
 * file "stdint.h".
 */
#include <stdint.h>
#else
/*
 * The C language implementation does not provide the standard header file
 * "stdint.h" as required by ISO/IEC 9899:1999.  Try to compensate for this
 * braindamage below.
 */
#include <limits.h>
/**********/
#if !defined(INT_FAST8_MIN)
typedef signed char int_fast8_t;
#define INT_FAST8_MIN	(-127)
#define INT_FAST8_MAX	128
#endif
/**********/
#if !defined(UINT_FAST8_MAX)
typedef unsigned char uint_fast8_t;
#define UINT_FAST8_MAX	255
#endif
/**********/
#if !defined(INT_FAST16_MIN)
typedef short int_fast16_t;
#define INT_FAST16_MIN	SHRT_MIN
#define INT_FAST16_MAX	SHRT_MAX
#endif
/**********/
#if !defined(UINT_FAST16_MAX)
typedef unsigned short uint_fast16_t;
#define UINT_FAST16_MAX	USHRT_MAX
#endif
/**********/
#if !defined(INT_FAST32_MIN)
typedef int int_fast32_t;
#define INT_FAST32_MIN	INT_MIN
#define INT_FAST32_MAX	INT_MAX
#endif
/**********/
#if !defined(UINT_FAST32_MAX)
typedef unsigned int uint_fast32_t;
#define UINT_FAST32_MAX	UINT_MAX
#endif
/**********/
#if !defined(INT_FAST64_MIN)
typedef longlong int_fast64_t;
#define INT_FAST64_MIN	LLONG_MIN
#define INT_FAST64_MAX	LLONG_MAX
#endif
/**********/
#if !defined(UINT_FAST64_MAX)
typedef ulonglong uint_fast64_t;
#define UINT_FAST64_MAX	ULLONG_MAX
#endif
/**********/
#endif

/* Hopefully, these macro definitions will fix more problems than they cause. */
#if !defined(uchar)
#define uchar unsigned char
#endif
#if !defined(ushort)
#define ushort unsigned short
#endif
#if !defined(uint)
#define uint unsigned int
#endif
#if !defined(ulong)
#define ulong unsigned long
#endif
#if !defined(longlong)
#define longlong long long
#endif
#if !defined(ulonglong)
#define ulonglong unsigned long long
#endif

/* The below macro is intended to be used for type casts.  By using this
  macro, type casts can be easily located in the source code with
  tools like "grep". */
#define	JAS_CAST(t, e) \
    ((t) (e))

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

#endif
