/*
 * Copyright (c) 2005, Hervé Drolon, FreeImage Team
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef OPJ_INCLUDES_H
#define OPJ_INCLUDES_H

/*
 ==========================================================
   Standard includes used by the library
 ==========================================================
*/
#include <memory.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include <assert.h>

/*
 ==========================================================
   OpenJPEG interface
 ==========================================================
 */

/*
 ==========================================================
   OpenJPEG modules
 ==========================================================
*/

/* Ignore GCC attributes if this is not GCC */
#ifndef __GNUC__
  #define __attribute__(x) /* __attribute__(x) */
#endif

/*
The inline keyword is supported by C99 but not by C90.
Most compilers implement their own version of this keyword ...
*/
#ifndef INLINE
  #if defined(_MSC_VER)
    #define INLINE __inline
  #elif defined(__GNUC__)
    #define INLINE __inline__
  #elif defined(__MWERKS__)
    #define INLINE inline
  #else
    /* add other compilers here ... */
    #define INLINE
  #endif /* defined(<Compiler>) */
#endif /* INLINE */

/* Are restricted pointers available? (C99) */
#if (__STDC_VERSION__ != 199901L)
  /* Not a C99 compiler */
  #ifdef __GNUC__
    #define restrict __restrict__
  #else
    #define restrict /* restrict */
  #endif
#endif

/* MSVC and Borland C do not have lrintf */
#if defined(_MSC_VER) || defined(__BORLANDC__)

/* MSVC 64bits doesn't support _asm */
#if !defined(_WIN64)
static INLINE long lrintf(float f){
  int i;

  _asm{
    fld f
    fistp i
  };

  return i;
}
#else
static INLINE long lrintf(float x){
  long r;
  if (x>=0.f)
  {
     x+=0.5f;
  }
  else
  {
     x-=0.5f;
  }
  r = (long)(x);
  if ( x != (float)(r) ) return r;
  return 2*(r/2);
}
#endif

#endif

#endif /* OPJ_INCLUDES_H */
