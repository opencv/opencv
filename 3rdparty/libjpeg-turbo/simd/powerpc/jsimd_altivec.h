/*
 * AltiVec optimizations for libjpeg-turbo
 *
 * Copyright (C) 2014-2015, 2024-2025, D. R. Commander.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#include "../jsimdint.h"
#include <altivec.h>


/* Common code */

#define __4X(a)      a, a, a, a
#define __4X2(a, b)  a, b, a, b, a, b, a, b
#define __8X(a)      __4X(a), __4X(a)
#define __16X(a)     __8X(a), __8X(a)

#define TRANSPOSE(row, col) { \
  __vector short r04l, r04h, r15l, r15h, r26l, r26h, r37l, r37h; \
  __vector short c01e, c01o, c23e, c23o, c45e, c45o, c67e, c67o; \
  \
                                      /* transpose coefficients (phase 1) */ \
  r04l = vec_mergeh(row##0, row##4);  /* r04l = (00 40 01 41 02 42 03 43) */ \
  r04h = vec_mergel(row##0, row##4);  /* r04h = (04 44 05 45 06 46 07 47) */ \
  r15l = vec_mergeh(row##1, row##5);  /* r15l = (10 50 11 51 12 52 13 53) */ \
  r15h = vec_mergel(row##1, row##5);  /* r15h = (14 54 15 55 16 56 17 57) */ \
  r26l = vec_mergeh(row##2, row##6);  /* r26l = (20 60 21 61 22 62 23 63) */ \
  r26h = vec_mergel(row##2, row##6);  /* r26h = (24 64 25 65 26 66 27 67) */ \
  r37l = vec_mergeh(row##3, row##7);  /* r37l = (30 70 31 71 32 72 33 73) */ \
  r37h = vec_mergel(row##3, row##7);  /* r37h = (34 74 35 75 36 76 37 77) */ \
  \
                                      /* transpose coefficients (phase 2) */ \
  c01e = vec_mergeh(r04l, r26l);      /* c01e = (00 20 40 60 01 21 41 61) */ \
  c23e = vec_mergel(r04l, r26l);      /* c23e = (02 22 42 62 03 23 43 63) */ \
  c45e = vec_mergeh(r04h, r26h);      /* c45e = (04 24 44 64 05 25 45 65) */ \
  c67e = vec_mergel(r04h, r26h);      /* c67e = (06 26 46 66 07 27 47 67) */ \
  c01o = vec_mergeh(r15l, r37l);      /* c01o = (10 30 50 70 11 31 51 71) */ \
  c23o = vec_mergel(r15l, r37l);      /* c23o = (12 32 52 72 13 33 53 73) */ \
  c45o = vec_mergeh(r15h, r37h);      /* c45o = (14 34 54 74 15 35 55 75) */ \
  c67o = vec_mergel(r15h, r37h);      /* c67o = (16 36 56 76 17 37 57 77) */ \
  \
                                      /* transpose coefficients (phase 3) */ \
  col##0 = vec_mergeh(c01e, c01o);    /* col0 = (00 10 20 30 40 50 60 70) */ \
  col##1 = vec_mergel(c01e, c01o);    /* col1 = (01 11 21 31 41 51 61 71) */ \
  col##2 = vec_mergeh(c23e, c23o);    /* col2 = (02 12 22 32 42 52 62 72) */ \
  col##3 = vec_mergel(c23e, c23o);    /* col3 = (03 13 23 33 43 53 63 73) */ \
  col##4 = vec_mergeh(c45e, c45o);    /* col4 = (04 14 24 34 44 54 64 74) */ \
  col##5 = vec_mergel(c45e, c45o);    /* col5 = (05 15 25 35 45 55 65 75) */ \
  col##6 = vec_mergeh(c67e, c67o);    /* col6 = (06 16 26 36 46 56 66 76) */ \
  col##7 = vec_mergel(c67e, c67o);    /* col7 = (07 17 27 37 47 57 67 77) */ \
}

#ifndef min
#define min(a, b)  ((a) < (b) ? (a) : (b))
#endif


/* Macros to abstract big/little endian bit twiddling */

#ifdef __BIG_ENDIAN__

#define VEC_LD(a, b)     vec_ld(a, b)
#define VEC_ST(a, b, c)  vec_st(a, b, c)
#define VEC_UNPACKHU(a)  vec_mergeh(pb_zero, a)
#define VEC_UNPACKLU(a)  vec_mergel(pb_zero, a)

#else

#define VEC_LD(a, b)     vec_vsx_ld(a, b)
#define VEC_ST(a, b, c)  vec_vsx_st(a, b, c)
#define VEC_UNPACKHU(a)  vec_mergeh(a, pb_zero)
#define VEC_UNPACKLU(a)  vec_mergel(a, pb_zero)

#endif


#if defined(__clang__)
#pragma clang diagnostic ignored "-Wc99-extensions"
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
