/*
  Copyright 2008-2021 LibRaw LLC (info@libraw.org)

LibRaw is free software; you can redistribute it and/or modify
it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

   This file is generated from Dave Coffin's dcraw.c
   dcraw.c -- Dave Coffin's raw photo decoder
   Copyright 1997-2010 by Dave Coffin, dcoffin a cybercom o net

   Look into dcraw homepage (probably http://cybercom.net/~dcoffin/dcraw/)
   for more information
*/

#ifndef LIBRAW_INT_DEFINES_H
#define LIBRAW_INT_DEFINES_H
#ifndef USE_JPEG
#define NO_JPEG
#endif
#ifndef USE_JASPER
#define NO_JASPER
#endif
#define DCRAW_VERSION "9.26"

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define _USE_MATH_DEFINES
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/types.h>
#ifdef __CYGWIN__
#include <io.h>
#endif
#if defined LIBRAW_WIN32_CALLS
#include <sys/utime.h>
#ifndef LIBRAW_NO_WINSOCK2
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
#endif
#define snprintf _snprintf
#define strcasecmp stricmp
#define strncasecmp strnicmp
#else
#include <unistd.h>
#include <utime.h>
#include <netinet/in.h>
typedef long long INT64;
typedef unsigned long long UINT64;
#endif

#ifdef NODEPS
#define NO_JASPER
#define NO_JPEG
#define NO_LCMS
#endif
#ifndef NO_JASPER
#include <jasper/jasper.h> /* Decode Red camera movies */
#endif
#ifndef NO_JPEG
#include <jpeglib.h> /* Decode compressed Kodak DC120 photos */
#endif               /* and Adobe Lossy DNGs */
#ifndef NO_LCMS
#ifdef USE_LCMS
#include <lcms.h> /* Support color profiles */
#else
#include <lcms2.h> /* Support color profiles */
#endif
#endif
#ifdef LOCALEDIR
#include <libintl.h>
#define _(String) gettext(String)
#else
#define _(String) (String)
#endif

#ifdef LJPEG_DECODE
#error Please compile dcraw.c by itself.
#error Do not link it with ljpeg_decode.
#endif

#ifndef LONG_BIT
#define LONG_BIT (8 * sizeof(long))
#endif
#define FORC(cnt) for (c = 0; c < cnt; c++)
#define FORC3 FORC(3)
#define FORC4 FORC(4)
#define FORCC for (c = 0; c < colors && c < 4; c++)

#define SQR(x) ((x) * (x))
#define ABS(x) (((int)(x) ^ ((int)(x) >> 31)) - ((int)(x) >> 31))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define LIM(x, min, max) MAX(min, MIN(x, max))
#define ULIM(x, y, z) ((y) < (z) ? LIM(x, y, z) : LIM(x, z, y))
#define CLIP(x) LIM((int)(x), 0, 65535)
#define CLIP15(x) LIM((int)(x), 0, 32767)
#define SWAP(a, b)                                                             \
  {                                                                            \
    a = a + b;                                                                 \
    b = a - b;                                                                 \
    a = a - b;                                                                 \
  }

#define my_swap(type, i, j)                                                    \
  {                                                                            \
    type t = i;                                                                \
    i = j;                                                                     \
    j = t;                                                                     \
  }

#ifdef __GNUC__
inline
#elif defined(_MSC_VER)
__forceinline
#else
static
#endif
float fMAX(float a, float b) { return MAX(a, b); }

/*
   In order to inline this calculation, I make the risky
   assumption that all filter patterns can be described
   by a repeating pattern of eight rows and two columns

   Do not use the FC or BAYER macros with the Leaf CatchLight,
   because its pattern is 16x16, not 2x8.

   Return values are either 0/1/2/3 = G/M/C/Y or 0/1/2/3 = R/G1/B/G2

        PowerShot 600	PowerShot A50	PowerShot Pro70	Pro90 & G1
        0xe1e4e1e4:	0x1b4e4b1e:	0x1e4b4e1b:	0xb4b4b4b4:

          0 1 2 3 4 5	  0 1 2 3 4 5	  0 1 2 3 4 5	  0 1 2 3 4 5
        0 G M G M G M	0 C Y C Y C Y	0 Y C Y C Y C	0 G M G M G M
        1 C Y C Y C Y	1 M G M G M G	1 M G M G M G	1 Y C Y C Y C
        2 M G M G M G	2 Y C Y C Y C	2 C Y C Y C Y
        3 C Y C Y C Y	3 G M G M G M	3 G M G M G M
                        4 C Y C Y C Y	4 Y C Y C Y C
        PowerShot A5	5 G M G M G M	5 G M G M G M
        0x1e4e1e4e:	6 Y C Y C Y C	6 C Y C Y C Y
                        7 M G M G M G	7 M G M G M G
          0 1 2 3 4 5
        0 C Y C Y C Y
        1 G M G M G M
        2 C Y C Y C Y
        3 M G M G M G

   All RGB cameras use one of these Bayer grids:

        0x16161616:	0x61616161:	0x49494949:	0x94949494:

          0 1 2 3 4 5	  0 1 2 3 4 5	  0 1 2 3 4 5	  0 1 2 3 4 5
        0 B G B G B G	0 G R G R G R	0 G B G B G B	0 R G R G R G
        1 G R G R G R	1 B G B G B G	1 R G R G R G	1 G B G B G B
        2 B G B G B G	2 G R G R G R	2 G B G B G B	2 R G R G R G
        3 G R G R G R	3 B G B G B G	3 R G R G R G	3 G B G B G B
 */

// _RGBG means R, G1, B, G2 sequence
#define GRBG_2_RGBG(q)    (q ^ (q >> 1) ^ 1)
#define RGGB_2_RGBG(q)    (q ^ (q >> 1))
#define BG2RG1_2_RGBG(q)  (q ^ 2)
#define G2BRG1_2_RGBG(q)  (q ^ (q >> 1) ^ 3)
#define GRGB_2_RGBG(q)    (q ^ 1)
#define RBGG_2_RGBG(q)    ((q >> 1) | ((q & 1) << 1))

#define RAWINDEX(row, col) ((row)*raw_width + (col))
#define RAW(row, col) raw_image[(row)*raw_width + (col)]
#define BAYER(row, col)                                                        \
  image[((row) >> shrink) * iwidth + ((col) >> shrink)][FC(row, col)]

#define BAYER2(row, col)                                                       \
  image[((row) >> shrink) * iwidth + ((col) >> shrink)][fcol(row, col)]
#define BAYERC(row, col, c)                                                    \
  imgdata.image[((row) >> IO.shrink) * S.iwidth + ((col) >> IO.shrink)][c]

#endif
