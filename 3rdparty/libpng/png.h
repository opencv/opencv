
/* png.h - header file for PNG reference library
 *
 * libpng version 1.4.3 - June 26, 2010
 * Copyright (c) 1998-2010 Glenn Randers-Pehrson
 * (Version 0.96 Copyright (c) 1996, 1997 Andreas Dilger)
 * (Version 0.88 Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.)
 *
 * This code is released under the libpng license (See LICENSE, below)
 *
 * Authors and maintainers:
 *  libpng versions 0.71, May 1995, through 0.88, January 1996: Guy Schalnat
 *  libpng versions 0.89c, June 1996, through 0.96, May 1997: Andreas Dilger
 *  libpng versions 0.97, January 1998, through 1.4.3 - June 26, 2010: Glenn
 *  See also "Contributing Authors", below.
 *
 * Note about libpng version numbers:
 *
 *    Due to various miscommunications, unforeseen code incompatibilities
 *    and occasional factors outside the authors' control, version numbering
 *    on the library has not always been consistent and straightforward.
 *    The following table summarizes matters since version 0.89c, which was
 *    the first widely used release:
 *
 *    source                 png.h  png.h  shared-lib
 *    version                string   int  version
 *    -------                ------ -----  ----------
 *    0.89c "1.0 beta 3"     0.89      89  1.0.89
 *    0.90  "1.0 beta 4"     0.90      90  0.90  [should have been 2.0.90]
 *    0.95  "1.0 beta 5"     0.95      95  0.95  [should have been 2.0.95]
 *    0.96  "1.0 beta 6"     0.96      96  0.96  [should have been 2.0.96]
 *    0.97b "1.00.97 beta 7" 1.00.97   97  1.0.1 [should have been 2.0.97]
 *    0.97c                  0.97      97  2.0.97
 *    0.98                   0.98      98  2.0.98
 *    0.99                   0.99      98  2.0.99
 *    0.99a-m                0.99      99  2.0.99
 *    1.00                   1.00     100  2.1.0 [100 should be 10000]
 *    1.0.0      (from here on, the   100  2.1.0 [100 should be 10000]
 *    1.0.1       png.h string is   10001  2.1.0
 *    1.0.1a-e    identical to the  10002  from here on, the shared library
 *    1.0.2       source version)   10002  is 2.V where V is the source code
 *    1.0.2a-b                      10003  version, except as noted.
 *    1.0.3                         10003
 *    1.0.3a-d                      10004
 *    1.0.4                         10004
 *    1.0.4a-f                      10005
 *    1.0.5 (+ 2 patches)           10005
 *    1.0.5a-d                      10006
 *    1.0.5e-r                      10100 (not source compatible)
 *    1.0.5s-v                      10006 (not binary compatible)
 *    1.0.6 (+ 3 patches)           10006 (still binary incompatible)
 *    1.0.6d-f                      10007 (still binary incompatible)
 *    1.0.6g                        10007
 *    1.0.6h                        10007  10.6h (testing xy.z so-numbering)
 *    1.0.6i                        10007  10.6i
 *    1.0.6j                        10007  2.1.0.6j (incompatible with 1.0.0)
 *    1.0.7beta11-14        DLLNUM  10007  2.1.0.7beta11-14 (binary compatible)
 *    1.0.7beta15-18           1    10007  2.1.0.7beta15-18 (binary compatible)
 *    1.0.7rc1-2               1    10007  2.1.0.7rc1-2 (binary compatible)
 *    1.0.7                    1    10007  (still compatible)
 *    1.0.8beta1-4             1    10008  2.1.0.8beta1-4
 *    1.0.8rc1                 1    10008  2.1.0.8rc1
 *    1.0.8                    1    10008  2.1.0.8
 *    1.0.9beta1-6             1    10009  2.1.0.9beta1-6
 *    1.0.9rc1                 1    10009  2.1.0.9rc1
 *    1.0.9beta7-10            1    10009  2.1.0.9beta7-10
 *    1.0.9rc2                 1    10009  2.1.0.9rc2
 *    1.0.9                    1    10009  2.1.0.9
 *    1.0.10beta1              1    10010  2.1.0.10beta1
 *    1.0.10rc1                1    10010  2.1.0.10rc1
 *    1.0.10                   1    10010  2.1.0.10
 *    1.0.11beta1-3            1    10011  2.1.0.11beta1-3
 *    1.0.11rc1                1    10011  2.1.0.11rc1
 *    1.0.11                   1    10011  2.1.0.11
 *    1.0.12beta1-2            2    10012  2.1.0.12beta1-2
 *    1.0.12rc1                2    10012  2.1.0.12rc1
 *    1.0.12                   2    10012  2.1.0.12
 *    1.1.0a-f                 -    10100  2.1.1.0a-f (branch abandoned)
 *    1.2.0beta1-2             2    10200  2.1.2.0beta1-2
 *    1.2.0beta3-5             3    10200  3.1.2.0beta3-5
 *    1.2.0rc1                 3    10200  3.1.2.0rc1
 *    1.2.0                    3    10200  3.1.2.0
 *    1.2.1beta1-4             3    10201  3.1.2.1beta1-4
 *    1.2.1rc1-2               3    10201  3.1.2.1rc1-2
 *    1.2.1                    3    10201  3.1.2.1
 *    1.2.2beta1-6            12    10202  12.so.0.1.2.2beta1-6
 *    1.0.13beta1             10    10013  10.so.0.1.0.13beta1
 *    1.0.13rc1               10    10013  10.so.0.1.0.13rc1
 *    1.2.2rc1                12    10202  12.so.0.1.2.2rc1
 *    1.0.13                  10    10013  10.so.0.1.0.13
 *    1.2.2                   12    10202  12.so.0.1.2.2
 *    1.2.3rc1-6              12    10203  12.so.0.1.2.3rc1-6
 *    1.2.3                   12    10203  12.so.0.1.2.3
 *    1.2.4beta1-3            13    10204  12.so.0.1.2.4beta1-3
 *    1.0.14rc1               13    10014  10.so.0.1.0.14rc1
 *    1.2.4rc1                13    10204  12.so.0.1.2.4rc1
 *    1.0.14                  10    10014  10.so.0.1.0.14
 *    1.2.4                   13    10204  12.so.0.1.2.4
 *    1.2.5beta1-2            13    10205  12.so.0.1.2.5beta1-2
 *    1.0.15rc1-3             10    10015  10.so.0.1.0.15rc1-3
 *    1.2.5rc1-3              13    10205  12.so.0.1.2.5rc1-3
 *    1.0.15                  10    10015  10.so.0.1.0.15
 *    1.2.5                   13    10205  12.so.0.1.2.5
 *    1.2.6beta1-4            13    10206  12.so.0.1.2.6beta1-4
 *    1.0.16                  10    10016  10.so.0.1.0.16
 *    1.2.6                   13    10206  12.so.0.1.2.6
 *    1.2.7beta1-2            13    10207  12.so.0.1.2.7beta1-2
 *    1.0.17rc1               10    10017  12.so.0.1.0.17rc1
 *    1.2.7rc1                13    10207  12.so.0.1.2.7rc1
 *    1.0.17                  10    10017  12.so.0.1.0.17
 *    1.2.7                   13    10207  12.so.0.1.2.7
 *    1.2.8beta1-5            13    10208  12.so.0.1.2.8beta1-5
 *    1.0.18rc1-5             10    10018  12.so.0.1.0.18rc1-5
 *    1.2.8rc1-5              13    10208  12.so.0.1.2.8rc1-5
 *    1.0.18                  10    10018  12.so.0.1.0.18
 *    1.2.8                   13    10208  12.so.0.1.2.8
 *    1.2.9beta1-3            13    10209  12.so.0.1.2.9beta1-3
 *    1.2.9beta4-11           13    10209  12.so.0.9[.0]
 *    1.2.9rc1                13    10209  12.so.0.9[.0]
 *    1.2.9                   13    10209  12.so.0.9[.0]
 *    1.2.10beta1-7           13    10210  12.so.0.10[.0]
 *    1.2.10rc1-2             13    10210  12.so.0.10[.0]
 *    1.2.10                  13    10210  12.so.0.10[.0]
 *    1.4.0beta1-5            14    10400  14.so.0.0[.0]
 *    1.2.11beta1-4           13    10211  12.so.0.11[.0]
 *    1.4.0beta7-8            14    10400  14.so.0.0[.0]
 *    1.2.11                  13    10211  12.so.0.11[.0]
 *    1.2.12                  13    10212  12.so.0.12[.0]
 *    1.4.0beta9-14           14    10400  14.so.0.0[.0]
 *    1.2.13                  13    10213  12.so.0.13[.0]
 *    1.4.0beta15-36          14    10400  14.so.0.0[.0]
 *    1.4.0beta37-87          14    10400  14.so.14.0[.0]
 *    1.4.0rc01               14    10400  14.so.14.0[.0]
 *    1.4.0beta88-109         14    10400  14.so.14.0[.0]
 *    1.4.0rc02-08            14    10400  14.so.14.0[.0]
 *    1.4.0                   14    10400  14.so.14.0[.0]
 *    1.4.1beta01-03          14    10401  14.so.14.1[.0]
 *    1.4.1rc01               14    10401  14.so.14.1[.0]
 *    1.4.1beta04-12          14    10401  14.so.14.1[.0]
 *    1.4.1rc02-04            14    10401  14.so.14.1[.0]
 *    1.4.1                   14    10401  14.so.14.1[.0]
 *    1.4.2beta01             14    10402  14.so.14.2[.0]
 *    1.4.2rc02-06            14    10402  14.so.14.2[.0]
 *    1.4.2                   14    10402  14.so.14.2[.0]
 *    1.4.3beta01-05          14    10403  14.so.14.3[.0]
 *    1.4.3rc01-03            14    10403  14.so.14.3[.0]
 *    1.4.3                   14    10403  14.so.14.3[.0]
 *
 *    Henceforth the source version will match the shared-library major
 *    and minor numbers; the shared-library major version number will be
 *    used for changes in backward compatibility, as it is intended.  The
 *    PNG_LIBPNG_VER macro, which is not used within libpng but is available
 *    for applications, is an unsigned integer of the form xyyzz corresponding
 *    to the source version x.y.z (leading zeros in y and z).  Beta versions
 *    were given the previous public release number plus a letter, until
 *    version 1.0.6j; from then on they were given the upcoming public
 *    release number plus "betaNN" or "rcN".
 *
 *    Binary incompatibility exists only when applications make direct access
 *    to the info_ptr or png_ptr members through png.h, and the compiled
 *    application is loaded with a different version of the library.
 *
 *    DLLNUM will change each time there are forward or backward changes
 *    in binary compatibility (e.g., when a new feature is added).
 *
 * See libpng.txt or libpng.3 for more information.  The PNG specification
 * is available as a W3C Recommendation and as an ISO Specification,
 * <http://www.w3.org/TR/2003/REC-PNG-20031110/
 */

/*
 * COPYRIGHT NOTICE, DISCLAIMER, and LICENSE:
 *
 * If you modify libpng you may insert additional notices immediately following
 * this sentence.
 *
 * This code is released under the libpng license.
 *
 * libpng versions 1.2.6, August 15, 2004, through 1.4.3, June 26, 2010, are
 * Copyright (c) 2004, 2006-2010 Glenn Randers-Pehrson, and are
 * distributed according to the same disclaimer and license as libpng-1.2.5
 * with the following individual added to the list of Contributing Authors:
 *
 *    Cosmin Truta
 *
 * libpng versions 1.0.7, July 1, 2000, through 1.2.5, October 3, 2002, are
 * Copyright (c) 2000-2002 Glenn Randers-Pehrson, and are
 * distributed according to the same disclaimer and license as libpng-1.0.6
 * with the following individuals added to the list of Contributing Authors:
 *
 *    Simon-Pierre Cadieux
 *    Eric S. Raymond
 *    Gilles Vollant
 *
 * and with the following additions to the disclaimer:
 *
 *    There is no warranty against interference with your enjoyment of the
 *    library or against infringement.  There is no warranty that our
 *    efforts or the library will fulfill any of your particular purposes
 *    or needs.  This library is provided with all faults, and the entire
 *    risk of satisfactory quality, performance, accuracy, and effort is with
 *    the user.
 *
 * libpng versions 0.97, January 1998, through 1.0.6, March 20, 2000, are
 * Copyright (c) 1998, 1999, 2000 Glenn Randers-Pehrson, and are
 * distributed according to the same disclaimer and license as libpng-0.96,
 * with the following individuals added to the list of Contributing Authors:
 *
 *    Tom Lane
 *    Glenn Randers-Pehrson
 *    Willem van Schaik
 *
 * libpng versions 0.89, June 1996, through 0.96, May 1997, are
 * Copyright (c) 1996, 1997 Andreas Dilger
 * Distributed according to the same disclaimer and license as libpng-0.88,
 * with the following individuals added to the list of Contributing Authors:
 *
 *    John Bowler
 *    Kevin Bracey
 *    Sam Bushell
 *    Magnus Holmgren
 *    Greg Roelofs
 *    Tom Tanner
 *
 * libpng versions 0.5, May 1995, through 0.88, January 1996, are
 * Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.
 *
 * For the purposes of this copyright and license, "Contributing Authors"
 * is defined as the following set of individuals:
 *
 *    Andreas Dilger
 *    Dave Martindale
 *    Guy Eric Schalnat
 *    Paul Schmidt
 *    Tim Wegner
 *
 * The PNG Reference Library is supplied "AS IS".  The Contributing Authors
 * and Group 42, Inc. disclaim all warranties, expressed or implied,
 * including, without limitation, the warranties of merchantability and of
 * fitness for any purpose.  The Contributing Authors and Group 42, Inc.
 * assume no liability for direct, indirect, incidental, special, exemplary,
 * or consequential damages, which may result from the use of the PNG
 * Reference Library, even if advised of the possibility of such damage.
 *
 * Permission is hereby granted to use, copy, modify, and distribute this
 * source code, or portions hereof, for any purpose, without fee, subject
 * to the following restrictions:
 *
 * 1. The origin of this source code must not be misrepresented.
 *
 * 2. Altered versions must be plainly marked as such and
 * must not be misrepresented as being the original source.
 *
 * 3. This Copyright notice may not be removed or altered from
 *    any source or altered source distribution.
 *
 * The Contributing Authors and Group 42, Inc. specifically permit, without
 * fee, and encourage the use of this source code as a component to
 * supporting the PNG file format in commercial products.  If you use this
 * source code in a product, acknowledgment is not required but would be
 * appreciated.
 */

/*
 * A "png_get_copyright" function is available, for convenient use in "about"
 * boxes and the like:
 *
 *     printf("%s",png_get_copyright(NULL));
 *
 * Also, the PNG logo (in PNG format, of course) is supplied in the
 * files "pngbar.png" and "pngbar.jpg (88x31) and "pngnow.png" (98x31).
 */

/*
 * Libpng is OSI Certified Open Source Software.  OSI Certified is a
 * certification mark of the Open Source Initiative.
 */

/*
 * The contributing authors would like to thank all those who helped
 * with testing, bug fixes, and patience.  This wouldn't have been
 * possible without all of you.
 *
 * Thanks to Frank J. T. Wojcik for helping with the documentation.
 */

/*
 * Y2K compliance in libpng:
 * =========================
 *
 *    June 26, 2010
 *
 *    Since the PNG Development group is an ad-hoc body, we can't make
 *    an official declaration.
 *
 *    This is your unofficial assurance that libpng from version 0.71 and
 *    upward through 1.4.3 are Y2K compliant.  It is my belief that earlier
 *    versions were also Y2K compliant.
 *
 *    Libpng only has three year fields.  One is a 2-byte unsigned integer
 *    that will hold years up to 65535.  The other two hold the date in text
 *    format, and will hold years up to 9999.
 *
 *    The integer is
 *        "png_uint_16 year" in png_time_struct.
 *
 *    The strings are
 *        "png_charp time_buffer" in png_struct and
 *        "near_time_buffer", which is a local character string in png.c.
 *
 *    There are seven time-related functions:
 *        png.c: png_convert_to_rfc_1123() in png.c
 *          (formerly png_convert_to_rfc_1152() in error)
 *        png_convert_from_struct_tm() in pngwrite.c, called in pngwrite.c
 *        png_convert_from_time_t() in pngwrite.c
 *        png_get_tIME() in pngget.c
 *        png_handle_tIME() in pngrutil.c, called in pngread.c
 *        png_set_tIME() in pngset.c
 *        png_write_tIME() in pngwutil.c, called in pngwrite.c
 *
 *    All handle dates properly in a Y2K environment.  The
 *    png_convert_from_time_t() function calls gmtime() to convert from system
 *    clock time, which returns (year - 1900), which we properly convert to
 *    the full 4-digit year.  There is a possibility that applications using
 *    libpng are not passing 4-digit years into the png_convert_to_rfc_1123()
 *    function, or that they are incorrectly passing only a 2-digit year
 *    instead of "year - 1900" into the png_convert_from_struct_tm() function,
 *    but this is not under our control.  The libpng documentation has always
 *    stated that it works with 4-digit years, and the APIs have been
 *    documented as such.
 *
 *    The tIME chunk itself is also Y2K compliant.  It uses a 2-byte unsigned
 *    integer to hold the year, and can hold years as large as 65535.
 *
 *    zlib, upon which libpng depends, is also Y2K compliant.  It contains
 *    no date-related code.
 *
 *       Glenn Randers-Pehrson
 *       libpng maintainer
 *       PNG Development Group
 */

#ifndef PNG_H
#define PNG_H

/* This is not the place to learn how to use libpng.  The file libpng.txt
 * describes how to use libpng, and the file example.c summarizes it
 * with some code on which to build.  This file is useful for looking
 * at the actual function definitions and structure components.
 */

/* Version information for png.h - this should match the version in png.c */
#define PNG_LIBPNG_VER_STRING "1.4.3"
#define PNG_HEADER_VERSION_STRING \
   " libpng version 1.4.3 - June 26, 2010\n"

#define PNG_LIBPNG_VER_SONUM   14
#define PNG_LIBPNG_VER_DLLNUM  14

/* These should match the first 3 components of PNG_LIBPNG_VER_STRING: */
#define PNG_LIBPNG_VER_MAJOR   1
#define PNG_LIBPNG_VER_MINOR   4
#define PNG_LIBPNG_VER_RELEASE 3
/* This should match the numeric part of the final component of
 * PNG_LIBPNG_VER_STRING, omitting any leading zero:
 */

#define PNG_LIBPNG_VER_BUILD  0

/* Release Status */
#define PNG_LIBPNG_BUILD_ALPHA    1
#define PNG_LIBPNG_BUILD_BETA     2
#define PNG_LIBPNG_BUILD_RC       3
#define PNG_LIBPNG_BUILD_STABLE   4
#define PNG_LIBPNG_BUILD_RELEASE_STATUS_MASK 7

/* Release-Specific Flags */
#define PNG_LIBPNG_BUILD_PATCH    8 /* Can be OR'ed with
                                       PNG_LIBPNG_BUILD_STABLE only */
#define PNG_LIBPNG_BUILD_PRIVATE 16 /* Cannot be OR'ed with
                                       PNG_LIBPNG_BUILD_SPECIAL */
#define PNG_LIBPNG_BUILD_SPECIAL 32 /* Cannot be OR'ed with
                                       PNG_LIBPNG_BUILD_PRIVATE */

#define PNG_LIBPNG_BUILD_BASE_TYPE PNG_LIBPNG_BUILD_BETA

/* Careful here.  At one time, Guy wanted to use 082, but that would be octal.
 * We must not include leading zeros.
 * Versions 0.7 through 1.0.0 were in the range 0 to 100 here (only
 * version 1.0.0 was mis-numbered 100 instead of 10000).  From
 * version 1.0.1 it's    xxyyzz, where x=major, y=minor, z=release
 */
#define PNG_LIBPNG_VER 10403 /* 1.4.3 */

#ifndef PNG_VERSION_INFO_ONLY
/* Include the compression library's header */
#include "zlib.h"
#endif

/* Include all user configurable info, including optional assembler routines */
#include "pngconf.h"

/*
 * Added at libpng-1.2.8
 *
 * Ref MSDN: Private as priority over Special
 * VS_FF_PRIVATEBUILD File *was not* built using standard release
 * procedures. If this value is given, the StringFileInfo block must
 * contain a PrivateBuild string.
 *
 * VS_FF_SPECIALBUILD File *was* built by the original company using
 * standard release procedures but is a variation of the standard
 * file of the same version number. If this value is given, the
 * StringFileInfo block must contain a SpecialBuild string.
 */

#ifdef PNG_USER_PRIVATEBUILD
#  define PNG_LIBPNG_BUILD_TYPE \
          (PNG_LIBPNG_BUILD_BASE_TYPE | PNG_LIBPNG_BUILD_PRIVATE)
#else
#  ifdef PNG_LIBPNG_SPECIALBUILD
#    define PNG_LIBPNG_BUILD_TYPE \
            (PNG_LIBPNG_BUILD_BASE_TYPE | PNG_LIBPNG_BUILD_SPECIAL)
#  else
#    define PNG_LIBPNG_BUILD_TYPE (PNG_LIBPNG_BUILD_BASE_TYPE)
#  endif
#endif

#ifndef PNG_VERSION_INFO_ONLY

/* Inhibit C++ name-mangling for libpng functions but not for system calls. */
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* This file is arranged in several sections.  The first section contains
 * structure and type definitions.  The second section contains the external
 * library functions, while the third has the internal library functions,
 * which applications aren't expected to use directly.
 */

/* Variables declared in png.c - only it needs to define PNG_NO_EXTERN */
#if !defined(PNG_NO_EXTERN) || defined(PNG_ALWAYS_EXTERN)
/* Version information for C files, stored in png.c.  This had better match
 * the version above.
 */
#define png_libpng_ver png_get_header_ver(NULL)

#endif /* PNG_NO_EXTERN */

/* Three color definitions.  The order of the red, green, and blue, (and the
 * exact size) is not important, although the size of the fields need to
 * be png_byte or png_uint_16 (as defined below).
 */
typedef struct png_color_struct
{
   png_byte red;
   png_byte green;
   png_byte blue;
} png_color;
typedef png_color FAR * png_colorp;
typedef png_color FAR * FAR * png_colorpp;

typedef struct png_color_16_struct
{
   png_byte index;    /* used for palette files */
   png_uint_16 red;   /* for use in red green blue files */
   png_uint_16 green;
   png_uint_16 blue;
   png_uint_16 gray;  /* for use in grayscale files */
} png_color_16;
typedef png_color_16 FAR * png_color_16p;
typedef png_color_16 FAR * FAR * png_color_16pp;

typedef struct png_color_8_struct
{
   png_byte red;   /* for use in red green blue files */
   png_byte green;
   png_byte blue;
   png_byte gray;  /* for use in grayscale files */
   png_byte alpha; /* for alpha channel files */
} png_color_8;
typedef png_color_8 FAR * png_color_8p;
typedef png_color_8 FAR * FAR * png_color_8pp;

/*
 * The following two structures are used for the in-core representation
 * of sPLT chunks.
 */
typedef struct png_sPLT_entry_struct
{
   png_uint_16 red;
   png_uint_16 green;
   png_uint_16 blue;
   png_uint_16 alpha;
   png_uint_16 frequency;
} png_sPLT_entry;
typedef png_sPLT_entry FAR * png_sPLT_entryp;
typedef png_sPLT_entry FAR * FAR * png_sPLT_entrypp;

/*  When the depth of the sPLT palette is 8 bits, the color and alpha samples
 *  occupy the LSB of their respective members, and the MSB of each member
 *  is zero-filled.  The frequency member always occupies the full 16 bits.
 */

typedef struct png_sPLT_struct
{
   png_charp name;           /* palette name */
   png_byte depth;           /* depth of palette samples */
   png_sPLT_entryp entries;  /* palette entries */
   png_int_32 nentries;      /* number of palette entries */
} png_sPLT_t;
typedef png_sPLT_t FAR * png_sPLT_tp;
typedef png_sPLT_t FAR * FAR * png_sPLT_tpp;

#ifdef PNG_TEXT_SUPPORTED
/* png_text holds the contents of a text/ztxt/itxt chunk in a PNG file,
 * and whether that contents is compressed or not.  The "key" field
 * points to a regular zero-terminated C string.  The "text", "lang", and
 * "lang_key" fields can be regular C strings, empty strings, or NULL pointers.
 * However, the * structure returned by png_get_text() will always contain
 * regular zero-terminated C strings (possibly empty), never NULL pointers,
 * so they can be safely used in printf() and other string-handling functions.
 */
typedef struct png_text_struct
{
   int  compression;       /* compression value:
                             -1: tEXt, none
                              0: zTXt, deflate
                              1: iTXt, none
                              2: iTXt, deflate  */
   png_charp key;          /* keyword, 1-79 character description of "text" */
   png_charp text;         /* comment, may be an empty string (ie "")
                              or a NULL pointer */
   png_size_t text_length; /* length of the text string */
#ifdef PNG_iTXt_SUPPORTED
   png_size_t itxt_length; /* length of the itxt string */
   png_charp lang;         /* language code, 0-79 characters
                              or a NULL pointer */
   png_charp lang_key;     /* keyword translated UTF-8 string, 0 or more
                              chars or a NULL pointer */
#endif
} png_text;
typedef png_text FAR * png_textp;
typedef png_text FAR * FAR * png_textpp;
#endif

/* Supported compression types for text in PNG files (tEXt, and zTXt).
 * The values of the PNG_TEXT_COMPRESSION_ defines should NOT be changed. */
#define PNG_TEXT_COMPRESSION_NONE_WR -3
#define PNG_TEXT_COMPRESSION_zTXt_WR -2
#define PNG_TEXT_COMPRESSION_NONE    -1
#define PNG_TEXT_COMPRESSION_zTXt     0
#define PNG_ITXT_COMPRESSION_NONE     1
#define PNG_ITXT_COMPRESSION_zTXt     2
#define PNG_TEXT_COMPRESSION_LAST     3  /* Not a valid value */

/* png_time is a way to hold the time in an machine independent way.
 * Two conversions are provided, both from time_t and struct tm.  There
 * is no portable way to convert to either of these structures, as far
 * as I know.  If you know of a portable way, send it to me.  As a side
 * note - PNG has always been Year 2000 compliant!
 */
typedef struct png_time_struct
{
   png_uint_16 year; /* full year, as in, 1995 */
   png_byte month;   /* month of year, 1 - 12 */
   png_byte day;     /* day of month, 1 - 31 */
   png_byte hour;    /* hour of day, 0 - 23 */
   png_byte minute;  /* minute of hour, 0 - 59 */
   png_byte second;  /* second of minute, 0 - 60 (for leap seconds) */
} png_time;
typedef png_time FAR * png_timep;
typedef png_time FAR * FAR * png_timepp;

#if defined(PNG_UNKNOWN_CHUNKS_SUPPORTED) || \
 defined(PNG_HANDLE_AS_UNKNOWN_SUPPORTED)
/* png_unknown_chunk is a structure to hold queued chunks for which there is
 * no specific support.  The idea is that we can use this to queue
 * up private chunks for output even though the library doesn't actually
 * know about their semantics.
 */
typedef struct png_unknown_chunk_t
{
    png_byte name[5];
    png_byte *data;
    png_size_t size;

    /* libpng-using applications should NOT directly modify this byte. */
    png_byte location; /* mode of operation at read time */
}
png_unknown_chunk;
typedef png_unknown_chunk FAR * png_unknown_chunkp;
typedef png_unknown_chunk FAR * FAR * png_unknown_chunkpp;
#endif

/* png_info is a structure that holds the information in a PNG file so
 * that the application can find out the characteristics of the image.
 * If you are reading the file, this structure will tell you what is
 * in the PNG file.  If you are writing the file, fill in the information
 * you want to put into the PNG file, then call png_write_info().
 * The names chosen should be very close to the PNG specification, so
 * consult that document for information about the meaning of each field.
 *
 * With libpng < 0.95, it was only possible to directly set and read the
 * the values in the png_info_struct, which meant that the contents and
 * order of the values had to remain fixed.  With libpng 0.95 and later,
 * however, there are now functions that abstract the contents of
 * png_info_struct from the application, so this makes it easier to use
 * libpng with dynamic libraries, and even makes it possible to use
 * libraries that don't have all of the libpng ancillary chunk-handing
 * functionality.
 *
 * In any case, the order of the parameters in png_info_struct should NOT
 * be changed for as long as possible to keep compatibility with applications
 * that use the old direct-access method with png_info_struct.
 *
 * The following members may have allocated storage attached that should be
 * cleaned up before the structure is discarded: palette, trans, text,
 * pcal_purpose, pcal_units, pcal_params, hist, iccp_name, iccp_profile,
 * splt_palettes, scal_unit, row_pointers, and unknowns.   By default, these
 * are automatically freed when the info structure is deallocated, if they were
 * allocated internally by libpng.  This behavior can be changed by means
 * of the png_data_freer() function.
 *
 * More allocation details: all the chunk-reading functions that
 * change these members go through the corresponding png_set_*
 * functions.  A function to clear these members is available: see
 * png_free_data().  The png_set_* functions do not depend on being
 * able to point info structure members to any of the storage they are
 * passed (they make their own copies), EXCEPT that the png_set_text
 * functions use the same storage passed to them in the text_ptr or
 * itxt_ptr structure argument, and the png_set_rows and png_set_unknowns
 * functions do not make their own copies.
 */
typedef struct png_info_struct
{
   /* the following are necessary for every PNG file */
   png_uint_32 width PNG_DEPSTRUCT;  /* width of image in pixels (from IHDR) */
   png_uint_32 height PNG_DEPSTRUCT; /* height of image in pixels (from IHDR) */
   png_uint_32 valid PNG_DEPSTRUCT;  /* valid chunk data (see PNG_INFO_
                                        below) */
   png_size_t rowbytes PNG_DEPSTRUCT; /* bytes needed to hold an untransformed
                                         row */
   png_colorp palette PNG_DEPSTRUCT;      /* array of color values
                                             (valid & PNG_INFO_PLTE) */
   png_uint_16 num_palette PNG_DEPSTRUCT; /* number of color entries in
                                             "palette" (PLTE) */
   png_uint_16 num_trans PNG_DEPSTRUCT;   /* number of transparent palette
                                             color (tRNS) */
   png_byte bit_depth PNG_DEPSTRUCT;      /* 1, 2, 4, 8, or 16 bits/channel
                                             (from IHDR) */
   png_byte color_type PNG_DEPSTRUCT;     /* see PNG_COLOR_TYPE_ below
                                             (from IHDR) */
   /* The following three should have been named *_method not *_type */
   png_byte compression_type PNG_DEPSTRUCT; /* must be
                                             PNG_COMPRESSION_TYPE_BASE (IHDR) */
   png_byte filter_type PNG_DEPSTRUCT;    /* must be PNG_FILTER_TYPE_BASE
                                             (from IHDR) */
   png_byte interlace_type PNG_DEPSTRUCT; /* One of PNG_INTERLACE_NONE,
                                             PNG_INTERLACE_ADAM7 */

   /* The following is informational only on read, and not used on writes. */
   png_byte channels PNG_DEPSTRUCT;       /* number of data channels per
                                             pixel (1, 2, 3, 4) */
   png_byte pixel_depth PNG_DEPSTRUCT;    /* number of bits per pixel */
   png_byte spare_byte PNG_DEPSTRUCT;     /* to align the data, and for
                                             future use */
   png_byte signature[8] PNG_DEPSTRUCT;   /* magic bytes read by libpng
                                             from start of file */

   /* The rest of the data is optional.  If you are reading, check the
    * valid field to see if the information in these are valid.  If you
    * are writing, set the valid field to those chunks you want written,
    * and initialize the appropriate fields below.
    */

#if defined(PNG_gAMA_SUPPORTED) && defined(PNG_FLOATING_POINT_SUPPORTED)
   /* The gAMA chunk describes the gamma characteristics of the system
    * on which the image was created, normally in the range [1.0, 2.5].
    * Data is valid if (valid & PNG_INFO_gAMA) is non-zero.
    */
   float gamma PNG_DEPSTRUCT; /* gamma value of image,
                                 if (valid & PNG_INFO_gAMA) */
#endif

#ifdef PNG_sRGB_SUPPORTED
    /* GR-P, 0.96a */
    /* Data valid if (valid & PNG_INFO_sRGB) non-zero. */
   png_byte srgb_intent PNG_DEPSTRUCT; /* sRGB rendering intent
                                          [0, 1, 2, or 3] */
#endif

#ifdef PNG_TEXT_SUPPORTED
   /* The tEXt, and zTXt chunks contain human-readable textual data in
    * uncompressed, compressed, and optionally compressed forms, respectively.
    * The data in "text" is an array of pointers to uncompressed,
    * null-terminated C strings. Each chunk has a keyword that describes the
    * textual data contained in that chunk.  Keywords are not required to be
    * unique, and the text string may be empty.  Any number of text chunks may
    * be in an image.
    */
   int num_text PNG_DEPSTRUCT; /* number of comments read/to write */
   int max_text PNG_DEPSTRUCT; /* current size of text array */
   png_textp text PNG_DEPSTRUCT; /* array of comments read/to write */
#endif /* PNG_TEXT_SUPPORTED */

#ifdef PNG_tIME_SUPPORTED
   /* The tIME chunk holds the last time the displayed image data was
    * modified.  See the png_time struct for the contents of this struct.
    */
   png_time mod_time PNG_DEPSTRUCT;
#endif

#ifdef PNG_sBIT_SUPPORTED
   /* The sBIT chunk specifies the number of significant high-order bits
    * in the pixel data.  Values are in the range [1, bit_depth], and are
    * only specified for the channels in the pixel data.  The contents of
    * the low-order bits is not specified.  Data is valid if
    * (valid & PNG_INFO_sBIT) is non-zero.
    */
   png_color_8 sig_bit PNG_DEPSTRUCT; /* significant bits in color channels */
#endif

#if defined(PNG_tRNS_SUPPORTED) || defined(PNG_READ_EXPAND_SUPPORTED) || \
defined(PNG_READ_BACKGROUND_SUPPORTED)
   /* The tRNS chunk supplies transparency data for paletted images and
    * other image types that don't need a full alpha channel.  There are
    * "num_trans" transparency values for a paletted image, stored in the
    * same order as the palette colors, starting from index 0.  Values
    * for the data are in the range [0, 255], ranging from fully transparent
    * to fully opaque, respectively.  For non-paletted images, there is a
    * single color specified that should be treated as fully transparent.
    * Data is valid if (valid & PNG_INFO_tRNS) is non-zero.
    */
   png_bytep trans_alpha PNG_DEPSTRUCT;    /* alpha values for paletted
                                              image */
   png_color_16 trans_color PNG_DEPSTRUCT; /* transparent color for
                                              non-palette image */
#endif

#if defined(PNG_bKGD_SUPPORTED) || defined(PNG_READ_BACKGROUND_SUPPORTED)
   /* The bKGD chunk gives the suggested image background color if the
    * display program does not have its own background color and the image
    * is needs to composited onto a background before display.  The colors
    * in "background" are normally in the same color space/depth as the
    * pixel data.  Data is valid if (valid & PNG_INFO_bKGD) is non-zero.
    */
   png_color_16 background PNG_DEPSTRUCT;
#endif

#ifdef PNG_oFFs_SUPPORTED
   /* The oFFs chunk gives the offset in "offset_unit_type" units rightwards
    * and downwards from the top-left corner of the display, page, or other
    * application-specific co-ordinate space.  See the PNG_OFFSET_ defines
    * below for the unit types.  Valid if (valid & PNG_INFO_oFFs) non-zero.
    */
   png_int_32 x_offset PNG_DEPSTRUCT; /* x offset on page */
   png_int_32 y_offset PNG_DEPSTRUCT; /* y offset on page */
   png_byte offset_unit_type PNG_DEPSTRUCT; /* offset units type */
#endif

#ifdef PNG_pHYs_SUPPORTED
   /* The pHYs chunk gives the physical pixel density of the image for
    * display or printing in "phys_unit_type" units (see PNG_RESOLUTION_
    * defines below).  Data is valid if (valid & PNG_INFO_pHYs) is non-zero.
    */
   png_uint_32 x_pixels_per_unit PNG_DEPSTRUCT; /* horizontal pixel density */
   png_uint_32 y_pixels_per_unit PNG_DEPSTRUCT; /* vertical pixel density */
   png_byte phys_unit_type PNG_DEPSTRUCT; /* resolution type (see
                                             PNG_RESOLUTION_ below) */
#endif

#ifdef PNG_hIST_SUPPORTED
   /* The hIST chunk contains the relative frequency or importance of the
    * various palette entries, so that a viewer can intelligently select a
    * reduced-color palette, if required.  Data is an array of "num_palette"
    * values in the range [0,65535]. Data valid if (valid & PNG_INFO_hIST)
    * is non-zero.
    */
   png_uint_16p hist PNG_DEPSTRUCT;
#endif

#ifdef PNG_cHRM_SUPPORTED
   /* The cHRM chunk describes the CIE color characteristics of the monitor
    * on which the PNG was created.  This data allows the viewer to do gamut
    * mapping of the input image to ensure that the viewer sees the same
    * colors in the image as the creator.  Values are in the range
    * [0.0, 0.8].  Data valid if (valid & PNG_INFO_cHRM) non-zero.
    */
#ifdef PNG_FLOATING_POINT_SUPPORTED
   float x_white PNG_DEPSTRUCT;
   float y_white PNG_DEPSTRUCT;
   float x_red PNG_DEPSTRUCT;
   float y_red PNG_DEPSTRUCT;
   float x_green PNG_DEPSTRUCT;
   float y_green PNG_DEPSTRUCT;
   float x_blue PNG_DEPSTRUCT;
   float y_blue PNG_DEPSTRUCT;
#endif
#endif

#ifdef PNG_pCAL_SUPPORTED
   /* The pCAL chunk describes a transformation between the stored pixel
    * values and original physical data values used to create the image.
    * The integer range [0, 2^bit_depth - 1] maps to the floating-point
    * range given by [pcal_X0, pcal_X1], and are further transformed by a
    * (possibly non-linear) transformation function given by "pcal_type"
    * and "pcal_params" into "pcal_units".  Please see the PNG_EQUATION_
    * defines below, and the PNG-Group's PNG extensions document for a
    * complete description of the transformations and how they should be
    * implemented, and for a description of the ASCII parameter strings.
    * Data values are valid if (valid & PNG_INFO_pCAL) non-zero.
    */
   png_charp pcal_purpose PNG_DEPSTRUCT;  /* pCAL chunk description string */
   png_int_32 pcal_X0 PNG_DEPSTRUCT;      /* minimum value */
   png_int_32 pcal_X1 PNG_DEPSTRUCT;      /* maximum value */
   png_charp pcal_units PNG_DEPSTRUCT;    /* Latin-1 string giving physical
                                             units */
   png_charpp pcal_params PNG_DEPSTRUCT;  /* ASCII strings containing
                                             parameter values */
   png_byte pcal_type PNG_DEPSTRUCT;      /* equation type
                                             (see PNG_EQUATION_ below) */
   png_byte pcal_nparams PNG_DEPSTRUCT;   /* number of parameters given
                                             in pcal_params */
#endif

/* New members added in libpng-1.0.6 */
   png_uint_32 free_me PNG_DEPSTRUCT;     /* flags items libpng is
                                             responsible for freeing */

#if defined(PNG_UNKNOWN_CHUNKS_SUPPORTED) || \
 defined(PNG_HANDLE_AS_UNKNOWN_SUPPORTED)
   /* Storage for unknown chunks that the library doesn't recognize. */
   png_unknown_chunkp unknown_chunks PNG_DEPSTRUCT;
   png_size_t unknown_chunks_num PNG_DEPSTRUCT;
#endif

#ifdef PNG_iCCP_SUPPORTED
   /* iCCP chunk data. */
   png_charp iccp_name PNG_DEPSTRUCT;     /* profile name */
   png_charp iccp_profile PNG_DEPSTRUCT;  /* International Color Consortium
                                             profile data */
                            /* Note to maintainer: should be png_bytep */
   png_uint_32 iccp_proflen PNG_DEPSTRUCT;  /* ICC profile data length */
   png_byte iccp_compression PNG_DEPSTRUCT; /* Always zero */
#endif

#ifdef PNG_sPLT_SUPPORTED
   /* Data on sPLT chunks (there may be more than one). */
   png_sPLT_tp splt_palettes PNG_DEPSTRUCT;
   png_uint_32 splt_palettes_num PNG_DEPSTRUCT;
#endif

#ifdef PNG_sCAL_SUPPORTED
   /* The sCAL chunk describes the actual physical dimensions of the
    * subject matter of the graphic.  The chunk contains a unit specification
    * a byte value, and two ASCII strings representing floating-point
    * values.  The values are width and height corresponsing to one pixel
    * in the image.  This external representation is converted to double
    * here.  Data values are valid if (valid & PNG_INFO_sCAL) is non-zero.
    */
   png_byte scal_unit PNG_DEPSTRUCT;         /* unit of physical scale */
#ifdef PNG_FLOATING_POINT_SUPPORTED
   double scal_pixel_width PNG_DEPSTRUCT;    /* width of one pixel */
   double scal_pixel_height PNG_DEPSTRUCT;   /* height of one pixel */
#endif
#ifdef PNG_FIXED_POINT_SUPPORTED
   png_charp scal_s_width PNG_DEPSTRUCT;     /* string containing height */
   png_charp scal_s_height PNG_DEPSTRUCT;    /* string containing width */
#endif
#endif

#ifdef PNG_INFO_IMAGE_SUPPORTED
   /* Memory has been allocated if (valid & PNG_ALLOCATED_INFO_ROWS)
      non-zero */
   /* Data valid if (valid & PNG_INFO_IDAT) non-zero */
   png_bytepp row_pointers PNG_DEPSTRUCT;        /* the image bits */
#endif

#if defined(PNG_FIXED_POINT_SUPPORTED) && defined(PNG_gAMA_SUPPORTED)
   png_fixed_point int_gamma PNG_DEPSTRUCT; /* gamma of image,
                                               if (valid & PNG_INFO_gAMA) */
#endif

#if defined(PNG_cHRM_SUPPORTED) && defined(PNG_FIXED_POINT_SUPPORTED)
   png_fixed_point int_x_white PNG_DEPSTRUCT;
   png_fixed_point int_y_white PNG_DEPSTRUCT;
   png_fixed_point int_x_red PNG_DEPSTRUCT;
   png_fixed_point int_y_red PNG_DEPSTRUCT;
   png_fixed_point int_x_green PNG_DEPSTRUCT;
   png_fixed_point int_y_green PNG_DEPSTRUCT;
   png_fixed_point int_x_blue PNG_DEPSTRUCT;
   png_fixed_point int_y_blue PNG_DEPSTRUCT;
#endif

} png_info;

typedef png_info FAR * png_infop;
typedef png_info FAR * FAR * png_infopp;

/* Maximum positive integer used in PNG is (2^31)-1 */
#define PNG_UINT_31_MAX ((png_uint_32)0x7fffffffL)
#define PNG_UINT_32_MAX ((png_uint_32)(-1))
#define PNG_SIZE_MAX ((png_size_t)(-1))

/* These describe the color_type field in png_info. */
/* color type masks */
#define PNG_COLOR_MASK_PALETTE    1
#define PNG_COLOR_MASK_COLOR      2
#define PNG_COLOR_MASK_ALPHA      4

/* color types.  Note that not all combinations are legal */
#define PNG_COLOR_TYPE_GRAY 0
#define PNG_COLOR_TYPE_PALETTE  (PNG_COLOR_MASK_COLOR | PNG_COLOR_MASK_PALETTE)
#define PNG_COLOR_TYPE_RGB        (PNG_COLOR_MASK_COLOR)
#define PNG_COLOR_TYPE_RGB_ALPHA  (PNG_COLOR_MASK_COLOR | PNG_COLOR_MASK_ALPHA)
#define PNG_COLOR_TYPE_GRAY_ALPHA (PNG_COLOR_MASK_ALPHA)
/* aliases */
#define PNG_COLOR_TYPE_RGBA  PNG_COLOR_TYPE_RGB_ALPHA
#define PNG_COLOR_TYPE_GA  PNG_COLOR_TYPE_GRAY_ALPHA

/* This is for compression type. PNG 1.0-1.2 only define the single type. */
#define PNG_COMPRESSION_TYPE_BASE 0 /* Deflate method 8, 32K window */
#define PNG_COMPRESSION_TYPE_DEFAULT PNG_COMPRESSION_TYPE_BASE

/* This is for filter type. PNG 1.0-1.2 only define the single type. */
#define PNG_FILTER_TYPE_BASE      0 /* Single row per-byte filtering */
#define PNG_INTRAPIXEL_DIFFERENCING 64 /* Used only in MNG datastreams */
#define PNG_FILTER_TYPE_DEFAULT   PNG_FILTER_TYPE_BASE

/* These are for the interlacing type.  These values should NOT be changed. */
#define PNG_INTERLACE_NONE        0 /* Non-interlaced image */
#define PNG_INTERLACE_ADAM7       1 /* Adam7 interlacing */
#define PNG_INTERLACE_LAST        2 /* Not a valid value */

/* These are for the oFFs chunk.  These values should NOT be changed. */
#define PNG_OFFSET_PIXEL          0 /* Offset in pixels */
#define PNG_OFFSET_MICROMETER     1 /* Offset in micrometers (1/10^6 meter) */
#define PNG_OFFSET_LAST           2 /* Not a valid value */

/* These are for the pCAL chunk.  These values should NOT be changed. */
#define PNG_EQUATION_LINEAR       0 /* Linear transformation */
#define PNG_EQUATION_BASE_E       1 /* Exponential base e transform */
#define PNG_EQUATION_ARBITRARY    2 /* Arbitrary base exponential transform */
#define PNG_EQUATION_HYPERBOLIC   3 /* Hyperbolic sine transformation */
#define PNG_EQUATION_LAST         4 /* Not a valid value */

/* These are for the sCAL chunk.  These values should NOT be changed. */
#define PNG_SCALE_UNKNOWN         0 /* unknown unit (image scale) */
#define PNG_SCALE_METER           1 /* meters per pixel */
#define PNG_SCALE_RADIAN          2 /* radians per pixel */
#define PNG_SCALE_LAST            3 /* Not a valid value */

/* These are for the pHYs chunk.  These values should NOT be changed. */
#define PNG_RESOLUTION_UNKNOWN    0 /* pixels/unknown unit (aspect ratio) */
#define PNG_RESOLUTION_METER      1 /* pixels/meter */
#define PNG_RESOLUTION_LAST       2 /* Not a valid value */

/* These are for the sRGB chunk.  These values should NOT be changed. */
#define PNG_sRGB_INTENT_PERCEPTUAL 0
#define PNG_sRGB_INTENT_RELATIVE   1
#define PNG_sRGB_INTENT_SATURATION 2
#define PNG_sRGB_INTENT_ABSOLUTE   3
#define PNG_sRGB_INTENT_LAST       4 /* Not a valid value */

/* This is for text chunks */
#define PNG_KEYWORD_MAX_LENGTH     79

/* Maximum number of entries in PLTE/sPLT/tRNS arrays */
#define PNG_MAX_PALETTE_LENGTH    256

/* These determine if an ancillary chunk's data has been successfully read
 * from the PNG header, or if the application has filled in the corresponding
 * data in the info_struct to be written into the output file.  The values
 * of the PNG_INFO_<chunk> defines should NOT be changed.
 */
#define PNG_INFO_gAMA 0x0001
#define PNG_INFO_sBIT 0x0002
#define PNG_INFO_cHRM 0x0004
#define PNG_INFO_PLTE 0x0008
#define PNG_INFO_tRNS 0x0010
#define PNG_INFO_bKGD 0x0020
#define PNG_INFO_hIST 0x0040
#define PNG_INFO_pHYs 0x0080
#define PNG_INFO_oFFs 0x0100
#define PNG_INFO_tIME 0x0200
#define PNG_INFO_pCAL 0x0400
#define PNG_INFO_sRGB 0x0800   /* GR-P, 0.96a */
#define PNG_INFO_iCCP 0x1000   /* ESR, 1.0.6 */
#define PNG_INFO_sPLT 0x2000   /* ESR, 1.0.6 */
#define PNG_INFO_sCAL 0x4000   /* ESR, 1.0.6 */
#define PNG_INFO_IDAT 0x8000L  /* ESR, 1.0.6 */

/* This is used for the transformation routines, as some of them
 * change these values for the row.  It also should enable using
 * the routines for other purposes.
 */
typedef struct png_row_info_struct
{
   png_uint_32 width; /* width of row */
   png_size_t rowbytes; /* number of bytes in row */
   png_byte color_type; /* color type of row */
   png_byte bit_depth; /* bit depth of row */
   png_byte channels; /* number of channels (1, 2, 3, or 4) */
   png_byte pixel_depth; /* bits per pixel (depth * channels) */
} png_row_info;

typedef png_row_info FAR * png_row_infop;
typedef png_row_info FAR * FAR * png_row_infopp;

/* These are the function types for the I/O functions and for the functions
 * that allow the user to override the default I/O functions with his or her
 * own.  The png_error_ptr type should match that of user-supplied warning
 * and error functions, while the png_rw_ptr type should match that of the
 * user read/write data functions.
 */
typedef struct png_struct_def png_struct;
typedef png_struct FAR * png_structp;

typedef void (PNGAPI *png_error_ptr) PNGARG((png_structp, png_const_charp));
typedef void (PNGAPI *png_rw_ptr) PNGARG((png_structp, png_bytep, png_size_t));
typedef void (PNGAPI *png_flush_ptr) PNGARG((png_structp));
typedef void (PNGAPI *png_read_status_ptr) PNGARG((png_structp, png_uint_32,
   int));
typedef void (PNGAPI *png_write_status_ptr) PNGARG((png_structp, png_uint_32,
   int));

#ifdef PNG_PROGRESSIVE_READ_SUPPORTED
typedef void (PNGAPI *png_progressive_info_ptr) PNGARG((png_structp,
   png_infop));
typedef void (PNGAPI *png_progressive_end_ptr) PNGARG((png_structp, png_infop));
typedef void (PNGAPI *png_progressive_row_ptr) PNGARG((png_structp, png_bytep,
   png_uint_32, int));
#endif

#if defined(PNG_READ_USER_TRANSFORM_SUPPORTED) || \
    defined(PNG_WRITE_USER_TRANSFORM_SUPPORTED)
typedef void (PNGAPI *png_user_transform_ptr) PNGARG((png_structp,
    png_row_infop, png_bytep));
#endif

#ifdef PNG_USER_CHUNKS_SUPPORTED
typedef int (PNGAPI *png_user_chunk_ptr) PNGARG((png_structp,
   png_unknown_chunkp));
#endif
#ifdef PNG_UNKNOWN_CHUNKS_SUPPORTED
typedef void (PNGAPI *png_unknown_chunk_ptr) PNGARG((png_structp));
#endif
#ifdef PNG_SETJMP_SUPPORTED
/* This must match the function definition in <setjmp.h>, and the
 * application must include this before png.h to obtain the definition
 * of jmp_buf.
 */
typedef void (PNGAPI *png_longjmp_ptr) PNGARG((jmp_buf, int));
#endif

/* Transform masks for the high-level interface */
#define PNG_TRANSFORM_IDENTITY       0x0000    /* read and write */
#define PNG_TRANSFORM_STRIP_16       0x0001    /* read only */
#define PNG_TRANSFORM_STRIP_ALPHA    0x0002    /* read only */
#define PNG_TRANSFORM_PACKING        0x0004    /* read and write */
#define PNG_TRANSFORM_PACKSWAP       0x0008    /* read and write */
#define PNG_TRANSFORM_EXPAND         0x0010    /* read only */
#define PNG_TRANSFORM_INVERT_MONO    0x0020    /* read and write */
#define PNG_TRANSFORM_SHIFT          0x0040    /* read and write */
#define PNG_TRANSFORM_BGR            0x0080    /* read and write */
#define PNG_TRANSFORM_SWAP_ALPHA     0x0100    /* read and write */
#define PNG_TRANSFORM_SWAP_ENDIAN    0x0200    /* read and write */
#define PNG_TRANSFORM_INVERT_ALPHA   0x0400    /* read and write */
#define PNG_TRANSFORM_STRIP_FILLER   0x0800    /* write only */
/* Added to libpng-1.2.34 */
#define PNG_TRANSFORM_STRIP_FILLER_BEFORE PNG_TRANSFORM_STRIP_FILLER
#define PNG_TRANSFORM_STRIP_FILLER_AFTER 0x1000 /* write only */
/* Added to libpng-1.4.0 */
#define PNG_TRANSFORM_GRAY_TO_RGB   0x2000      /* read only */

/* Flags for MNG supported features */
#define PNG_FLAG_MNG_EMPTY_PLTE     0x01
#define PNG_FLAG_MNG_FILTER_64      0x04
#define PNG_ALL_MNG_FEATURES        0x05

typedef png_voidp (*png_malloc_ptr) PNGARG((png_structp, png_alloc_size_t));
typedef void (*png_free_ptr) PNGARG((png_structp, png_voidp));

/* The structure that holds the information to read and write PNG files.
 * The only people who need to care about what is inside of this are the
 * people who will be modifying the library for their own special needs.
 * It should NOT be accessed directly by an application, except to store
 * the jmp_buf.
 */

struct png_struct_def
{
#ifdef PNG_SETJMP_SUPPORTED
   jmp_buf jmpbuf PNG_DEPSTRUCT;            /* used in png_error */
   png_longjmp_ptr longjmp_fn PNG_DEPSTRUCT;/* setjmp non-local goto
                                               function. */
#endif
   png_error_ptr error_fn PNG_DEPSTRUCT;    /* function for printing
                                               errors and aborting */
   png_error_ptr warning_fn PNG_DEPSTRUCT;  /* function for printing
                                               warnings */
   png_voidp error_ptr PNG_DEPSTRUCT;       /* user supplied struct for
                                               error functions */
   png_rw_ptr write_data_fn PNG_DEPSTRUCT;  /* function for writing
                                               output data */
   png_rw_ptr read_data_fn PNG_DEPSTRUCT;   /* function for reading
                                               input data */
   png_voidp io_ptr PNG_DEPSTRUCT;          /* ptr to application struct
                                               for I/O functions */

#ifdef PNG_READ_USER_TRANSFORM_SUPPORTED
   png_user_transform_ptr read_user_transform_fn PNG_DEPSTRUCT; /* user read
                                                                 transform */
#endif

#ifdef PNG_WRITE_USER_TRANSFORM_SUPPORTED
   png_user_transform_ptr write_user_transform_fn PNG_DEPSTRUCT; /* user write
                                                                  transform */
#endif

/* These were added in libpng-1.0.2 */
#ifdef PNG_USER_TRANSFORM_PTR_SUPPORTED
#if defined(PNG_READ_USER_TRANSFORM_SUPPORTED) || \
    defined(PNG_WRITE_USER_TRANSFORM_SUPPORTED)
   png_voidp user_transform_ptr PNG_DEPSTRUCT; /* user supplied struct
                                                  for user transform */
   png_byte user_transform_depth PNG_DEPSTRUCT;    /* bit depth of user
                                                      transformed pixels */
   png_byte user_transform_channels PNG_DEPSTRUCT; /* channels in user
                                                      transformed pixels */
#endif
#endif

   png_uint_32 mode PNG_DEPSTRUCT;          /* tells us where we are in
                                               the PNG file */
   png_uint_32 flags PNG_DEPSTRUCT;         /* flags indicating various
                                               things to libpng */
   png_uint_32 transformations PNG_DEPSTRUCT; /* which transformations
                                                 to perform */

   z_stream zstream PNG_DEPSTRUCT;          /* pointer to decompression
                                               structure (below) */
   png_bytep zbuf PNG_DEPSTRUCT;            /* buffer for zlib */
   png_size_t zbuf_size PNG_DEPSTRUCT;      /* size of zbuf */
   int zlib_level PNG_DEPSTRUCT;            /* holds zlib compression level */
   int zlib_method PNG_DEPSTRUCT;           /* holds zlib compression method */
   int zlib_window_bits PNG_DEPSTRUCT;      /* holds zlib compression window
                                               bits */
   int zlib_mem_level PNG_DEPSTRUCT;        /* holds zlib compression memory
                                               level */
   int zlib_strategy PNG_DEPSTRUCT;         /* holds zlib compression
                                               strategy */

   png_uint_32 width PNG_DEPSTRUCT;         /* width of image in pixels */
   png_uint_32 height PNG_DEPSTRUCT;        /* height of image in pixels */
   png_uint_32 num_rows PNG_DEPSTRUCT;      /* number of rows in current pass */
   png_uint_32 usr_width PNG_DEPSTRUCT;     /* width of row at start of write */
   png_size_t rowbytes PNG_DEPSTRUCT;       /* size of row in bytes */
#if 0 /* Replaced with the following in libpng-1.4.1 */
   png_size_t irowbytes PNG_DEPSTRUCT;
#endif
/* Added in libpng-1.4.1 */
#ifdef PNG_USER_LIMITS_SUPPORTED
   /* Total memory that a zTXt, sPLT, iTXt, iCCP, or unknown chunk
    * can occupy when decompressed.  0 means unlimited.
    * We will change the typedef from png_size_t to png_alloc_size_t
    * in libpng-1.6.0
    */
   png_alloc_size_t user_chunk_malloc_max PNG_DEPSTRUCT;
#endif
   png_uint_32 iwidth PNG_DEPSTRUCT;        /* width of current interlaced
                                               row in pixels */
   png_uint_32 row_number PNG_DEPSTRUCT;    /* current row in interlace pass */
   png_bytep prev_row PNG_DEPSTRUCT;        /* buffer to save previous
                                               (unfiltered) row */
   png_bytep row_buf PNG_DEPSTRUCT;         /* buffer to save current
                                               (unfiltered) row */
   png_bytep sub_row PNG_DEPSTRUCT;         /* buffer to save "sub" row
                                               when filtering */
   png_bytep up_row PNG_DEPSTRUCT;          /* buffer to save "up" row
                                               when filtering */
   png_bytep avg_row PNG_DEPSTRUCT;         /* buffer to save "avg" row
                                               when filtering */
   png_bytep paeth_row PNG_DEPSTRUCT;       /* buffer to save "Paeth" row
                                               when filtering */
   png_row_info row_info PNG_DEPSTRUCT;     /* used for transformation
                                               routines */

   png_uint_32 idat_size PNG_DEPSTRUCT;     /* current IDAT size for read */
   png_uint_32 crc PNG_DEPSTRUCT;           /* current chunk CRC value */
   png_colorp palette PNG_DEPSTRUCT;        /* palette from the input file */
   png_uint_16 num_palette PNG_DEPSTRUCT;   /* number of color entries in
                                               palette */
   png_uint_16 num_trans PNG_DEPSTRUCT;     /* number of transparency values */
   png_byte chunk_name[5] PNG_DEPSTRUCT;    /* null-terminated name of current
                                               chunk */
   png_byte compression PNG_DEPSTRUCT;      /* file compression type
                                               (always 0) */
   png_byte filter PNG_DEPSTRUCT;           /* file filter type (always 0) */
   png_byte interlaced PNG_DEPSTRUCT;       /* PNG_INTERLACE_NONE,
                                               PNG_INTERLACE_ADAM7 */
   png_byte pass PNG_DEPSTRUCT;             /* current interlace pass (0 - 6) */
   png_byte do_filter PNG_DEPSTRUCT;        /* row filter flags (see
                                               PNG_FILTER_ below ) */
   png_byte color_type PNG_DEPSTRUCT;       /* color type of file */
   png_byte bit_depth PNG_DEPSTRUCT;        /* bit depth of file */
   png_byte usr_bit_depth PNG_DEPSTRUCT;    /* bit depth of users row */
   png_byte pixel_depth PNG_DEPSTRUCT;      /* number of bits per pixel */
   png_byte channels PNG_DEPSTRUCT;         /* number of channels in file */
   png_byte usr_channels PNG_DEPSTRUCT;     /* channels at start of write */
   png_byte sig_bytes PNG_DEPSTRUCT;        /* magic bytes read/written from
                                               start of file */

#if defined(PNG_READ_FILLER_SUPPORTED) || defined(PNG_WRITE_FILLER_SUPPORTED)
   png_uint_16 filler PNG_DEPSTRUCT;           /* filler bytes for pixel
                                                  expansion */
#endif

#ifdef PNG_bKGD_SUPPORTED
   png_byte background_gamma_type PNG_DEPSTRUCT;
#  ifdef PNG_FLOATING_POINT_SUPPORTED
   float background_gamma PNG_DEPSTRUCT;
#  endif
   png_color_16 background PNG_DEPSTRUCT;   /* background color in
                                               screen gamma space */
#ifdef PNG_READ_GAMMA_SUPPORTED
   png_color_16 background_1 PNG_DEPSTRUCT; /* background normalized
                                               to gamma 1.0 */
#endif
#endif /* PNG_bKGD_SUPPORTED */

#ifdef PNG_WRITE_FLUSH_SUPPORTED
   png_flush_ptr output_flush_fn PNG_DEPSTRUCT; /* Function for flushing
                                               output */
   png_uint_32 flush_dist PNG_DEPSTRUCT;    /* how many rows apart to flush,
                                               0 - no flush */
   png_uint_32 flush_rows PNG_DEPSTRUCT;    /* number of rows written since
                                               last flush */
#endif

#if defined(PNG_READ_GAMMA_SUPPORTED) || defined(PNG_READ_BACKGROUND_SUPPORTED)
   int gamma_shift PNG_DEPSTRUCT;      /* number of "insignificant" bits
                                          16-bit gamma */
#ifdef PNG_FLOATING_POINT_SUPPORTED
   float gamma PNG_DEPSTRUCT;          /* file gamma value */
   float screen_gamma PNG_DEPSTRUCT;   /* screen gamma value
                                          (display_exponent) */
#endif
#endif

#if defined(PNG_READ_GAMMA_SUPPORTED) || defined(PNG_READ_BACKGROUND_SUPPORTED)
   png_bytep gamma_table PNG_DEPSTRUCT;     /* gamma table for 8-bit
                                               depth files */
   png_bytep gamma_from_1 PNG_DEPSTRUCT;    /* converts from 1.0 to screen */
   png_bytep gamma_to_1 PNG_DEPSTRUCT;      /* converts from file to 1.0 */
   png_uint_16pp gamma_16_table PNG_DEPSTRUCT; /* gamma table for 16-bit
                                                  depth files */
   png_uint_16pp gamma_16_from_1 PNG_DEPSTRUCT; /* converts from 1.0 to
                                                   screen */
   png_uint_16pp gamma_16_to_1 PNG_DEPSTRUCT; /* converts from file to 1.0 */
#endif

#if defined(PNG_READ_GAMMA_SUPPORTED) || defined(PNG_sBIT_SUPPORTED)
   png_color_8 sig_bit PNG_DEPSTRUCT;       /* significant bits in each
                                               available channel */
#endif

#if defined(PNG_READ_SHIFT_SUPPORTED) || defined(PNG_WRITE_SHIFT_SUPPORTED)
   png_color_8 shift PNG_DEPSTRUCT;         /* shift for significant bit
                                               tranformation */
#endif

#if defined(PNG_tRNS_SUPPORTED) || defined(PNG_READ_BACKGROUND_SUPPORTED) \
 || defined(PNG_READ_EXPAND_SUPPORTED) || defined(PNG_READ_BACKGROUND_SUPPORTED)
   png_bytep trans_alpha PNG_DEPSTRUCT;           /* alpha values for
                                                     paletted files */
   png_color_16 trans_color PNG_DEPSTRUCT;  /* transparent color for
                                               non-paletted files */
#endif

   png_read_status_ptr read_row_fn PNG_DEPSTRUCT;   /* called after each
                                                       row is decoded */
   png_write_status_ptr write_row_fn PNG_DEPSTRUCT; /* called after each
                                                       row is encoded */
#ifdef PNG_PROGRESSIVE_READ_SUPPORTED
   png_progressive_info_ptr info_fn PNG_DEPSTRUCT; /* called after header
                                                      data fully read */
   png_progressive_row_ptr row_fn PNG_DEPSTRUCT;   /* called after each
                                                      prog. row is decoded */
   png_progressive_end_ptr end_fn PNG_DEPSTRUCT;   /* called after image
                                                      is complete */
   png_bytep save_buffer_ptr PNG_DEPSTRUCT;        /* current location in
                                                      save_buffer */
   png_bytep save_buffer PNG_DEPSTRUCT;            /* buffer for previously
                                                      read data */
   png_bytep current_buffer_ptr PNG_DEPSTRUCT;     /* current location in
                                                      current_buffer */
   png_bytep current_buffer PNG_DEPSTRUCT;         /* buffer for recently
                                                      used data */
   png_uint_32 push_length PNG_DEPSTRUCT;          /* size of current input
                                                      chunk */
   png_uint_32 skip_length PNG_DEPSTRUCT;          /* bytes to skip in
                                                      input data */
   png_size_t save_buffer_size PNG_DEPSTRUCT;      /* amount of data now
                                                      in save_buffer */
   png_size_t save_buffer_max PNG_DEPSTRUCT;       /* total size of
                                                      save_buffer */
   png_size_t buffer_size PNG_DEPSTRUCT;           /* total amount of
                                                      available input data */
   png_size_t current_buffer_size PNG_DEPSTRUCT;   /* amount of data now
                                                      in current_buffer */
   int process_mode PNG_DEPSTRUCT;                 /* what push library
                                                      is currently doing */
   int cur_palette PNG_DEPSTRUCT;                  /* current push library
                                                      palette index */

#  ifdef PNG_TEXT_SUPPORTED
     png_size_t current_text_size PNG_DEPSTRUCT;   /* current size of
                                                      text input data */
     png_size_t current_text_left PNG_DEPSTRUCT;   /* how much text left
                                                      to read in input */
     png_charp current_text PNG_DEPSTRUCT;         /* current text chunk
                                                      buffer */
     png_charp current_text_ptr PNG_DEPSTRUCT;     /* current location
                                                      in current_text */
#  endif /* PNG_PROGRESSIVE_READ_SUPPORTED && PNG_TEXT_SUPPORTED */

#endif /* PNG_PROGRESSIVE_READ_SUPPORTED */

#if defined(__TURBOC__) && !defined(_Windows) && !defined(__FLAT__)
/* For the Borland special 64K segment handler */
   png_bytepp offset_table_ptr PNG_DEPSTRUCT;
   png_bytep offset_table PNG_DEPSTRUCT;
   png_uint_16 offset_table_number PNG_DEPSTRUCT;
   png_uint_16 offset_table_count PNG_DEPSTRUCT;
   png_uint_16 offset_table_count_free PNG_DEPSTRUCT;
#endif

#ifdef PNG_READ_QUANTIZE_SUPPORTED
   png_bytep palette_lookup PNG_DEPSTRUCT; /* lookup table for quantizing */
   png_bytep quantize_index PNG_DEPSTRUCT; /* index translation for palette
                                              files */
#endif

#if defined(PNG_READ_QUANTIZE_SUPPORTED) || defined(PNG_hIST_SUPPORTED)
   png_uint_16p hist PNG_DEPSTRUCT;                /* histogram */
#endif

#ifdef PNG_WRITE_WEIGHTED_FILTER_SUPPORTED
   png_byte heuristic_method PNG_DEPSTRUCT;        /* heuristic for row
                                                      filter selection */
   png_byte num_prev_filters PNG_DEPSTRUCT;        /* number of weights
                                                      for previous rows */
   png_bytep prev_filters PNG_DEPSTRUCT;           /* filter type(s) of
                                                      previous row(s) */
   png_uint_16p filter_weights PNG_DEPSTRUCT;      /* weight(s) for previous
                                                      line(s) */
   png_uint_16p inv_filter_weights PNG_DEPSTRUCT;  /* 1/weight(s) for
                                                      previous line(s) */
   png_uint_16p filter_costs PNG_DEPSTRUCT;        /* relative filter
                                                      calculation cost */
   png_uint_16p inv_filter_costs PNG_DEPSTRUCT;    /* 1/relative filter
                                                      calculation cost */
#endif

#ifdef PNG_TIME_RFC1123_SUPPORTED
   png_charp time_buffer PNG_DEPSTRUCT; /* String to hold RFC 1123 time text */
#endif

/* New members added in libpng-1.0.6 */

   png_uint_32 free_me PNG_DEPSTRUCT;    /* flags items libpng is
                                            responsible for freeing */

#ifdef PNG_USER_CHUNKS_SUPPORTED
   png_voidp user_chunk_ptr PNG_DEPSTRUCT;
   png_user_chunk_ptr read_user_chunk_fn PNG_DEPSTRUCT; /* user read
                                                           chunk handler */
#endif

#ifdef PNG_HANDLE_AS_UNKNOWN_SUPPORTED
   int num_chunk_list PNG_DEPSTRUCT;
   png_bytep chunk_list PNG_DEPSTRUCT;
#endif

/* New members added in libpng-1.0.3 */
#ifdef PNG_READ_RGB_TO_GRAY_SUPPORTED
   png_byte rgb_to_gray_status PNG_DEPSTRUCT;
   /* These were changed from png_byte in libpng-1.0.6 */
   png_uint_16 rgb_to_gray_red_coeff PNG_DEPSTRUCT;
   png_uint_16 rgb_to_gray_green_coeff PNG_DEPSTRUCT;
   png_uint_16 rgb_to_gray_blue_coeff PNG_DEPSTRUCT;
#endif

/* New member added in libpng-1.0.4 (renamed in 1.0.9) */
#if defined(PNG_MNG_FEATURES_SUPPORTED) || \
    defined(PNG_READ_EMPTY_PLTE_SUPPORTED) || \
    defined(PNG_WRITE_EMPTY_PLTE_SUPPORTED)
/* Changed from png_byte to png_uint_32 at version 1.2.0 */
   png_uint_32 mng_features_permitted PNG_DEPSTRUCT;
#endif

/* New member added in libpng-1.0.7 */
#if defined(PNG_READ_GAMMA_SUPPORTED) || defined(PNG_READ_BACKGROUND_SUPPORTED)
   png_fixed_point int_gamma PNG_DEPSTRUCT;
#endif

/* New member added in libpng-1.0.9, ifdef'ed out in 1.0.12, enabled in 1.2.0 */
#ifdef PNG_MNG_FEATURES_SUPPORTED
   png_byte filter_type PNG_DEPSTRUCT;
#endif

/* New members added in libpng-1.2.0 */

/* New members added in libpng-1.0.2 but first enabled by default in 1.2.0 */
#ifdef PNG_USER_MEM_SUPPORTED
   png_voidp mem_ptr PNG_DEPSTRUCT;             /* user supplied struct for
                                                   mem functions */
   png_malloc_ptr malloc_fn PNG_DEPSTRUCT;      /* function for
                                                   allocating memory */
   png_free_ptr free_fn PNG_DEPSTRUCT;          /* function for
                                                   freeing memory */
#endif

/* New member added in libpng-1.0.13 and 1.2.0 */
   png_bytep big_row_buf PNG_DEPSTRUCT;         /* buffer to save current
                                                   (unfiltered) row */

#ifdef PNG_READ_QUANTIZE_SUPPORTED
/* The following three members were added at version 1.0.14 and 1.2.4 */
   png_bytep quantize_sort PNG_DEPSTRUCT;          /* working sort array */
   png_bytep index_to_palette PNG_DEPSTRUCT;       /* where the original
                                                     index currently is
                                                     in the palette */
   png_bytep palette_to_index PNG_DEPSTRUCT;       /* which original index
                                                      points to this
                                                      palette color */
#endif

/* New members added in libpng-1.0.16 and 1.2.6 */
   png_byte compression_type PNG_DEPSTRUCT;

#ifdef PNG_USER_LIMITS_SUPPORTED
   png_uint_32 user_width_max PNG_DEPSTRUCT;
   png_uint_32 user_height_max PNG_DEPSTRUCT;
   /* Added in libpng-1.4.0: Total number of sPLT, text, and unknown
    * chunks that can be stored (0 means unlimited).
    */
   png_uint_32 user_chunk_cache_max PNG_DEPSTRUCT;
#endif

/* New member added in libpng-1.0.25 and 1.2.17 */
#ifdef PNG_UNKNOWN_CHUNKS_SUPPORTED
   /* Storage for unknown chunk that the library doesn't recognize. */
   png_unknown_chunk unknown_chunk PNG_DEPSTRUCT;
#endif

/* New members added in libpng-1.2.26 */
  png_uint_32 old_big_row_buf_size PNG_DEPSTRUCT;
  png_uint_32 old_prev_row_size PNG_DEPSTRUCT;

/* New member added in libpng-1.2.30 */
  png_charp chunkdata PNG_DEPSTRUCT;  /* buffer for reading chunk data */

#ifdef PNG_IO_STATE_SUPPORTED
/* New member added in libpng-1.4.0 */
   png_uint_32 io_state PNG_DEPSTRUCT;
#endif
};


/* This triggers a compiler error in png.c, if png.c and png.h
 * do not agree upon the version number.
 */
typedef png_structp version_1_4_3;

typedef png_struct FAR * FAR * png_structpp;

/* Here are the function definitions most commonly used.  This is not
 * the place to find out how to use libpng.  See libpng.txt for the
 * full explanation, see example.c for the summary.  This just provides
 * a simple one line description of the use of each function.
 */

/* Returns the version number of the library */
extern PNG_EXPORT(png_uint_32,png_access_version_number) PNGARG((void));

/* Tell lib we have already handled the first <num_bytes> magic bytes.
 * Handling more than 8 bytes from the beginning of the file is an error.
 */
extern PNG_EXPORT(void,png_set_sig_bytes) PNGARG((png_structp png_ptr,
   int num_bytes));

/* Check sig[start] through sig[start + num_to_check - 1] to see if it's a
 * PNG file.  Returns zero if the supplied bytes match the 8-byte PNG
 * signature, and non-zero otherwise.  Having num_to_check == 0 or
 * start > 7 will always fail (ie return non-zero).
 */
extern PNG_EXPORT(int,png_sig_cmp) PNGARG((png_bytep sig, png_size_t start,
   png_size_t num_to_check));

/* Simple signature checking function.  This is the same as calling
 * png_check_sig(sig, n) := !png_sig_cmp(sig, 0, n).
 */
#define png_check_sig(sig,n) !png_sig_cmp((sig), 0, (n))

/* Allocate and initialize png_ptr struct for reading, and any other memory. */
extern PNG_EXPORT(png_structp,png_create_read_struct)
   PNGARG((png_const_charp user_png_ver, png_voidp error_ptr,
   png_error_ptr error_fn, png_error_ptr warn_fn)) PNG_ALLOCATED;

/* Allocate and initialize png_ptr struct for writing, and any other memory */
extern PNG_EXPORT(png_structp,png_create_write_struct)
   PNGARG((png_const_charp user_png_ver, png_voidp error_ptr,
   png_error_ptr error_fn, png_error_ptr warn_fn)) PNG_ALLOCATED;

extern PNG_EXPORT(png_size_t,png_get_compression_buffer_size)
   PNGARG((png_structp png_ptr));

extern PNG_EXPORT(void,png_set_compression_buffer_size)
   PNGARG((png_structp png_ptr, png_size_t size));

/* Moved from pngconf.h in 1.4.0 and modified to ensure setjmp/longjmp
 * match up.
 */
#ifdef PNG_SETJMP_SUPPORTED
/* This function returns the jmp_buf built in to *png_ptr.  It must be
 * supplied with an appropriate 'longjmp' function to use on that jmp_buf
 * unless the default error function is overridden in which case NULL is
 * acceptable.  The size of the jmp_buf is checked against the actual size
 * allocated by the library - the call will return NULL on a mismatch
 * indicating an ABI mismatch.
 */
extern PNG_EXPORT(jmp_buf*, png_set_longjmp_fn)
   PNGARG((png_structp png_ptr, png_longjmp_ptr longjmp_fn, size_t
       jmp_buf_size));
#  define png_jmpbuf(png_ptr) \
   (*png_set_longjmp_fn((png_ptr), longjmp, sizeof (jmp_buf)))
#else
#  define png_jmpbuf(png_ptr) \
   (LIBPNG_WAS_COMPILED_WITH__PNG_NO_SETJMP)
#endif

#ifdef PNG_READ_SUPPORTED
/* Reset the compression stream */
extern PNG_EXPORT(int,png_reset_zstream) PNGARG((png_structp png_ptr));
#endif

/* New functions added in libpng-1.0.2 (not enabled by default until 1.2.0) */
#ifdef PNG_USER_MEM_SUPPORTED
extern PNG_EXPORT(png_structp,png_create_read_struct_2)
   PNGARG((png_const_charp user_png_ver, png_voidp error_ptr,
   png_error_ptr error_fn, png_error_ptr warn_fn, png_voidp mem_ptr,
   png_malloc_ptr malloc_fn, png_free_ptr free_fn)) PNG_ALLOCATED;
extern PNG_EXPORT(png_structp,png_create_write_struct_2)
   PNGARG((png_const_charp user_png_ver, png_voidp error_ptr,
   png_error_ptr error_fn, png_error_ptr warn_fn, png_voidp mem_ptr,
   png_malloc_ptr malloc_fn, png_free_ptr free_fn)) PNG_ALLOCATED;
#endif

/* Write the PNG file signature. */
extern PNG_EXPORT(void,png_write_sig) PNGARG((png_structp png_ptr));

/* Write a PNG chunk - size, type, (optional) data, CRC. */
extern PNG_EXPORT(void,png_write_chunk) PNGARG((png_structp png_ptr,
   png_bytep chunk_name, png_bytep data, png_size_t length));

/* Write the start of a PNG chunk - length and chunk name. */
extern PNG_EXPORT(void,png_write_chunk_start) PNGARG((png_structp png_ptr,
   png_bytep chunk_name, png_uint_32 length));

/* Write the data of a PNG chunk started with png_write_chunk_start(). */
extern PNG_EXPORT(void,png_write_chunk_data) PNGARG((png_structp png_ptr,
   png_bytep data, png_size_t length));

/* Finish a chunk started with png_write_chunk_start() (includes CRC). */
extern PNG_EXPORT(void,png_write_chunk_end) PNGARG((png_structp png_ptr));

/* Allocate and initialize the info structure */
extern PNG_EXPORT(png_infop,png_create_info_struct)
   PNGARG((png_structp png_ptr)) PNG_ALLOCATED;

extern PNG_EXPORT(void,png_info_init_3) PNGARG((png_infopp info_ptr,
    png_size_t png_info_struct_size));

/* Writes all the PNG information before the image. */
extern PNG_EXPORT(void,png_write_info_before_PLTE) PNGARG((png_structp png_ptr,
   png_infop info_ptr));
extern PNG_EXPORT(void,png_write_info) PNGARG((png_structp png_ptr,
   png_infop info_ptr));

#ifdef PNG_SEQUENTIAL_READ_SUPPORTED
/* Read the information before the actual image data. */
extern PNG_EXPORT(void,png_read_info) PNGARG((png_structp png_ptr,
   png_infop info_ptr));
#endif

#ifdef PNG_TIME_RFC1123_SUPPORTED
extern PNG_EXPORT(png_charp,png_convert_to_rfc1123)
   PNGARG((png_structp png_ptr, png_timep ptime));
#endif

#ifdef PNG_CONVERT_tIME_SUPPORTED
/* Convert from a struct tm to png_time */
extern PNG_EXPORT(void,png_convert_from_struct_tm) PNGARG((png_timep ptime,
   struct tm FAR * ttime));

/* Convert from time_t to png_time.  Uses gmtime() */
extern PNG_EXPORT(void,png_convert_from_time_t) PNGARG((png_timep ptime,
   time_t ttime));
#endif /* PNG_CONVERT_tIME_SUPPORTED */

#ifdef PNG_READ_EXPAND_SUPPORTED
/* Expand data to 24-bit RGB, or 8-bit grayscale, with alpha if available. */
extern PNG_EXPORT(void,png_set_expand) PNGARG((png_structp png_ptr));
extern PNG_EXPORT(void,png_set_expand_gray_1_2_4_to_8) PNGARG((png_structp
  png_ptr));
extern PNG_EXPORT(void,png_set_palette_to_rgb) PNGARG((png_structp png_ptr));
extern PNG_EXPORT(void,png_set_tRNS_to_alpha) PNGARG((png_structp png_ptr));
#endif

#if defined(PNG_READ_BGR_SUPPORTED) || defined(PNG_WRITE_BGR_SUPPORTED)
/* Use blue, green, red order for pixels. */
extern PNG_EXPORT(void,png_set_bgr) PNGARG((png_structp png_ptr));
#endif

#ifdef PNG_READ_GRAY_TO_RGB_SUPPORTED
/* Expand the grayscale to 24-bit RGB if necessary. */
extern PNG_EXPORT(void,png_set_gray_to_rgb) PNGARG((png_structp png_ptr));
#endif

#ifdef PNG_READ_RGB_TO_GRAY_SUPPORTED
/* Reduce RGB to grayscale. */
#ifdef PNG_FLOATING_POINT_SUPPORTED
extern PNG_EXPORT(void,png_set_rgb_to_gray) PNGARG((png_structp png_ptr,
   int error_action, double red, double green ));
#endif
extern PNG_EXPORT(void,png_set_rgb_to_gray_fixed) PNGARG((png_structp png_ptr,
   int error_action, png_fixed_point red, png_fixed_point green ));
extern PNG_EXPORT(png_byte,png_get_rgb_to_gray_status) PNGARG((png_structp
   png_ptr));
#endif

extern PNG_EXPORT(void,png_build_grayscale_palette) PNGARG((int bit_depth,
   png_colorp palette));

#ifdef PNG_READ_STRIP_ALPHA_SUPPORTED
extern PNG_EXPORT(void,png_set_strip_alpha) PNGARG((png_structp png_ptr));
#endif

#if defined(PNG_READ_SWAP_ALPHA_SUPPORTED) || \
    defined(PNG_WRITE_SWAP_ALPHA_SUPPORTED)
extern PNG_EXPORT(void,png_set_swap_alpha) PNGARG((png_structp png_ptr));
#endif

#if defined(PNG_READ_INVERT_ALPHA_SUPPORTED) || \
    defined(PNG_WRITE_INVERT_ALPHA_SUPPORTED)
extern PNG_EXPORT(void,png_set_invert_alpha) PNGARG((png_structp png_ptr));
#endif

#if defined(PNG_READ_FILLER_SUPPORTED) || defined(PNG_WRITE_FILLER_SUPPORTED)
/* Add a filler byte to 8-bit Gray or 24-bit RGB images. */
extern PNG_EXPORT(void,png_set_filler) PNGARG((png_structp png_ptr,
   png_uint_32 filler, int flags));
/* The values of the PNG_FILLER_ defines should NOT be changed */
#define PNG_FILLER_BEFORE 0
#define PNG_FILLER_AFTER 1
/* Add an alpha byte to 8-bit Gray or 24-bit RGB images. */
extern PNG_EXPORT(void,png_set_add_alpha) PNGARG((png_structp png_ptr,
   png_uint_32 filler, int flags));
#endif /* PNG_READ_FILLER_SUPPORTED || PNG_WRITE_FILLER_SUPPORTED */

#if defined(PNG_READ_SWAP_SUPPORTED) || defined(PNG_WRITE_SWAP_SUPPORTED)
/* Swap bytes in 16-bit depth files. */
extern PNG_EXPORT(void,png_set_swap) PNGARG((png_structp png_ptr));
#endif

#if defined(PNG_READ_PACK_SUPPORTED) || defined(PNG_WRITE_PACK_SUPPORTED)
/* Use 1 byte per pixel in 1, 2, or 4-bit depth files. */
extern PNG_EXPORT(void,png_set_packing) PNGARG((png_structp png_ptr));
#endif

#if defined(PNG_READ_PACKSWAP_SUPPORTED) || \
    defined(PNG_WRITE_PACKSWAP_SUPPORTED)
/* Swap packing order of pixels in bytes. */
extern PNG_EXPORT(void,png_set_packswap) PNGARG((png_structp png_ptr));
#endif

#if defined(PNG_READ_SHIFT_SUPPORTED) || defined(PNG_WRITE_SHIFT_SUPPORTED)
/* Converts files to legal bit depths. */
extern PNG_EXPORT(void,png_set_shift) PNGARG((png_structp png_ptr,
   png_color_8p true_bits));
#endif

#if defined(PNG_READ_INTERLACING_SUPPORTED) || \
    defined(PNG_WRITE_INTERLACING_SUPPORTED)
/* Have the code handle the interlacing.  Returns the number of passes. */
extern PNG_EXPORT(int,png_set_interlace_handling) PNGARG((png_structp png_ptr));
#endif

#if defined(PNG_READ_INVERT_SUPPORTED) || defined(PNG_WRITE_INVERT_SUPPORTED)
/* Invert monochrome files */
extern PNG_EXPORT(void,png_set_invert_mono) PNGARG((png_structp png_ptr));
#endif

#ifdef PNG_READ_BACKGROUND_SUPPORTED
/* Handle alpha and tRNS by replacing with a background color. */
#ifdef PNG_FLOATING_POINT_SUPPORTED
extern PNG_EXPORT(void,png_set_background) PNGARG((png_structp png_ptr,
   png_color_16p background_color, int background_gamma_code,
   int need_expand, double background_gamma));
#endif
#define PNG_BACKGROUND_GAMMA_UNKNOWN 0
#define PNG_BACKGROUND_GAMMA_SCREEN  1
#define PNG_BACKGROUND_GAMMA_FILE    2
#define PNG_BACKGROUND_GAMMA_UNIQUE  3
#endif

#ifdef PNG_READ_16_TO_8_SUPPORTED
/* Strip the second byte of information from a 16-bit depth file. */
extern PNG_EXPORT(void,png_set_strip_16) PNGARG((png_structp png_ptr));
#endif

#ifdef PNG_READ_QUANTIZE_SUPPORTED
/* Turn on quantizing, and reduce the palette to the number of colors
 * available.  Prior to libpng-1.4.2, this was png_set_dither().
 */
extern PNG_EXPORT(void,png_set_quantize) PNGARG((png_structp png_ptr,
   png_colorp palette, int num_palette, int maximum_colors,
   png_uint_16p histogram, int full_quantize));
#endif
/* This migration aid will be removed from libpng-1.5.0 */
#define png_set_dither png_set_quantize

#ifdef PNG_READ_GAMMA_SUPPORTED
/* Handle gamma correction. Screen_gamma=(display_exponent) */
#ifdef PNG_FLOATING_POINT_SUPPORTED
extern PNG_EXPORT(void,png_set_gamma) PNGARG((png_structp png_ptr,
   double screen_gamma, double default_file_gamma));
#endif
#endif


#ifdef PNG_WRITE_FLUSH_SUPPORTED
/* Set how many lines between output flushes - 0 for no flushing */
extern PNG_EXPORT(void,png_set_flush) PNGARG((png_structp png_ptr, int nrows));
/* Flush the current PNG output buffer */
extern PNG_EXPORT(void,png_write_flush) PNGARG((png_structp png_ptr));
#endif

/* Optional update palette with requested transformations */
extern PNG_EXPORT(void,png_start_read_image) PNGARG((png_structp png_ptr));

/* Optional call to update the users info structure */
extern PNG_EXPORT(void,png_read_update_info) PNGARG((png_structp png_ptr,
   png_infop info_ptr));

#ifdef PNG_SEQUENTIAL_READ_SUPPORTED
/* Read one or more rows of image data. */
extern PNG_EXPORT(void,png_read_rows) PNGARG((png_structp png_ptr,
   png_bytepp row, png_bytepp display_row, png_uint_32 num_rows));
#endif

#ifdef PNG_SEQUENTIAL_READ_SUPPORTED
/* Read a row of data. */
extern PNG_EXPORT(void,png_read_row) PNGARG((png_structp png_ptr,
   png_bytep row,
   png_bytep display_row));
#endif

#ifdef PNG_SEQUENTIAL_READ_SUPPORTED
/* Read the whole image into memory at once. */
extern PNG_EXPORT(void,png_read_image) PNGARG((png_structp png_ptr,
   png_bytepp image));
#endif

/* Write a row of image data */
extern PNG_EXPORT(void,png_write_row) PNGARG((png_structp png_ptr,
   png_bytep row));

/* Write a few rows of image data */
extern PNG_EXPORT(void,png_write_rows) PNGARG((png_structp png_ptr,
   png_bytepp row, png_uint_32 num_rows));

/* Write the image data */
extern PNG_EXPORT(void,png_write_image) PNGARG((png_structp png_ptr,
   png_bytepp image));

/* Write the end of the PNG file. */
extern PNG_EXPORT(void,png_write_end) PNGARG((png_structp png_ptr,
   png_infop info_ptr));

#ifdef PNG_SEQUENTIAL_READ_SUPPORTED
/* Read the end of the PNG file. */
extern PNG_EXPORT(void,png_read_end) PNGARG((png_structp png_ptr,
   png_infop info_ptr));
#endif

/* Free any memory associated with the png_info_struct */
extern PNG_EXPORT(void,png_destroy_info_struct) PNGARG((png_structp png_ptr,
   png_infopp info_ptr_ptr));

/* Free any memory associated with the png_struct and the png_info_structs */
extern PNG_EXPORT(void,png_destroy_read_struct) PNGARG((png_structpp
   png_ptr_ptr, png_infopp info_ptr_ptr, png_infopp end_info_ptr_ptr));

/* Free any memory associated with the png_struct and the png_info_structs */
extern PNG_EXPORT(void,png_destroy_write_struct)
   PNGARG((png_structpp png_ptr_ptr, png_infopp info_ptr_ptr));

/* Set the libpng method of handling chunk CRC errors */
extern PNG_EXPORT(void,png_set_crc_action) PNGARG((png_structp png_ptr,
   int crit_action, int ancil_action));

/* Values for png_set_crc_action() to say how to handle CRC errors in
 * ancillary and critical chunks, and whether to use the data contained
 * therein.  Note that it is impossible to "discard" data in a critical
 * chunk.  For versions prior to 0.90, the action was always error/quit,
 * whereas in version 0.90 and later, the action for CRC errors in ancillary
 * chunks is warn/discard.  These values should NOT be changed.
 *
 *      value                       action:critical     action:ancillary
 */
#define PNG_CRC_DEFAULT       0  /* error/quit          warn/discard data */
#define PNG_CRC_ERROR_QUIT    1  /* error/quit          error/quit        */
#define PNG_CRC_WARN_DISCARD  2  /* (INVALID)           warn/discard data */
#define PNG_CRC_WARN_USE      3  /* warn/use data       warn/use data     */
#define PNG_CRC_QUIET_USE     4  /* quiet/use data      quiet/use data    */
#define PNG_CRC_NO_CHANGE     5  /* use current value   use current value */

/* These functions give the user control over the scan-line filtering in
 * libpng and the compression methods used by zlib.  These functions are
 * mainly useful for testing, as the defaults should work with most users.
 * Those users who are tight on memory or want faster performance at the
 * expense of compression can modify them.  See the compression library
 * header file (zlib.h) for an explination of the compression functions.
 */

/* Set the filtering method(s) used by libpng.  Currently, the only valid
 * value for "method" is 0.
 */
extern PNG_EXPORT(void,png_set_filter) PNGARG((png_structp png_ptr, int method,
   int filters));

/* Flags for png_set_filter() to say which filters to use.  The flags
 * are chosen so that they don't conflict with real filter types
 * below, in case they are supplied instead of the #defined constants.
 * These values should NOT be changed.
 */
#define PNG_NO_FILTERS     0x00
#define PNG_FILTER_NONE    0x08
#define PNG_FILTER_SUB     0x10
#define PNG_FILTER_UP      0x20
#define PNG_FILTER_AVG     0x40
#define PNG_FILTER_PAETH   0x80
#define PNG_ALL_FILTERS (PNG_FILTER_NONE | PNG_FILTER_SUB | PNG_FILTER_UP | \
                         PNG_FILTER_AVG | PNG_FILTER_PAETH)

/* Filter values (not flags) - used in pngwrite.c, pngwutil.c for now.
 * These defines should NOT be changed.
 */
#define PNG_FILTER_VALUE_NONE  0
#define PNG_FILTER_VALUE_SUB   1
#define PNG_FILTER_VALUE_UP    2
#define PNG_FILTER_VALUE_AVG   3
#define PNG_FILTER_VALUE_PAETH 4
#define PNG_FILTER_VALUE_LAST  5

#ifdef PNG_WRITE_WEIGHTED_FILTER_SUPPORTED /* EXPERIMENTAL */
/* The "heuristic_method" is given by one of the PNG_FILTER_HEURISTIC_
 * defines, either the default (minimum-sum-of-absolute-differences), or
 * the experimental method (weighted-minimum-sum-of-absolute-differences).
 *
 * Weights are factors >= 1.0, indicating how important it is to keep the
 * filter type consistent between rows.  Larger numbers mean the current
 * filter is that many times as likely to be the same as the "num_weights"
 * previous filters.  This is cumulative for each previous row with a weight.
 * There needs to be "num_weights" values in "filter_weights", or it can be
 * NULL if the weights aren't being specified.  Weights have no influence on
 * the selection of the first row filter.  Well chosen weights can (in theory)
 * improve the compression for a given image.
 *
 * Costs are factors >= 1.0 indicating the relative decoding costs of a
 * filter type.  Higher costs indicate more decoding expense, and are
 * therefore less likely to be selected over a filter with lower computational
 * costs.  There needs to be a value in "filter_costs" for each valid filter
 * type (given by PNG_FILTER_VALUE_LAST), or it can be NULL if you aren't
 * setting the costs.  Costs try to improve the speed of decompression without
 * unduly increasing the compressed image size.
 *
 * A negative weight or cost indicates the default value is to be used, and
 * values in the range [0.0, 1.0) indicate the value is to remain unchanged.
 * The default values for both weights and costs are currently 1.0, but may
 * change if good general weighting/cost heuristics can be found.  If both
 * the weights and costs are set to 1.0, this degenerates the WEIGHTED method
 * to the UNWEIGHTED method, but with added encoding time/computation.
 */
#ifdef PNG_FLOATING_POINT_SUPPORTED
extern PNG_EXPORT(void,png_set_filter_heuristics) PNGARG((png_structp png_ptr,
   int heuristic_method, int num_weights, png_doublep filter_weights,
   png_doublep filter_costs));
#endif
#endif /*  PNG_WRITE_WEIGHTED_FILTER_SUPPORTED */

/* Heuristic used for row filter selection.  These defines should NOT be
 * changed.
 */
#define PNG_FILTER_HEURISTIC_DEFAULT    0  /* Currently "UNWEIGHTED" */
#define PNG_FILTER_HEURISTIC_UNWEIGHTED 1  /* Used by libpng < 0.95 */
#define PNG_FILTER_HEURISTIC_WEIGHTED   2  /* Experimental feature */
#define PNG_FILTER_HEURISTIC_LAST       3  /* Not a valid value */

/* Set the library compression level.  Currently, valid values range from
 * 0 - 9, corresponding directly to the zlib compression levels 0 - 9
 * (0 - no compression, 9 - "maximal" compression).  Note that tests have
 * shown that zlib compression levels 3-6 usually perform as well as level 9
 * for PNG images, and do considerably fewer caclulations.  In the future,
 * these values may not correspond directly to the zlib compression levels.
 */
extern PNG_EXPORT(void,png_set_compression_level) PNGARG((png_structp png_ptr,
   int level));

extern PNG_EXPORT(void,png_set_compression_mem_level)
   PNGARG((png_structp png_ptr, int mem_level));

extern PNG_EXPORT(void,png_set_compression_strategy)
   PNGARG((png_structp png_ptr, int strategy));

extern PNG_EXPORT(void,png_set_compression_window_bits)
   PNGARG((png_structp png_ptr, int window_bits));

extern PNG_EXPORT(void,png_set_compression_method) PNGARG((png_structp png_ptr,
   int method));

/* These next functions are called for input/output, memory, and error
 * handling.  They are in the file pngrio.c, pngwio.c, and pngerror.c,
 * and call standard C I/O routines such as fread(), fwrite(), and
 * fprintf().  These functions can be made to use other I/O routines
 * at run time for those applications that need to handle I/O in a
 * different manner by calling png_set_???_fn().  See libpng.txt for
 * more information.
 */

#ifdef PNG_STDIO_SUPPORTED
/* Initialize the input/output for the PNG file to the default functions. */
extern PNG_EXPORT(void,png_init_io) PNGARG((png_structp png_ptr,
    png_FILE_p fp));
#endif

/* Replace the (error and abort), and warning functions with user
 * supplied functions.  If no messages are to be printed you must still
 * write and use replacement functions. The replacement error_fn should
 * still do a longjmp to the last setjmp location if you are using this
 * method of error handling.  If error_fn or warning_fn is NULL, the
 * default function will be used.
 */

extern PNG_EXPORT(void,png_set_error_fn) PNGARG((png_structp png_ptr,
   png_voidp error_ptr, png_error_ptr error_fn, png_error_ptr warning_fn));

/* Return the user pointer associated with the error functions */
extern PNG_EXPORT(png_voidp,png_get_error_ptr) PNGARG((png_structp png_ptr));

/* Replace the default data output functions with a user supplied one(s).
 * If buffered output is not used, then output_flush_fn can be set to NULL.
 * If PNG_WRITE_FLUSH_SUPPORTED is not defined at libpng compile time
 * output_flush_fn will be ignored (and thus can be NULL).
 * It is probably a mistake to use NULL for output_flush_fn if
 * write_data_fn is not also NULL unless you have built libpng with
 * PNG_WRITE_FLUSH_SUPPORTED undefined, because in this case libpng's
 * default flush function, which uses the standard *FILE structure, will
 * be used.
 */
extern PNG_EXPORT(void,png_set_write_fn) PNGARG((png_structp png_ptr,
   png_voidp io_ptr, png_rw_ptr write_data_fn, png_flush_ptr output_flush_fn));

/* Replace the default data input function with a user supplied one. */
extern PNG_EXPORT(void,png_set_read_fn) PNGARG((png_structp png_ptr,
   png_voidp io_ptr, png_rw_ptr read_data_fn));

/* Return the user pointer associated with the I/O functions */
extern PNG_EXPORT(png_voidp,png_get_io_ptr) PNGARG((png_structp png_ptr));

extern PNG_EXPORT(void,png_set_read_status_fn) PNGARG((png_structp png_ptr,
   png_read_status_ptr read_row_fn));

extern PNG_EXPORT(void,png_set_write_status_fn) PNGARG((png_structp png_ptr,
   png_write_status_ptr write_row_fn));

#ifdef PNG_USER_MEM_SUPPORTED
/* Replace the default memory allocation functions with user supplied one(s). */
extern PNG_EXPORT(void,png_set_mem_fn) PNGARG((png_structp png_ptr,
   png_voidp mem_ptr, png_malloc_ptr malloc_fn, png_free_ptr free_fn));
/* Return the user pointer associated with the memory functions */
extern PNG_EXPORT(png_voidp,png_get_mem_ptr) PNGARG((png_structp png_ptr));
#endif

#ifdef PNG_READ_USER_TRANSFORM_SUPPORTED
extern PNG_EXPORT(void,png_set_read_user_transform_fn) PNGARG((png_structp
   png_ptr, png_user_transform_ptr read_user_transform_fn));
#endif

#ifdef PNG_WRITE_USER_TRANSFORM_SUPPORTED
extern PNG_EXPORT(void,png_set_write_user_transform_fn) PNGARG((png_structp
   png_ptr, png_user_transform_ptr write_user_transform_fn));
#endif

#if defined(PNG_READ_USER_TRANSFORM_SUPPORTED) || \
    defined(PNG_WRITE_USER_TRANSFORM_SUPPORTED)
extern PNG_EXPORT(void,png_set_user_transform_info) PNGARG((png_structp
   png_ptr, png_voidp user_transform_ptr, int user_transform_depth,
   int user_transform_channels));
/* Return the user pointer associated with the user transform functions */
extern PNG_EXPORT(png_voidp,png_get_user_transform_ptr)
   PNGARG((png_structp png_ptr));
#endif

#ifdef PNG_USER_CHUNKS_SUPPORTED
extern PNG_EXPORT(void,png_set_read_user_chunk_fn) PNGARG((png_structp png_ptr,
   png_voidp user_chunk_ptr, png_user_chunk_ptr read_user_chunk_fn));
extern PNG_EXPORT(png_voidp,png_get_user_chunk_ptr) PNGARG((png_structp
   png_ptr));
#endif

#ifdef PNG_PROGRESSIVE_READ_SUPPORTED
/* Sets the function callbacks for the push reader, and a pointer to a
 * user-defined structure available to the callback functions.
 */
extern PNG_EXPORT(void,png_set_progressive_read_fn) PNGARG((png_structp png_ptr,
   png_voidp progressive_ptr,
   png_progressive_info_ptr info_fn, png_progressive_row_ptr row_fn,
   png_progressive_end_ptr end_fn));

/* Returns the user pointer associated with the push read functions */
extern PNG_EXPORT(png_voidp,png_get_progressive_ptr)
   PNGARG((png_structp png_ptr));

/* Function to be called when data becomes available */
extern PNG_EXPORT(void,png_process_data) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_bytep buffer, png_size_t buffer_size));

/* Function that combines rows.  Not very much different than the
 * png_combine_row() call.  Is this even used?????
 */
extern PNG_EXPORT(void,png_progressive_combine_row) PNGARG((png_structp png_ptr,
   png_bytep old_row, png_bytep new_row));
#endif /* PNG_PROGRESSIVE_READ_SUPPORTED */

extern PNG_EXPORT(png_voidp,png_malloc) PNGARG((png_structp png_ptr,
   png_alloc_size_t size)) PNG_ALLOCATED;
/* Added at libpng version 1.4.0 */
extern PNG_EXPORT(png_voidp,png_calloc) PNGARG((png_structp png_ptr,
   png_alloc_size_t size)) PNG_ALLOCATED;

/* Added at libpng version 1.2.4 */
extern PNG_EXPORT(png_voidp,png_malloc_warn) PNGARG((png_structp png_ptr,
   png_alloc_size_t size)) PNG_ALLOCATED;

/* Frees a pointer allocated by png_malloc() */
extern PNG_EXPORT(void,png_free) PNGARG((png_structp png_ptr, png_voidp ptr));

/* Free data that was allocated internally */
extern PNG_EXPORT(void,png_free_data) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_32 free_me, int num));
/* Reassign responsibility for freeing existing data, whether allocated
 * by libpng or by the application */
extern PNG_EXPORT(void,png_data_freer) PNGARG((png_structp png_ptr,
   png_infop info_ptr, int freer, png_uint_32 mask));
/* Assignments for png_data_freer */
#define PNG_DESTROY_WILL_FREE_DATA 1
#define PNG_SET_WILL_FREE_DATA 1
#define PNG_USER_WILL_FREE_DATA 2
/* Flags for png_ptr->free_me and info_ptr->free_me */
#define PNG_FREE_HIST 0x0008
#define PNG_FREE_ICCP 0x0010
#define PNG_FREE_SPLT 0x0020
#define PNG_FREE_ROWS 0x0040
#define PNG_FREE_PCAL 0x0080
#define PNG_FREE_SCAL 0x0100
#define PNG_FREE_UNKN 0x0200
#define PNG_FREE_LIST 0x0400
#define PNG_FREE_PLTE 0x1000
#define PNG_FREE_TRNS 0x2000
#define PNG_FREE_TEXT 0x4000
#define PNG_FREE_ALL  0x7fff
#define PNG_FREE_MUL  0x4220 /* PNG_FREE_SPLT|PNG_FREE_TEXT|PNG_FREE_UNKN */

#ifdef PNG_USER_MEM_SUPPORTED
extern PNG_EXPORT(png_voidp,png_malloc_default) PNGARG((png_structp png_ptr,
   png_alloc_size_t size)) PNG_ALLOCATED;
extern PNG_EXPORT(void,png_free_default) PNGARG((png_structp png_ptr,
   png_voidp ptr));
#endif

#ifndef PNG_NO_ERROR_TEXT
/* Fatal error in PNG image of libpng - can't continue */
extern PNG_EXPORT(void,png_error) PNGARG((png_structp png_ptr,
   png_const_charp error_message)) PNG_NORETURN;

/* The same, but the chunk name is prepended to the error string. */
extern PNG_EXPORT(void,png_chunk_error) PNGARG((png_structp png_ptr,
   png_const_charp error_message)) PNG_NORETURN;

#else
/* Fatal error in PNG image of libpng - can't continue */
extern PNG_EXPORT(void,png_err) PNGARG((png_structp png_ptr)) PNG_NORETURN;
#endif

/* Non-fatal error in libpng.  Can continue, but may have a problem. */
extern PNG_EXPORT(void,png_warning) PNGARG((png_structp png_ptr,
   png_const_charp warning_message));

/* Non-fatal error in libpng, chunk name is prepended to message. */
extern PNG_EXPORT(void,png_chunk_warning) PNGARG((png_structp png_ptr,
   png_const_charp warning_message));

#ifdef PNG_BENIGN_ERRORS_SUPPORTED
/* Benign error in libpng.  Can continue, but may have a problem.
 * User can choose whether to handle as a fatal error or as a warning. */
extern PNG_EXPORT(void,png_benign_error) PNGARG((png_structp png_ptr,
   png_const_charp warning_message));

/* Same, chunk name is prepended to message. */
extern PNG_EXPORT(void,png_chunk_benign_error) PNGARG((png_structp png_ptr,
   png_const_charp warning_message));

extern PNG_EXPORT(void,png_set_benign_errors) PNGARG((png_structp
   png_ptr, int allowed));
#endif

/* The png_set_<chunk> functions are for storing values in the png_info_struct.
 * Similarly, the png_get_<chunk> calls are used to read values from the
 * png_info_struct, either storing the parameters in the passed variables, or
 * setting pointers into the png_info_struct where the data is stored.  The
 * png_get_<chunk> functions return a non-zero value if the data was available
 * in info_ptr, or return zero and do not change any of the parameters if the
 * data was not available.
 *
 * These functions should be used instead of directly accessing png_info
 * to avoid problems with future changes in the size and internal layout of
 * png_info_struct.
 */
/* Returns "flag" if chunk data is valid in info_ptr. */
extern PNG_EXPORT(png_uint_32,png_get_valid) PNGARG((png_structp png_ptr,
png_infop info_ptr, png_uint_32 flag));

/* Returns number of bytes needed to hold a transformed row. */
extern PNG_EXPORT(png_size_t,png_get_rowbytes) PNGARG((png_structp png_ptr,
png_infop info_ptr));

#ifdef PNG_INFO_IMAGE_SUPPORTED
/* Returns row_pointers, which is an array of pointers to scanlines that was
 * returned from png_read_png().
 */
extern PNG_EXPORT(png_bytepp,png_get_rows) PNGARG((png_structp png_ptr,
png_infop info_ptr));
/* Set row_pointers, which is an array of pointers to scanlines for use
 * by png_write_png().
 */
extern PNG_EXPORT(void,png_set_rows) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_bytepp row_pointers));
#endif

/* Returns number of color channels in image. */
extern PNG_EXPORT(png_byte,png_get_channels) PNGARG((png_structp png_ptr,
png_infop info_ptr));

#ifdef PNG_EASY_ACCESS_SUPPORTED
/* Returns image width in pixels. */
extern PNG_EXPORT(png_uint_32, png_get_image_width) PNGARG((png_structp
png_ptr, png_infop info_ptr));

/* Returns image height in pixels. */
extern PNG_EXPORT(png_uint_32, png_get_image_height) PNGARG((png_structp
png_ptr, png_infop info_ptr));

/* Returns image bit_depth. */
extern PNG_EXPORT(png_byte, png_get_bit_depth) PNGARG((png_structp
png_ptr, png_infop info_ptr));

/* Returns image color_type. */
extern PNG_EXPORT(png_byte, png_get_color_type) PNGARG((png_structp
png_ptr, png_infop info_ptr));

/* Returns image filter_type. */
extern PNG_EXPORT(png_byte, png_get_filter_type) PNGARG((png_structp
png_ptr, png_infop info_ptr));

/* Returns image interlace_type. */
extern PNG_EXPORT(png_byte, png_get_interlace_type) PNGARG((png_structp
png_ptr, png_infop info_ptr));

/* Returns image compression_type. */
extern PNG_EXPORT(png_byte, png_get_compression_type) PNGARG((png_structp
png_ptr, png_infop info_ptr));

/* Returns image resolution in pixels per meter, from pHYs chunk data. */
extern PNG_EXPORT(png_uint_32, png_get_pixels_per_meter) PNGARG((png_structp
png_ptr, png_infop info_ptr));
extern PNG_EXPORT(png_uint_32, png_get_x_pixels_per_meter) PNGARG((png_structp
png_ptr, png_infop info_ptr));
extern PNG_EXPORT(png_uint_32, png_get_y_pixels_per_meter) PNGARG((png_structp
png_ptr, png_infop info_ptr));

/* Returns pixel aspect ratio, computed from pHYs chunk data.  */
#ifdef PNG_FLOATING_POINT_SUPPORTED
extern PNG_EXPORT(float, png_get_pixel_aspect_ratio) PNGARG((png_structp
png_ptr, png_infop info_ptr));
#endif

/* Returns image x, y offset in pixels or microns, from oFFs chunk data. */
extern PNG_EXPORT(png_int_32, png_get_x_offset_pixels) PNGARG((png_structp
png_ptr, png_infop info_ptr));
extern PNG_EXPORT(png_int_32, png_get_y_offset_pixels) PNGARG((png_structp
png_ptr, png_infop info_ptr));
extern PNG_EXPORT(png_int_32, png_get_x_offset_microns) PNGARG((png_structp
png_ptr, png_infop info_ptr));
extern PNG_EXPORT(png_int_32, png_get_y_offset_microns) PNGARG((png_structp
png_ptr, png_infop info_ptr));

#endif /* PNG_EASY_ACCESS_SUPPORTED */

/* Returns pointer to signature string read from PNG header */
extern PNG_EXPORT(png_bytep,png_get_signature) PNGARG((png_structp png_ptr,
png_infop info_ptr));

#ifdef PNG_bKGD_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_bKGD) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_color_16p *background));
#endif

#ifdef PNG_bKGD_SUPPORTED
extern PNG_EXPORT(void,png_set_bKGD) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_color_16p background));
#endif

#ifdef PNG_cHRM_SUPPORTED
#ifdef PNG_FLOATING_POINT_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_cHRM) PNGARG((png_structp png_ptr,
   png_infop info_ptr, double *white_x, double *white_y, double *red_x,
   double *red_y, double *green_x, double *green_y, double *blue_x,
   double *blue_y));
#endif
#ifdef PNG_FIXED_POINT_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_cHRM_fixed) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_fixed_point *int_white_x, png_fixed_point
   *int_white_y, png_fixed_point *int_red_x, png_fixed_point *int_red_y,
   png_fixed_point *int_green_x, png_fixed_point *int_green_y, png_fixed_point
   *int_blue_x, png_fixed_point *int_blue_y));
#endif
#endif

#ifdef PNG_cHRM_SUPPORTED
#ifdef PNG_FLOATING_POINT_SUPPORTED
extern PNG_EXPORT(void,png_set_cHRM) PNGARG((png_structp png_ptr,
   png_infop info_ptr, double white_x, double white_y, double red_x,
   double red_y, double green_x, double green_y, double blue_x, double blue_y));
#endif
#ifdef PNG_FIXED_POINT_SUPPORTED
extern PNG_EXPORT(void,png_set_cHRM_fixed) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_fixed_point int_white_x, png_fixed_point int_white_y,
   png_fixed_point int_red_x, png_fixed_point int_red_y, png_fixed_point
   int_green_x, png_fixed_point int_green_y, png_fixed_point int_blue_x,
   png_fixed_point int_blue_y));
#endif
#endif

#ifdef PNG_gAMA_SUPPORTED
#ifdef PNG_FLOATING_POINT_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_gAMA) PNGARG((png_structp png_ptr,
   png_infop info_ptr, double *file_gamma));
#endif
extern PNG_EXPORT(png_uint_32,png_get_gAMA_fixed) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_fixed_point *int_file_gamma));
#endif

#ifdef PNG_gAMA_SUPPORTED
#ifdef PNG_FLOATING_POINT_SUPPORTED
extern PNG_EXPORT(void,png_set_gAMA) PNGARG((png_structp png_ptr,
   png_infop info_ptr, double file_gamma));
#endif
extern PNG_EXPORT(void,png_set_gAMA_fixed) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_fixed_point int_file_gamma));
#endif

#ifdef PNG_hIST_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_hIST) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_16p *hist));
#endif

#ifdef PNG_hIST_SUPPORTED
extern PNG_EXPORT(void,png_set_hIST) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_16p hist));
#endif

extern PNG_EXPORT(png_uint_32,png_get_IHDR) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_32 *width, png_uint_32 *height,
   int *bit_depth, int *color_type, int *interlace_method,
   int *compression_method, int *filter_method));

extern PNG_EXPORT(void,png_set_IHDR) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_32 width, png_uint_32 height, int bit_depth,
   int color_type, int interlace_method, int compression_method,
   int filter_method));

#ifdef PNG_oFFs_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_oFFs) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_int_32 *offset_x, png_int_32 *offset_y,
   int *unit_type));
#endif

#ifdef PNG_oFFs_SUPPORTED
extern PNG_EXPORT(void,png_set_oFFs) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_int_32 offset_x, png_int_32 offset_y,
   int unit_type));
#endif

#ifdef PNG_pCAL_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_pCAL) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_charp *purpose, png_int_32 *X0, png_int_32 *X1,
   int *type, int *nparams, png_charp *units, png_charpp *params));
#endif

#ifdef PNG_pCAL_SUPPORTED
extern PNG_EXPORT(void,png_set_pCAL) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_charp purpose, png_int_32 X0, png_int_32 X1,
   int type, int nparams, png_charp units, png_charpp params));
#endif

#ifdef PNG_pHYs_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_pHYs) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_32 *res_x, png_uint_32 *res_y, int *unit_type));
#endif

#ifdef PNG_pHYs_SUPPORTED
extern PNG_EXPORT(void,png_set_pHYs) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_uint_32 res_x, png_uint_32 res_y, int unit_type));
#endif

extern PNG_EXPORT(png_uint_32,png_get_PLTE) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_colorp *palette, int *num_palette));

extern PNG_EXPORT(void,png_set_PLTE) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_colorp palette, int num_palette));

#ifdef PNG_sBIT_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_sBIT) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_color_8p *sig_bit));
#endif

#ifdef PNG_sBIT_SUPPORTED
extern PNG_EXPORT(void,png_set_sBIT) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_color_8p sig_bit));
#endif

#ifdef PNG_sRGB_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_sRGB) PNGARG((png_structp png_ptr,
   png_infop info_ptr, int *intent));
#endif

#ifdef PNG_sRGB_SUPPORTED
extern PNG_EXPORT(void,png_set_sRGB) PNGARG((png_structp png_ptr,
   png_infop info_ptr, int intent));
extern PNG_EXPORT(void,png_set_sRGB_gAMA_and_cHRM) PNGARG((png_structp png_ptr,
   png_infop info_ptr, int intent));
#endif

#ifdef PNG_iCCP_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_iCCP) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_charpp name, int *compression_type,
   png_charpp profile, png_uint_32 *proflen));
   /* Note to maintainer: profile should be png_bytepp */
#endif

#ifdef PNG_iCCP_SUPPORTED
extern PNG_EXPORT(void,png_set_iCCP) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_charp name, int compression_type,
   png_charp profile, png_uint_32 proflen));
   /* Note to maintainer: profile should be png_bytep */
#endif

#ifdef PNG_sPLT_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_sPLT) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_sPLT_tpp entries));
#endif

#ifdef PNG_sPLT_SUPPORTED
extern PNG_EXPORT(void,png_set_sPLT) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_sPLT_tp entries, int nentries));
#endif

#ifdef PNG_TEXT_SUPPORTED
/* png_get_text also returns the number of text chunks in *num_text */
extern PNG_EXPORT(png_uint_32,png_get_text) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_textp *text_ptr, int *num_text));
#endif

/* Note while png_set_text() will accept a structure whose text,
 * language, and  translated keywords are NULL pointers, the structure
 * returned by png_get_text will always contain regular
 * zero-terminated C strings.  They might be empty strings but
 * they will never be NULL pointers.
 */

#ifdef PNG_TEXT_SUPPORTED
extern PNG_EXPORT(void,png_set_text) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_textp text_ptr, int num_text));
#endif

#ifdef PNG_tIME_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_tIME) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_timep *mod_time));
#endif

#ifdef PNG_tIME_SUPPORTED
extern PNG_EXPORT(void,png_set_tIME) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_timep mod_time));
#endif

#ifdef PNG_tRNS_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_tRNS) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_bytep *trans_alpha, int *num_trans,
   png_color_16p *trans_color));
#endif

#ifdef PNG_tRNS_SUPPORTED
extern PNG_EXPORT(void,png_set_tRNS) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_bytep trans_alpha, int num_trans,
   png_color_16p trans_color));
#endif

#ifdef PNG_tRNS_SUPPORTED
#endif

#ifdef PNG_sCAL_SUPPORTED
#ifdef PNG_FLOATING_POINT_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_sCAL) PNGARG((png_structp png_ptr,
   png_infop info_ptr, int *unit, double *width, double *height));
#else
#ifdef PNG_FIXED_POINT_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_sCAL_s) PNGARG((png_structp png_ptr,
   png_infop info_ptr, int *unit, png_charpp swidth, png_charpp sheight));
#endif
#endif
#endif /* PNG_sCAL_SUPPORTED */

#ifdef PNG_sCAL_SUPPORTED
#ifdef PNG_FLOATING_POINT_SUPPORTED
extern PNG_EXPORT(void,png_set_sCAL) PNGARG((png_structp png_ptr,
   png_infop info_ptr, int unit, double width, double height));
#else
#ifdef PNG_FIXED_POINT_SUPPORTED
extern PNG_EXPORT(void,png_set_sCAL_s) PNGARG((png_structp png_ptr,
   png_infop info_ptr, int unit, png_charp swidth, png_charp sheight));
#endif
#endif
#endif /* PNG_sCAL_SUPPORTED || PNG_WRITE_sCAL_SUPPORTED */

#ifdef PNG_HANDLE_AS_UNKNOWN_SUPPORTED
/* Provide a list of chunks and how they are to be handled, if the built-in
   handling or default unknown chunk handling is not desired.  Any chunks not
   listed will be handled in the default manner.  The IHDR and IEND chunks
   must not be listed.
      keep = 0: follow default behaviour
           = 1: do not keep
           = 2: keep only if safe-to-copy
           = 3: keep even if unsafe-to-copy
*/
extern PNG_EXPORT(void, png_set_keep_unknown_chunks) PNGARG((png_structp
   png_ptr, int keep, png_bytep chunk_list, int num_chunks));
PNG_EXPORT(int,png_handle_as_unknown) PNGARG((png_structp png_ptr, png_bytep
   chunk_name));
#endif
#ifdef PNG_UNKNOWN_CHUNKS_SUPPORTED
extern PNG_EXPORT(void, png_set_unknown_chunks) PNGARG((png_structp png_ptr,
   png_infop info_ptr, png_unknown_chunkp unknowns, int num_unknowns));
extern PNG_EXPORT(void, png_set_unknown_chunk_location)
   PNGARG((png_structp png_ptr, png_infop info_ptr, int chunk, int location));
extern PNG_EXPORT(png_uint_32,png_get_unknown_chunks) PNGARG((png_structp
   png_ptr, png_infop info_ptr, png_unknown_chunkpp entries));
#endif

/* Png_free_data() will turn off the "valid" flag for anything it frees.
 * If you need to turn it off for a chunk that your application has freed,
 * you can use png_set_invalid(png_ptr, info_ptr, PNG_INFO_CHNK);
 */
extern PNG_EXPORT(void, png_set_invalid) PNGARG((png_structp png_ptr,
   png_infop info_ptr, int mask));

#ifdef PNG_INFO_IMAGE_SUPPORTED
/* The "params" pointer is currently not used and is for future expansion. */
extern PNG_EXPORT(void, png_read_png) PNGARG((png_structp png_ptr,
                        png_infop info_ptr,
                        int transforms,
                        png_voidp params));
extern PNG_EXPORT(void, png_write_png) PNGARG((png_structp png_ptr,
                        png_infop info_ptr,
                        int transforms,
                        png_voidp params));
#endif

extern PNG_EXPORT(png_charp,png_get_copyright) PNGARG((png_structp png_ptr));
extern PNG_EXPORT(png_charp,png_get_header_ver) PNGARG((png_structp png_ptr));
extern PNG_EXPORT(png_charp,png_get_header_version) PNGARG((png_structp
    png_ptr));
extern PNG_EXPORT(png_charp,png_get_libpng_ver) PNGARG((png_structp png_ptr));

#ifdef PNG_MNG_FEATURES_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_permit_mng_features) PNGARG((png_structp
   png_ptr, png_uint_32 mng_features_permitted));
#endif

/* For use in png_set_keep_unknown, added to version 1.2.6 */
#define PNG_HANDLE_CHUNK_AS_DEFAULT   0
#define PNG_HANDLE_CHUNK_NEVER        1
#define PNG_HANDLE_CHUNK_IF_SAFE      2
#define PNG_HANDLE_CHUNK_ALWAYS       3

/* Strip the prepended error numbers ("#nnn ") from error and warning
 * messages before passing them to the error or warning handler.
 */
#ifdef PNG_ERROR_NUMBERS_SUPPORTED
extern PNG_EXPORT(void,png_set_strip_error_numbers) PNGARG((png_structp
   png_ptr, png_uint_32 strip_mode));
#endif

/* Added in libpng-1.2.6 */
#ifdef PNG_SET_USER_LIMITS_SUPPORTED
extern PNG_EXPORT(void,png_set_user_limits) PNGARG((png_structp
   png_ptr, png_uint_32 user_width_max, png_uint_32 user_height_max));
extern PNG_EXPORT(png_uint_32,png_get_user_width_max) PNGARG((png_structp
   png_ptr));
extern PNG_EXPORT(png_uint_32,png_get_user_height_max) PNGARG((png_structp
   png_ptr));
/* Added in libpng-1.4.0 */
extern PNG_EXPORT(void,png_set_chunk_cache_max) PNGARG((png_structp
   png_ptr, png_uint_32 user_chunk_cache_max));
extern PNG_EXPORT(png_uint_32,png_get_chunk_cache_max)
   PNGARG((png_structp png_ptr));
/* Added in libpng-1.4.1 */
extern PNG_EXPORT(void,png_set_chunk_malloc_max) PNGARG((png_structp
   png_ptr, png_alloc_size_t user_chunk_cache_max));
extern PNG_EXPORT(png_alloc_size_t,png_get_chunk_malloc_max)
   PNGARG((png_structp png_ptr));
#endif

#if defined(PNG_INCH_CONVERSIONS) && defined(PNG_FLOATING_POINT_SUPPORTED)
PNG_EXPORT(png_uint_32,png_get_pixels_per_inch) PNGARG((png_structp png_ptr,
png_infop info_ptr));

PNG_EXPORT(png_uint_32,png_get_x_pixels_per_inch) PNGARG((png_structp png_ptr,
png_infop info_ptr));

PNG_EXPORT(png_uint_32,png_get_y_pixels_per_inch) PNGARG((png_structp png_ptr,
png_infop info_ptr));

PNG_EXPORT(float,png_get_x_offset_inches) PNGARG((png_structp png_ptr,
png_infop info_ptr));

PNG_EXPORT(float,png_get_y_offset_inches) PNGARG((png_structp png_ptr,
png_infop info_ptr));

#ifdef PNG_pHYs_SUPPORTED
PNG_EXPORT(png_uint_32,png_get_pHYs_dpi) PNGARG((png_structp png_ptr,
png_infop info_ptr, png_uint_32 *res_x, png_uint_32 *res_y, int *unit_type));
#endif /* PNG_pHYs_SUPPORTED */
#endif  /* PNG_INCH_CONVERSIONS && PNG_FLOATING_POINT_SUPPORTED */

/* Added in libpng-1.4.0 */
#ifdef PNG_IO_STATE_SUPPORTED
extern PNG_EXPORT(png_uint_32,png_get_io_state) PNGARG((png_structp png_ptr));

extern PNG_EXPORT(png_bytep,png_get_io_chunk_name)
   PNGARG((png_structp png_ptr));

/* The flags returned by png_get_io_state() are the following: */
#define PNG_IO_NONE        0x0000   /* no I/O at this moment */
#define PNG_IO_READING     0x0001   /* currently reading */
#define PNG_IO_WRITING     0x0002   /* currently writing */
#define PNG_IO_SIGNATURE   0x0010   /* currently at the file signature */
#define PNG_IO_CHUNK_HDR   0x0020   /* currently at the chunk header */
#define PNG_IO_CHUNK_DATA  0x0040   /* currently at the chunk data */
#define PNG_IO_CHUNK_CRC   0x0080   /* currently at the chunk crc */
#define PNG_IO_MASK_OP     0x000f   /* current operation: reading/writing */
#define PNG_IO_MASK_LOC    0x00f0   /* current location: sig/hdr/data/crc */
#endif /* ?PNG_IO_STATE_SUPPORTED */

/* Maintainer: Put new public prototypes here ^, in libpng.3, and project
 * defs
 */

#ifdef PNG_READ_COMPOSITE_NODIV_SUPPORTED
/* With these routines we avoid an integer divide, which will be slower on
 * most machines.  However, it does take more operations than the corresponding
 * divide method, so it may be slower on a few RISC systems.  There are two
 * shifts (by 8 or 16 bits) and an addition, versus a single integer divide.
 *
 * Note that the rounding factors are NOT supposed to be the same!  128 and
 * 32768 are correct for the NODIV code; 127 and 32767 are correct for the
 * standard method.
 *
 * [Optimized code by Greg Roelofs and Mark Adler...blame us for bugs. :-) ]
 */

 /* fg and bg should be in `gamma 1.0' space; alpha is the opacity          */

#  define png_composite(composite, fg, alpha, bg)         \
     { png_uint_16 temp = (png_uint_16)((png_uint_16)(fg) \
           * (png_uint_16)(alpha)                         \
           + (png_uint_16)(bg)*(png_uint_16)(255          \
           - (png_uint_16)(alpha)) + (png_uint_16)128);   \
       (composite) = (png_byte)((temp + (temp >> 8)) >> 8); }

#  define png_composite_16(composite, fg, alpha, bg)       \
     { png_uint_32 temp = (png_uint_32)((png_uint_32)(fg)  \
           * (png_uint_32)(alpha)                          \
           + (png_uint_32)(bg)*(png_uint_32)(65535L        \
           - (png_uint_32)(alpha)) + (png_uint_32)32768L); \
       (composite) = (png_uint_16)((temp + (temp >> 16)) >> 16); }

#else  /* Standard method using integer division */

#  define png_composite(composite, fg, alpha, bg)                            \
     (composite) = (png_byte)(((png_uint_16)(fg) * (png_uint_16)(alpha) +    \
       (png_uint_16)(bg) * (png_uint_16)(255 - (png_uint_16)(alpha)) +       \
       (png_uint_16)127) / 255)

#  define png_composite_16(composite, fg, alpha, bg)                         \
     (composite) = (png_uint_16)(((png_uint_32)(fg) * (png_uint_32)(alpha) + \
       (png_uint_32)(bg)*(png_uint_32)(65535L - (png_uint_32)(alpha)) +      \
       (png_uint_32)32767) / (png_uint_32)65535L)
#endif /* PNG_READ_COMPOSITE_NODIV_SUPPORTED */

#ifdef PNG_USE_READ_MACROS
/* Inline macros to do direct reads of bytes from the input buffer.
 * The png_get_int_32() routine assumes we are using two's complement
 * format for negative values, which is almost certainly true.
 */
/* We could make special-case BIG_ENDIAN macros that do direct reads here */
#  define png_get_uint_32(buf) \
     (((png_uint_32)(*(buf)) << 24) + \
      ((png_uint_32)(*((buf) + 1)) << 16) + \
      ((png_uint_32)(*((buf) + 2)) << 8) + \
      ((png_uint_32)(*((buf) + 3))))
#  define png_get_uint_16(buf) \
     (((png_uint_32)(*(buf)) << 8) + \
      ((png_uint_32)(*((buf) + 1))))
#ifdef PNG_GET_INT_32_SUPPORTED
#  define png_get_int_32(buf) \
     (((png_int_32)(*(buf)) << 24) + \
      ((png_int_32)(*((buf) + 1)) << 16) + \
      ((png_int_32)(*((buf) + 2)) << 8) + \
      ((png_int_32)(*((buf) + 3))))
#endif
#else
extern PNG_EXPORT(png_uint_32,png_get_uint_32) PNGARG((png_bytep buf));
extern PNG_EXPORT(png_uint_16,png_get_uint_16) PNGARG((png_bytep buf));
#ifdef PNG_GET_INT_32_SUPPORTED
extern PNG_EXPORT(png_int_32,png_get_int_32) PNGARG((png_bytep buf));
#endif
#endif
extern PNG_EXPORT(png_uint_32,png_get_uint_31)
  PNGARG((png_structp png_ptr, png_bytep buf));
/* No png_get_int_16 -- may be added if there's a real need for it. */

/* Place a 32-bit number into a buffer in PNG byte order (big-endian). */
extern PNG_EXPORT(void,png_save_uint_32)
   PNGARG((png_bytep buf, png_uint_32 i));
extern PNG_EXPORT(void,png_save_int_32)
   PNGARG((png_bytep buf, png_int_32 i));

/* Place a 16-bit number into a buffer in PNG byte order.
 * The parameter is declared unsigned int, not png_uint_16,
 * just to avoid potential problems on pre-ANSI C compilers.
 */
extern PNG_EXPORT(void,png_save_uint_16)
   PNGARG((png_bytep buf, unsigned int i));
/* No png_save_int_16 -- may be added if there's a real need for it. */

/* ************************************************************************* */

/* Various modes of operation.  Note that after an init, mode is set to
 * zero automatically when the structure is created.
 */
#define PNG_HAVE_IHDR               0x01
#define PNG_HAVE_PLTE               0x02
#define PNG_HAVE_IDAT               0x04
#define PNG_AFTER_IDAT              0x08 /* Have complete zlib datastream */
#define PNG_HAVE_IEND               0x10
#define PNG_HAVE_gAMA               0x20
#define PNG_HAVE_cHRM               0x40

#ifdef __cplusplus
}
#endif

#endif /* PNG_VERSION_INFO_ONLY */
/* Do not put anything past this line */
#endif /* PNG_H */
