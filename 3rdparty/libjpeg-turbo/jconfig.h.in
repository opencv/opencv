/* Version ID for the JPEG library.
 * Might be useful for tests like "#if JPEG_LIB_VERSION >= 60".
 */
#define JPEG_LIB_VERSION  @JPEG_LIB_VERSION@

/* libjpeg-turbo version */
#define LIBJPEG_TURBO_VERSION  @VERSION@

/* libjpeg-turbo version in integer form */
#define LIBJPEG_TURBO_VERSION_NUMBER  @LIBJPEG_TURBO_VERSION_NUMBER@

/* Support arithmetic encoding */
#cmakedefine C_ARITH_CODING_SUPPORTED 1

/* Support arithmetic decoding */
#cmakedefine D_ARITH_CODING_SUPPORTED 1

/* Support in-memory source/destination managers */
#cmakedefine MEM_SRCDST_SUPPORTED 1

/* Use accelerated SIMD routines. */
#cmakedefine WITH_SIMD 1

/*
 * Define BITS_IN_JSAMPLE as either
 *   8   for 8-bit sample values (the usual setting)
 *   12  for 12-bit sample values
 * Only 8 and 12 are legal data precisions for lossy JPEG according to the
 * JPEG standard, and the IJG code does not support anything else!
 * We do not support run-time selection of data precision, sorry.
 */

#define BITS_IN_JSAMPLE  @BITS_IN_JSAMPLE@      /* use 8 or 12 */

/* Define to 1 if you have the <locale.h> header file. */
#cmakedefine HAVE_LOCALE_H 1

/* Define to 1 if you have the <stddef.h> header file. */
#cmakedefine HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#cmakedefine HAVE_STDLIB_H 1

/* Define if you need to include <sys/types.h> to get size_t. */
#cmakedefine NEED_SYS_TYPES_H 1

/* Define if you have BSD-like bzero and bcopy in <strings.h> rather than
   memset/memcpy in <string.h>. */
#cmakedefine NEED_BSD_STRINGS 1

/* Define to 1 if the system has the type `unsigned char'. */
#cmakedefine HAVE_UNSIGNED_CHAR 1

/* Define to 1 if the system has the type `unsigned short'. */
#cmakedefine HAVE_UNSIGNED_SHORT 1

/* Compiler does not support pointers to undefined structures. */
#cmakedefine INCOMPLETE_TYPES_BROKEN 1

/* Define if your (broken) compiler shifts signed values as if they were
   unsigned. */
#cmakedefine RIGHT_SHIFT_IS_UNSIGNED 1

/* Define to 1 if type `char' is unsigned and you are not using gcc.  */
#ifndef __CHAR_UNSIGNED__
  #cmakedefine __CHAR_UNSIGNED__ 1
#endif

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to `unsigned int' if <sys/types.h> does not define. */
/* #undef size_t */
