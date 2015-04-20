/*      $NetBSD: getopt.h,v 1.4 2000/07/07 10:43:54 ad Exp $    */
/*      $FreeBSD: src/include/getopt.h,v 1.1 2002/09/29 04:14:30 eric Exp $ */

/*-
 * Copyright (c) 2000 The NetBSD Foundation, Inc.
 * All rights reserved.
 *
 * This code is derived from software contributed to The NetBSD Foundation
 * by Dieter Baron and Thomas Klausner.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *        This product includes software developed by the NetBSD
 *        Foundation, Inc. and its contributors.
 * 4. Neither the name of The NetBSD Foundation nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE NETBSD FOUNDATION, INC. AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _GETOPT_H_
#define _GETOPT_H_

#ifdef _WIN32
/* from <sys/cdefs.h> */
# ifdef  __cplusplus
#  define __BEGIN_DECLS  extern "C" {
#  define __END_DECLS    }
# else
#  define __BEGIN_DECLS
#  define __END_DECLS
# endif
# define __P(args)      args
#endif

/*#ifndef _WIN32
#include <sys/cdefs.h>
//#include <unistd.h>
#endif*/

#if defined(_WIN32) && defined(GETOPT_DLL)
  #if defined(gdcmgetopt_EXPORTS)
    #define GETOPT_EXPORT __declspec( dllexport )
  #else
    #define GETOPT_EXPORT __declspec( dllimport )
  #endif
#else
  #define GETOPT_EXPORT
#endif /*defined(WIN32)*/

#if defined(_WIN32) && defined(GETOPT_DLL)
  #if defined(gdcmgetopt_EXPORTS)
    #define GETOPT_EXTERN __declspec( dllexport )
  #else
    #define GETOPT_EXTERN __declspec( dllimport )
  #endif
#else
  #define GETOPT_EXTERN extern
#endif /*defined(WIN32)*/



/*
 * Gnu like getopt_long() and BSD4.4 getsubopt()/optreset extensions
 */
#if !defined(_POSIX_SOURCE) && !defined(_XOPEN_SOURCE)
#define no_argument        0
#define required_argument  1
#define optional_argument  2

struct option {
        /* name of long option */
        const char *name;
        /*
         * one of no_argument, required_argument, and optional_argument:
         * whether option takes an argument
         */
        int has_arg;
        /* if not NULL, set *flag to val when option found */
        int *flag;
        /* if flag not NULL, value to set *flag to; else return value */
        int val;
};

__BEGIN_DECLS
GETOPT_EXPORT int getopt_long __P((int, char * const *, const char *,
    const struct option *, int *));
__END_DECLS
#endif

#ifdef _WIN32
/* These are global getopt variables */
__BEGIN_DECLS

GETOPT_EXTERN int   opterr,   /* if error message should be printed */
                        optind,   /* index into parent argv vector */
                        optopt,   /* character checked for validity */
                        optreset; /* reset getopt */
GETOPT_EXTERN char* optarg;   /* argument associated with option */

/* Original getopt */
GETOPT_EXPORT int getopt __P((int, char * const *, const char *));

__END_DECLS
#endif
 
#endif /* !_GETOPT_H_ */
