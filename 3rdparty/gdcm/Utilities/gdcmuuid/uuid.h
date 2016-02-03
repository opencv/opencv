/* vi: set sw=4 ts=4: */
/*
 * Public include file for the UUID library
 *
 * Copyright (C) 1996, 1997, 1998 Theodore Ts'o.
 *
 * %Begin-Header%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, and the entire permission notice in its entirety,
 *    including the disclaimer of warranties.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote
 *    products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ALL OF
 * WHICH ARE HEREBY DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
 * USE OF THIS SOFTWARE, EVEN IF NOT ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 * %End-Header%
 */

#ifndef _UUID_UUID_H
#define _UUID_UUID_H

#include "uuid_mangle.h"

#if defined(_WIN32) && defined(UUID_DLL)
  #if defined(uuid_EXPORTS)
    #define UUID_EXPORT __declspec( dllexport )
  #else
    #define UUID_EXPORT __declspec( dllimport )
  #endif
#else
#if __GNUC__ >= 4
#define UUID_EXPORT    __attribute__ ((visibility ("default")))
#else
  #define UUID_EXPORT
#endif
#endif /*defined(WIN32)*/


#include <sys/types.h>
#include <time.h>
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h> /* timeval CYGWIN is important */
#endif

/* apparently types.h or time.h is polluting our namespace on Win32... */
#if defined(uuid_t)
#undef uuid_t
#endif
typedef unsigned char uuid_t[16];

/* UUID Variant definitions */
#define UUID_VARIANT_NCS  0
#define UUID_VARIANT_DCE  1
#define UUID_VARIANT_MICROSOFT  2
#define UUID_VARIANT_OTHER  3

/* UUID Type definitions */
#define UUID_TYPE_DCE_TIME   1
#define UUID_TYPE_DCE_RANDOM 4

/* Allow UUID constants to be defined */
#ifdef __GNUC__
#define UUID_DEFINE(name,u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15) \
  static const uuid_t name ATTRIBUTE_UNUSED = {u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15}
#else
#define UUID_DEFINE(name,u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15) \
  static const uuid_t name = {u0,u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14,u15}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* clear.c */
/*void uuid_clear(uuid_t uu);*/
#define uuid_clear(uu) memset(uu, 0, sizeof(uu))

/* compare.c */
int uuid_compare(const uuid_t uu1, const uuid_t uu2);

/* copy.c */
/*void uuid_copy(uuid_t dst, const uuid_t src);*/
#define uuid_copy(dst,src) memcpy(dst, src, sizeof(dst))

/* gen_uuid.c */
UUID_EXPORT void uuid_generate(uuid_t out);
void uuid_generate_random(uuid_t out);
UUID_EXPORT int uuid_get_node_id(unsigned char *node_id);
void uuid_generate_time(uuid_t out);

/* isnull.c */
/*int uuid_is_null(const uuid_t uu);*/
#define uuid_is_null(uu) (!memcmp(uu, "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0", sizeof(uu)))

/* parse.c */
UUID_EXPORT int uuid_parse(const char *in, uuid_t uu);

/* unparse.c */
UUID_EXPORT void uuid_unparse(const uuid_t uu, char *out);
void uuid_unparse_lower(const uuid_t uu, char *out);
void uuid_unparse_upper(const uuid_t uu, char *out);

/* uuid_time.c */
time_t uuid_time(const uuid_t uu, struct timeval *ret_tv);
int uuid_type(const uuid_t uu);
int uuid_variant(const uuid_t uu);

#ifdef __cplusplus
}
#endif

#endif /* _UUID_UUID_H */
