/*
 * Copyright (c) 2009 Frank Warmerdam
 *
 * Permission to use, copy, modify, distribute, and sell this software and 
 * its documentation for any purpose is hereby granted without fee, provided
 * that (i) the above copyright notices and this permission notice appear in
 * all copies of the software and related documentation, and (ii) the names of
 * Sam Leffler and Silicon Graphics may not be used in any advertising or
 * publicity relating to the software without the specific, prior written
 * permission of Sam Leffler and Silicon Graphics.
 * 
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY 
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  
 * 
 * IN NO EVENT SHALL SAM LEFFLER OR SILICON GRAPHICS BE LIABLE FOR
 * ANY SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND,
 * OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THIS SOFTWARE.
 */

#ifndef _LIBPORT_
#define	_LIBPORT_

#if defined(HAVE_CONFIG_H)
#  include <tif_config.h>
#endif

int getopt(int argc, char * const argv[], const char *optstring);
extern   char *optarg;
extern   int opterr;
extern   int optind;
extern   int optopt;

int strcasecmp(const char *s1, const char *s2);

#ifndef HAVE_GETOPT
#  define HAVE_GETOPT 1
#endif

#if !defined(HAVE_STRTOL)
long strtol(const char *nptr, char **endptr, int base);
#endif
#if !defined(HAVE_STRTOLL)
long long strtoll(const char *nptr, char **endptr, int base);
#endif
#if !defined(HAVE_STRTOUL)
unsigned long strtoul(const char *nptr, char **endptr, int base);
#endif
#if !defined(HAVE_STRTOULL)
unsigned long long strtoull(const char *nptr, char **endptr, int base);
#endif

#if 0
void *
lfind(const void *key, const void *base, size_t *nmemb, size_t size,
      int(*compar)(const void *, const void *));
#endif

#if !defined(HAVE_SNPRINTF)
#undef vsnprintf
#define vsnprintf _TIFF_vsnprintf_f

#undef snprintf
#define snprintf _TIFF_snprintf_f
int snprintf(char* str, size_t size, const char* format, ...);
#endif

#endif /* ndef _LIBPORT_ */
