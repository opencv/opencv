/*
 * cpu_features_common.h - code shared by all lib/$arch/cpu_features.c
 *
 * Copyright 2020 Eric Biggers
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef LIB_CPU_FEATURES_COMMON_H
#define LIB_CPU_FEATURES_COMMON_H

#if defined(TEST_SUPPORT__DO_NOT_USE) && !defined(FREESTANDING)
   /* for strdup() and strtok_r() */
#  undef _ANSI_SOURCE
#  ifndef __APPLE__
#    undef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
#  include <stdio.h>
#  include <stdlib.h>
#  include <string.h>
#endif

#include "lib_common.h"

struct cpu_feature {
	u32 bit;
	const char *name;
};

#if defined(TEST_SUPPORT__DO_NOT_USE) && !defined(FREESTANDING)
/* Disable any features that are listed in $LIBDEFLATE_DISABLE_CPU_FEATURES. */
static inline void
disable_cpu_features_for_testing(u32 *features,
				 const struct cpu_feature *feature_table,
				 size_t feature_table_length)
{
	char *env_value, *strbuf, *p, *saveptr = NULL;
	size_t i;

	env_value = getenv("LIBDEFLATE_DISABLE_CPU_FEATURES");
	if (!env_value)
		return;
	strbuf = strdup(env_value);
	if (!strbuf)
		abort();
	p = strtok_r(strbuf, ",", &saveptr);
	while (p) {
		for (i = 0; i < feature_table_length; i++) {
			if (strcmp(p, feature_table[i].name) == 0) {
				*features &= ~feature_table[i].bit;
				break;
			}
		}
		if (i == feature_table_length) {
			fprintf(stderr,
				"unrecognized feature in LIBDEFLATE_DISABLE_CPU_FEATURES: \"%s\"\n",
				p);
			abort();
		}
		p = strtok_r(NULL, ",", &saveptr);
	}
	free(strbuf);
}
#else /* TEST_SUPPORT__DO_NOT_USE */
static inline void
disable_cpu_features_for_testing(u32 *features,
				 const struct cpu_feature *feature_table,
				 size_t feature_table_length)
{
}
#endif /* !TEST_SUPPORT__DO_NOT_USE */

#endif /* LIB_CPU_FEATURES_COMMON_H */
