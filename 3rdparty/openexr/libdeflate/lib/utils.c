/*
 * utils.c - utility functions for libdeflate
 *
 * Copyright 2016 Eric Biggers
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

#include "lib_common.h"

#ifdef FREESTANDING
#  define malloc NULL
#  define free NULL
#else
#  include <stdlib.h>
#endif

malloc_func_t libdeflate_default_malloc_func = malloc;
free_func_t libdeflate_default_free_func = free;

void *
libdeflate_aligned_malloc(malloc_func_t malloc_func,
			  size_t alignment, size_t size)
{
	void *ptr = (*malloc_func)(sizeof(void *) + alignment - 1 + size);

	if (ptr) {
		void *orig_ptr = ptr;

		ptr = (void *)ALIGN((uintptr_t)ptr + sizeof(void *), alignment);
		((void **)ptr)[-1] = orig_ptr;
	}
	return ptr;
}

void
libdeflate_aligned_free(free_func_t free_func, void *ptr)
{
	(*free_func)(((void **)ptr)[-1]);
}

LIBDEFLATEAPI void
libdeflate_set_memory_allocator(malloc_func_t malloc_func,
				free_func_t free_func)
{
	libdeflate_default_malloc_func = malloc_func;
	libdeflate_default_free_func = free_func;
}

/*
 * Implementations of libc functions for freestanding library builds.
 * Normal library builds don't use these.  Not optimized yet; usually the
 * compiler expands these functions and doesn't actually call them anyway.
 */
#ifdef FREESTANDING
#undef memset
void * __attribute__((weak))
memset(void *s, int c, size_t n)
{
	u8 *p = s;
	size_t i;

	for (i = 0; i < n; i++)
		p[i] = c;
	return s;
}

#undef memcpy
void * __attribute__((weak))
memcpy(void *dest, const void *src, size_t n)
{
	u8 *d = dest;
	const u8 *s = src;
	size_t i;

	for (i = 0; i < n; i++)
		d[i] = s[i];
	return dest;
}

#undef memmove
void * __attribute__((weak))
memmove(void *dest, const void *src, size_t n)
{
	u8 *d = dest;
	const u8 *s = src;
	size_t i;

	if (d <= s)
		return memcpy(d, s, n);

	for (i = n; i > 0; i--)
		d[i - 1] = s[i - 1];
	return dest;
}

#undef memcmp
int __attribute__((weak))
memcmp(const void *s1, const void *s2, size_t n)
{
	const u8 *p1 = s1;
	const u8 *p2 = s2;
	size_t i;

	for (i = 0; i < n; i++) {
		if (p1[i] != p2[i])
			return (int)p1[i] - (int)p2[i];
	}
	return 0;
}
#endif /* FREESTANDING */

#ifdef LIBDEFLATE_ENABLE_ASSERTIONS
#include <stdio.h>
#include <stdlib.h>
void
libdeflate_assertion_failed(const char *expr, const char *file, int line)
{
	fprintf(stderr, "Assertion failed: %s at %s:%d\n", expr, file, line);
	abort();
}
#endif /* LIBDEFLATE_ENABLE_ASSERTIONS */
