/*
 * x86/crc32_impl.h - x86 implementations of the gzip CRC-32 algorithm
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

#ifndef LIB_X86_CRC32_IMPL_H
#define LIB_X86_CRC32_IMPL_H

#include "cpu_features.h"

/* PCLMUL implementation */
#if HAVE_PCLMUL_INTRIN
#  define crc32_x86_pclmul	crc32_x86_pclmul
#  define SUFFIX			 _pclmul
#  if HAVE_PCLMUL_NATIVE
#    define ATTRIBUTES
#  else
#    define ATTRIBUTES		_target_attribute("pclmul")
#  endif
#  define FOLD_PARTIAL_VECS	0
#  include "crc32_pclmul_template.h"
#endif

/*
 * PCLMUL/AVX implementation.  This implementation has two benefits over the
 * regular PCLMUL one.  First, simply compiling against the AVX target can
 * improve performance significantly (e.g. 10100 MB/s to 16700 MB/s on Skylake)
 * without actually using any AVX intrinsics, probably due to the availability
 * of non-destructive VEX-encoded instructions.  Second, AVX support implies
 * SSSE3 and SSE4.1 support, and we can use SSSE3 and SSE4.1 intrinsics for
 * efficient handling of partial blocks.  (We *could* compile a variant with
 * PCLMUL+SSSE3+SSE4.1 w/o AVX, but for simplicity we don't currently bother.)
 *
 * FIXME: with MSVC, this isn't actually compiled with AVX code generation
 * enabled yet.  That would require that this be moved to its own .c file.
 */
#if HAVE_PCLMUL_INTRIN && HAVE_AVX_INTRIN
#  define crc32_x86_pclmul_avx	crc32_x86_pclmul_avx
#  define SUFFIX			 _pclmul_avx
#  if HAVE_PCLMUL_NATIVE && HAVE_AVX_NATIVE
#    define ATTRIBUTES
#  else
#    define ATTRIBUTES		_target_attribute("pclmul,avx")
#  endif
#  define FOLD_PARTIAL_VECS	1
#  include "crc32_pclmul_template.h"
#endif

/*
 * If the best implementation is statically available, use it unconditionally.
 * Otherwise choose the best implementation at runtime.
 */
#if defined(crc32_x86_pclmul_avx) && HAVE_PCLMUL_NATIVE && HAVE_AVX_NATIVE
#define DEFAULT_IMPL	crc32_x86_pclmul_avx
#else
static inline crc32_func_t
arch_select_crc32_func(void)
{
	const u32 features MAYBE_UNUSED = get_x86_cpu_features();

#ifdef crc32_x86_pclmul_avx
	if (HAVE_PCLMUL(features) && HAVE_AVX(features))
		return crc32_x86_pclmul_avx;
#endif
#ifdef crc32_x86_pclmul
	if (HAVE_PCLMUL(features))
		return crc32_x86_pclmul;
#endif
	return NULL;
}
#define arch_select_crc32_func	arch_select_crc32_func
#endif

#endif /* LIB_X86_CRC32_IMPL_H */
