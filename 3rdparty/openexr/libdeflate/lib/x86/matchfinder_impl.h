/*
 * x86/matchfinder_impl.h - x86 implementations of matchfinder functions
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

#ifndef LIB_X86_MATCHFINDER_IMPL_H
#define LIB_X86_MATCHFINDER_IMPL_H

#include "cpu_features.h"

#if HAVE_AVX2_NATIVE
#  include <immintrin.h>
static forceinline void
matchfinder_init_avx2(mf_pos_t *data, size_t size)
{
	__m256i *p = (__m256i *)data;
	__m256i v = _mm256_set1_epi16(MATCHFINDER_INITVAL);

	STATIC_ASSERT(MATCHFINDER_MEM_ALIGNMENT % sizeof(*p) == 0);
	STATIC_ASSERT(MATCHFINDER_SIZE_ALIGNMENT % (4 * sizeof(*p)) == 0);
	STATIC_ASSERT(sizeof(mf_pos_t) == 2);

	do {
		p[0] = v;
		p[1] = v;
		p[2] = v;
		p[3] = v;
		p += 4;
		size -= 4 * sizeof(*p);
	} while (size != 0);
}
#define matchfinder_init matchfinder_init_avx2

static forceinline void
matchfinder_rebase_avx2(mf_pos_t *data, size_t size)
{
	__m256i *p = (__m256i *)data;
	__m256i v = _mm256_set1_epi16((u16)-MATCHFINDER_WINDOW_SIZE);

	STATIC_ASSERT(MATCHFINDER_MEM_ALIGNMENT % sizeof(*p) == 0);
	STATIC_ASSERT(MATCHFINDER_SIZE_ALIGNMENT % (4 * sizeof(*p)) == 0);
	STATIC_ASSERT(sizeof(mf_pos_t) == 2);

	do {
		/* PADDSW: Add Packed Signed Integers With Signed Saturation  */
		p[0] = _mm256_adds_epi16(p[0], v);
		p[1] = _mm256_adds_epi16(p[1], v);
		p[2] = _mm256_adds_epi16(p[2], v);
		p[3] = _mm256_adds_epi16(p[3], v);
		p += 4;
		size -= 4 * sizeof(*p);
	} while (size != 0);
}
#define matchfinder_rebase matchfinder_rebase_avx2

#elif HAVE_SSE2_NATIVE
#  include <emmintrin.h>
static forceinline void
matchfinder_init_sse2(mf_pos_t *data, size_t size)
{
	__m128i *p = (__m128i *)data;
	__m128i v = _mm_set1_epi16(MATCHFINDER_INITVAL);

	STATIC_ASSERT(MATCHFINDER_MEM_ALIGNMENT % sizeof(*p) == 0);
	STATIC_ASSERT(MATCHFINDER_SIZE_ALIGNMENT % (4 * sizeof(*p)) == 0);
	STATIC_ASSERT(sizeof(mf_pos_t) == 2);

	do {
		p[0] = v;
		p[1] = v;
		p[2] = v;
		p[3] = v;
		p += 4;
		size -= 4 * sizeof(*p);
	} while (size != 0);
}
#define matchfinder_init matchfinder_init_sse2

static forceinline void
matchfinder_rebase_sse2(mf_pos_t *data, size_t size)
{
	__m128i *p = (__m128i *)data;
	__m128i v = _mm_set1_epi16((u16)-MATCHFINDER_WINDOW_SIZE);

	STATIC_ASSERT(MATCHFINDER_MEM_ALIGNMENT % sizeof(*p) == 0);
	STATIC_ASSERT(MATCHFINDER_SIZE_ALIGNMENT % (4 * sizeof(*p)) == 0);
	STATIC_ASSERT(sizeof(mf_pos_t) == 2);

	do {
		/* PADDSW: Add Packed Signed Integers With Signed Saturation  */
		p[0] = _mm_adds_epi16(p[0], v);
		p[1] = _mm_adds_epi16(p[1], v);
		p[2] = _mm_adds_epi16(p[2], v);
		p[3] = _mm_adds_epi16(p[3], v);
		p += 4;
		size -= 4 * sizeof(*p);
	} while (size != 0);
}
#define matchfinder_rebase matchfinder_rebase_sse2
#endif /* HAVE_SSE2_NATIVE */

#endif /* LIB_X86_MATCHFINDER_IMPL_H */
