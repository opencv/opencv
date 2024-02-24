/*
 * x86/adler32_impl.h - x86 implementations of Adler-32 checksum algorithm
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

#ifndef LIB_X86_ADLER32_IMPL_H
#define LIB_X86_ADLER32_IMPL_H

#include "cpu_features.h"

/*
 * The following macros horizontally sum the s1 counters and add them to the
 * real s1, and likewise for s2.  They do this via a series of reductions, each
 * of which halves the vector length, until just one counter remains.
 *
 * The s1 reductions don't depend on the s2 reductions and vice versa, so for
 * efficiency they are interleaved.  Also, every other s1 counter is 0 due to
 * the 'psadbw' instruction (_mm_sad_epu8) summing groups of 8 bytes rather than
 * 4; hence, one of the s1 reductions is skipped when going from 128 => 32 bits.
 */

#define ADLER32_FINISH_VEC_CHUNK_128(s1, s2, v_s1, v_s2)		    \
{									    \
	__m128i /* __v4su */ s1_last = (v_s1), s2_last = (v_s2);	    \
									    \
	/* 128 => 32 bits */						    \
	s2_last = _mm_add_epi32(s2_last, _mm_shuffle_epi32(s2_last, 0x31)); \
	s1_last = _mm_add_epi32(s1_last, _mm_shuffle_epi32(s1_last, 0x02)); \
	s2_last = _mm_add_epi32(s2_last, _mm_shuffle_epi32(s2_last, 0x02)); \
									    \
	*(s1) += (u32)_mm_cvtsi128_si32(s1_last);			    \
	*(s2) += (u32)_mm_cvtsi128_si32(s2_last);			    \
}

#define ADLER32_FINISH_VEC_CHUNK_256(s1, s2, v_s1, v_s2)		    \
{									    \
	__m128i /* __v4su */ s1_128bit, s2_128bit;			    \
									    \
	/* 256 => 128 bits */						    \
	s1_128bit = _mm_add_epi32(_mm256_extracti128_si256((v_s1), 0),	    \
				  _mm256_extracti128_si256((v_s1), 1));	    \
	s2_128bit = _mm_add_epi32(_mm256_extracti128_si256((v_s2), 0),	    \
				  _mm256_extracti128_si256((v_s2), 1));	    \
									    \
	ADLER32_FINISH_VEC_CHUNK_128((s1), (s2), s1_128bit, s2_128bit);	    \
}

/*
 * This is a very silly partial workaround for gcc bug
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107892.  The bug causes gcc to
 * generate extra move instructions in some loops containing vector intrinsics.
 *
 * An alternate workaround would be to use gcc native vector operations instead
 * of vector intrinsics.  But that would result in MSVC needing its own code.
 */
#if GCC_PREREQ(1, 0)
#  define GCC_UPDATE_VARS(a, b, c, d, e, f) \
	__asm__("" : "+x" (a), "+x" (b), "+x" (c), "+x" (d), "+x" (e), "+x" (f))
#else
#  define GCC_UPDATE_VARS(a, b, c, d, e, f) \
	(void)a, (void)b, (void)c, (void)d, (void)e, (void)f
#endif

/* SSE2 implementation */
#if HAVE_SSE2_INTRIN
#  define adler32_sse2		adler32_sse2
#  define FUNCNAME		adler32_sse2
#  define FUNCNAME_CHUNK	adler32_sse2_chunk
#  define IMPL_ALIGNMENT	16
#  define IMPL_SEGMENT_LEN	32
/*
 * The 16-bit precision byte counters must not be allowed to undergo *signed*
 * overflow, otherwise the signed multiplications at the end (_mm_madd_epi16)
 * would behave incorrectly.
 */
#  define IMPL_MAX_CHUNK_LEN	(32 * (0x7FFF / 0xFF))
#  if HAVE_SSE2_NATIVE
#    define ATTRIBUTES
#  else
#    define ATTRIBUTES		_target_attribute("sse2")
#  endif
#  include <emmintrin.h>
static forceinline ATTRIBUTES void
adler32_sse2_chunk(const __m128i *p, const __m128i *const end, u32 *s1, u32 *s2)
{
	const __m128i zeroes = _mm_setzero_si128();
	const __m128i /* __v8hu */ mults_a =
		_mm_setr_epi16(32, 31, 30, 29, 28, 27, 26, 25);
	const __m128i /* __v8hu */ mults_b =
		_mm_setr_epi16(24, 23, 22, 21, 20, 19, 18, 17);
	const __m128i /* __v8hu */ mults_c =
		_mm_setr_epi16(16, 15, 14, 13, 12, 11, 10, 9);
	const __m128i /* __v8hu */ mults_d =
		_mm_setr_epi16(8,  7,  6,  5,  4,  3,  2,  1);

	/* s1 counters: 32-bit, sum of bytes */
	__m128i /* __v4su */ v_s1 = zeroes;

	/* s2 counters: 32-bit, sum of s1 values */
	__m128i /* __v4su */ v_s2 = zeroes;

	/*
	 * Thirty-two 16-bit counters for byte sums.  Each accumulates the bytes
	 * that eventually need to be multiplied by a number 32...1 for addition
	 * into s2.
	 */
	__m128i /* __v8hu */ v_byte_sums_a = zeroes;
	__m128i /* __v8hu */ v_byte_sums_b = zeroes;
	__m128i /* __v8hu */ v_byte_sums_c = zeroes;
	__m128i /* __v8hu */ v_byte_sums_d = zeroes;

	do {
		/* Load the next 32 bytes. */
		const __m128i bytes1 = *p++;
		const __m128i bytes2 = *p++;

		/*
		 * Accumulate the previous s1 counters into the s2 counters.
		 * Logically, this really should be v_s2 += v_s1 * 32, but we
		 * can do the multiplication (or left shift) later.
		 */
		v_s2 = _mm_add_epi32(v_s2, v_s1);

		/*
		 * s1 update: use "Packed Sum of Absolute Differences" to add
		 * the bytes horizontally with 8 bytes per sum.  Then add the
		 * sums to the s1 counters.
		 */
		v_s1 = _mm_add_epi32(v_s1, _mm_sad_epu8(bytes1, zeroes));
		v_s1 = _mm_add_epi32(v_s1, _mm_sad_epu8(bytes2, zeroes));

		/*
		 * Also accumulate the bytes into 32 separate counters that have
		 * 16-bit precision.
		 */
		v_byte_sums_a = _mm_add_epi16(
			v_byte_sums_a, _mm_unpacklo_epi8(bytes1, zeroes));
		v_byte_sums_b = _mm_add_epi16(
			v_byte_sums_b, _mm_unpackhi_epi8(bytes1, zeroes));
		v_byte_sums_c = _mm_add_epi16(
			v_byte_sums_c, _mm_unpacklo_epi8(bytes2, zeroes));
		v_byte_sums_d = _mm_add_epi16(
			v_byte_sums_d, _mm_unpackhi_epi8(bytes2, zeroes));

		GCC_UPDATE_VARS(v_s1, v_s2, v_byte_sums_a, v_byte_sums_b,
				v_byte_sums_c, v_byte_sums_d);
	} while (p != end);

	/* Finish calculating the s2 counters. */
	v_s2 = _mm_slli_epi32(v_s2, 5);
	v_s2 = _mm_add_epi32(v_s2, _mm_madd_epi16(v_byte_sums_a, mults_a));
	v_s2 = _mm_add_epi32(v_s2, _mm_madd_epi16(v_byte_sums_b, mults_b));
	v_s2 = _mm_add_epi32(v_s2, _mm_madd_epi16(v_byte_sums_c, mults_c));
	v_s2 = _mm_add_epi32(v_s2, _mm_madd_epi16(v_byte_sums_d, mults_d));

	/* Add the counters to the real s1 and s2. */
	ADLER32_FINISH_VEC_CHUNK_128(s1, s2, v_s1, v_s2);
}
#  include "../adler32_vec_template.h"
#endif /* HAVE_SSE2_INTRIN */

/*
 * AVX2 implementation.  Basically the same as the SSE2 one, but with the vector
 * width doubled.
 */
#if HAVE_AVX2_INTRIN
#  define adler32_avx2		adler32_avx2
#  define FUNCNAME		adler32_avx2
#  define FUNCNAME_CHUNK	adler32_avx2_chunk
#  define IMPL_ALIGNMENT	32
#  define IMPL_SEGMENT_LEN	64
#  define IMPL_MAX_CHUNK_LEN	(64 * (0x7FFF / 0xFF))
#  if HAVE_AVX2_NATIVE
#    define ATTRIBUTES
#  else
#    define ATTRIBUTES		_target_attribute("avx2")
#  endif
#  include <immintrin.h>
  /*
   * With clang in MSVC compatibility mode, immintrin.h incorrectly skips
   * including some sub-headers.
   */
#  if defined(__clang__) && defined(_MSC_VER)
#    include <avxintrin.h>
#    include <avx2intrin.h>
#  endif
static forceinline ATTRIBUTES void
adler32_avx2_chunk(const __m256i *p, const __m256i *const end, u32 *s1, u32 *s2)
{
	const __m256i zeroes = _mm256_setzero_si256();
	/*
	 * Note, the multipliers have to be in this order because
	 * _mm256_unpack{lo,hi}_epi8 work on each 128-bit lane separately.
	 */
	const __m256i /* __v16hu */ mults_a =
		_mm256_setr_epi16(64, 63, 62, 61, 60, 59, 58, 57,
				  48, 47, 46, 45, 44, 43, 42, 41);
	const __m256i /* __v16hu */ mults_b =
		_mm256_setr_epi16(56, 55, 54, 53, 52, 51, 50, 49,
				  40, 39, 38, 37, 36, 35, 34, 33);
	const __m256i /* __v16hu */ mults_c =
		_mm256_setr_epi16(32, 31, 30, 29, 28, 27, 26, 25,
				  16, 15, 14, 13, 12, 11, 10,  9);
	const __m256i /* __v16hu */ mults_d =
		_mm256_setr_epi16(24, 23, 22, 21, 20, 19, 18, 17,
				  8,  7,  6,  5,  4,  3,  2,  1);
	__m256i /* __v8su */ v_s1 = zeroes;
	__m256i /* __v8su */ v_s2 = zeroes;
	__m256i /* __v16hu */ v_byte_sums_a = zeroes;
	__m256i /* __v16hu */ v_byte_sums_b = zeroes;
	__m256i /* __v16hu */ v_byte_sums_c = zeroes;
	__m256i /* __v16hu */ v_byte_sums_d = zeroes;

	do {
		const __m256i bytes1 = *p++;
		const __m256i bytes2 = *p++;

		v_s2 = _mm256_add_epi32(v_s2, v_s1);
		v_s1 = _mm256_add_epi32(v_s1, _mm256_sad_epu8(bytes1, zeroes));
		v_s1 = _mm256_add_epi32(v_s1, _mm256_sad_epu8(bytes2, zeroes));
		v_byte_sums_a = _mm256_add_epi16(
			v_byte_sums_a, _mm256_unpacklo_epi8(bytes1, zeroes));
		v_byte_sums_b = _mm256_add_epi16(
			v_byte_sums_b, _mm256_unpackhi_epi8(bytes1, zeroes));
		v_byte_sums_c = _mm256_add_epi16(
			v_byte_sums_c, _mm256_unpacklo_epi8(bytes2, zeroes));
		v_byte_sums_d = _mm256_add_epi16(
			v_byte_sums_d, _mm256_unpackhi_epi8(bytes2, zeroes));

		GCC_UPDATE_VARS(v_s1, v_s2, v_byte_sums_a, v_byte_sums_b,
				v_byte_sums_c, v_byte_sums_d);
	} while (p != end);

	v_s2 = _mm256_slli_epi32(v_s2, 6);
	v_s2 = _mm256_add_epi32(v_s2, _mm256_madd_epi16(v_byte_sums_a, mults_a));
	v_s2 = _mm256_add_epi32(v_s2, _mm256_madd_epi16(v_byte_sums_b, mults_b));
	v_s2 = _mm256_add_epi32(v_s2, _mm256_madd_epi16(v_byte_sums_c, mults_c));
	v_s2 = _mm256_add_epi32(v_s2, _mm256_madd_epi16(v_byte_sums_d, mults_d));
	ADLER32_FINISH_VEC_CHUNK_256(s1, s2, v_s1, v_s2);
}
#  include "../adler32_vec_template.h"
#endif /* HAVE_AVX2_INTRIN */

#if defined(adler32_avx2) && HAVE_AVX2_NATIVE
#define DEFAULT_IMPL	adler32_avx2
#else
static inline adler32_func_t
arch_select_adler32_func(void)
{
	const u32 features MAYBE_UNUSED = get_x86_cpu_features();

#ifdef adler32_avx2
	if (HAVE_AVX2(features))
		return adler32_avx2;
#endif
#ifdef adler32_sse2
	if (HAVE_SSE2(features))
		return adler32_sse2;
#endif
	return NULL;
}
#define arch_select_adler32_func	arch_select_adler32_func
#endif

#endif /* LIB_X86_ADLER32_IMPL_H */
