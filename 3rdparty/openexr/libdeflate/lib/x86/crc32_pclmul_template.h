/*
 * x86/crc32_pclmul_template.h - gzip CRC-32 with PCLMULQDQ instructions
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

/*
 * This file is a "template" for instantiating PCLMULQDQ-based crc32_x86
 * functions.  The "parameters" are:
 *
 * SUFFIX:
 *	Name suffix to append to all instantiated functions.
 * ATTRIBUTES:
 *	Target function attributes to use.
 * FOLD_PARTIAL_VECS:
 *	Use vector instructions to handle any partial blocks at the beginning
 *	and end, instead of falling back to scalar instructions for those parts.
 *	Requires SSSE3 and SSE4.1 intrinsics.
 *
 * The overall algorithm used is CRC folding with carryless multiplication
 * instructions.  Note that the x86 crc32 instruction cannot be used, as it is
 * for a different polynomial, not the gzip one.  For an explanation of CRC
 * folding with carryless multiplication instructions, see
 * scripts/gen_crc32_multipliers.c and the following paper:
 *
 *	"Fast CRC Computation for Generic Polynomials Using PCLMULQDQ Instruction"
 *	https://www.intel.com/content/dam/www/public/us/en/documents/white-papers/fast-crc-computation-generic-polynomials-pclmulqdq-paper.pdf
 */

#include <immintrin.h>
/*
 * With clang in MSVC compatibility mode, immintrin.h incorrectly skips
 * including some sub-headers.
 */
#if defined(__clang__) && defined(_MSC_VER)
#  include <tmmintrin.h>
#  include <smmintrin.h>
#  include <wmmintrin.h>
#endif

#undef fold_vec
static forceinline ATTRIBUTES __m128i
ADD_SUFFIX(fold_vec)(__m128i src, __m128i dst, __m128i /* __v2di */ multipliers)
{
	/*
	 * The immediate constant for PCLMULQDQ specifies which 64-bit halves of
	 * the 128-bit vectors to multiply:
	 *
	 * 0x00 means low halves (higher degree polynomial terms for us)
	 * 0x11 means high halves (lower degree polynomial terms for us)
	 */
	dst = _mm_xor_si128(dst, _mm_clmulepi64_si128(src, multipliers, 0x00));
	dst = _mm_xor_si128(dst, _mm_clmulepi64_si128(src, multipliers, 0x11));
	return dst;
}
#define fold_vec	ADD_SUFFIX(fold_vec)

#if FOLD_PARTIAL_VECS
/*
 * Given v containing a 16-byte polynomial, and a pointer 'p' that points to the
 * next '1 <= len <= 15' data bytes, rearrange the concatenation of v and the
 * data into vectors x0 and x1 that contain 'len' bytes and 16 bytes,
 * respectively.  Then fold x0 into x1 and return the result.  Assumes that
 * 'p + len - 16' is in-bounds.
 */
#undef fold_partial_vec
static forceinline ATTRIBUTES __m128i
ADD_SUFFIX(fold_partial_vec)(__m128i v, const u8 *p, size_t len,
			     __m128i /* __v2du */ multipliers_1)
{
	/*
	 * pshufb(v, shift_tab[len..len+15]) left shifts v by 16-len bytes.
	 * pshufb(v, shift_tab[len+16..len+31]) right shifts v by len bytes.
	 */
	static const u8 shift_tab[48] = {
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	};
	__m128i lshift = _mm_loadu_si128((const void *)&shift_tab[len]);
	__m128i rshift = _mm_loadu_si128((const void *)&shift_tab[len + 16]);
	__m128i x0, x1;

	/* x0 = v left-shifted by '16 - len' bytes */
	x0 = _mm_shuffle_epi8(v, lshift);

	/*
	 * x1 = the last '16 - len' bytes from v (i.e. v right-shifted by 'len'
	 * bytes) followed by the remaining data.
	 */
	x1 = _mm_blendv_epi8(_mm_shuffle_epi8(v, rshift),
			     _mm_loadu_si128((const void *)(p + len - 16)),
			     /* msb 0/1 of each byte selects byte from arg1/2 */
			     rshift);

	return fold_vec(x0, x1, multipliers_1);
}
#define fold_partial_vec	ADD_SUFFIX(fold_partial_vec)
#endif /* FOLD_PARTIAL_VECS */

static u32 ATTRIBUTES MAYBE_UNUSED
ADD_SUFFIX(crc32_x86)(u32 crc, const u8 *p, size_t len)
{
	const __m128i /* __v2du */ multipliers_8 =
		_mm_set_epi64x(CRC32_8VECS_MULT_2, CRC32_8VECS_MULT_1);
	const __m128i /* __v2du */ multipliers_4 =
		_mm_set_epi64x(CRC32_4VECS_MULT_2, CRC32_4VECS_MULT_1);
	const __m128i /* __v2du */ multipliers_2 =
		_mm_set_epi64x(CRC32_2VECS_MULT_2, CRC32_2VECS_MULT_1);
	const __m128i /* __v2du */ multipliers_1 =
		_mm_set_epi64x(CRC32_1VECS_MULT_2, CRC32_1VECS_MULT_1);
	const __m128i /* __v2du */ final_multiplier =
		_mm_set_epi64x(0, CRC32_FINAL_MULT);
	const __m128i mask32 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
	const __m128i /* __v2du */ barrett_reduction_constants =
		_mm_set_epi64x(CRC32_BARRETT_CONSTANT_2,
			       CRC32_BARRETT_CONSTANT_1);
	__m128i v0, v1, v2, v3, v4, v5, v6, v7;

	/*
	 * There are two overall code paths.  The first path supports all
	 * lengths, but is intended for short lengths; it uses unaligned loads
	 * and does at most 4-way folds.  The second path only supports longer
	 * lengths, aligns the pointer in order to do aligned loads, and does up
	 * to 8-way folds.  The length check below decides which path to take.
	 */
	if (len < 1024) {
		if (len < 16)
			return crc32_slice1(crc, p, len);

		v0 = _mm_xor_si128(_mm_loadu_si128((const void *)p),
				   _mm_cvtsi32_si128(crc));
		p += 16;

		if (len >= 64) {
			v1 = _mm_loadu_si128((const void *)(p + 0));
			v2 = _mm_loadu_si128((const void *)(p + 16));
			v3 = _mm_loadu_si128((const void *)(p + 32));
			p += 48;
			while (len >= 64 + 64) {
				v0 = fold_vec(v0, _mm_loadu_si128((const void *)(p + 0)),
					      multipliers_4);
				v1 = fold_vec(v1, _mm_loadu_si128((const void *)(p + 16)),
					      multipliers_4);
				v2 = fold_vec(v2, _mm_loadu_si128((const void *)(p + 32)),
					      multipliers_4);
				v3 = fold_vec(v3, _mm_loadu_si128((const void *)(p + 48)),
					      multipliers_4);
				p += 64;
				len -= 64;
			}
			v0 = fold_vec(v0, v2, multipliers_2);
			v1 = fold_vec(v1, v3, multipliers_2);
			if (len & 32) {
				v0 = fold_vec(v0, _mm_loadu_si128((const void *)(p + 0)),
					      multipliers_2);
				v1 = fold_vec(v1, _mm_loadu_si128((const void *)(p + 16)),
					      multipliers_2);
				p += 32;
			}
			v0 = fold_vec(v0, v1, multipliers_1);
			if (len & 16) {
				v0 = fold_vec(v0, _mm_loadu_si128((const void *)p),
					      multipliers_1);
				p += 16;
			}
		} else {
			if (len >= 32) {
				v0 = fold_vec(v0, _mm_loadu_si128((const void *)p),
					      multipliers_1);
				p += 16;
				if (len >= 48) {
					v0 = fold_vec(v0, _mm_loadu_si128((const void *)p),
						      multipliers_1);
					p += 16;
				}
			}
		}
	} else {
		const size_t align = -(uintptr_t)p & 15;
		const __m128i *vp;

	#if FOLD_PARTIAL_VECS
		v0 = _mm_xor_si128(_mm_loadu_si128((const void *)p),
				   _mm_cvtsi32_si128(crc));
		p += 16;
		/* Align p to the next 16-byte boundary. */
		if (align) {
			v0 = fold_partial_vec(v0, p, align, multipliers_1);
			p += align;
			len -= align;
		}
		vp = (const __m128i *)p;
	#else
		/* Align p to the next 16-byte boundary. */
		if (align) {
			crc = crc32_slice1(crc, p, align);
			p += align;
			len -= align;
		}
		vp = (const __m128i *)p;
		v0 = _mm_xor_si128(*vp++, _mm_cvtsi32_si128(crc));
	#endif
		v1 = *vp++;
		v2 = *vp++;
		v3 = *vp++;
		v4 = *vp++;
		v5 = *vp++;
		v6 = *vp++;
		v7 = *vp++;
		do {
			v0 = fold_vec(v0, *vp++, multipliers_8);
			v1 = fold_vec(v1, *vp++, multipliers_8);
			v2 = fold_vec(v2, *vp++, multipliers_8);
			v3 = fold_vec(v3, *vp++, multipliers_8);
			v4 = fold_vec(v4, *vp++, multipliers_8);
			v5 = fold_vec(v5, *vp++, multipliers_8);
			v6 = fold_vec(v6, *vp++, multipliers_8);
			v7 = fold_vec(v7, *vp++, multipliers_8);
			len -= 128;
		} while (len >= 128 + 128);

		v0 = fold_vec(v0, v4, multipliers_4);
		v1 = fold_vec(v1, v5, multipliers_4);
		v2 = fold_vec(v2, v6, multipliers_4);
		v3 = fold_vec(v3, v7, multipliers_4);
		if (len & 64) {
			v0 = fold_vec(v0, *vp++, multipliers_4);
			v1 = fold_vec(v1, *vp++, multipliers_4);
			v2 = fold_vec(v2, *vp++, multipliers_4);
			v3 = fold_vec(v3, *vp++, multipliers_4);
		}

		v0 = fold_vec(v0, v2, multipliers_2);
		v1 = fold_vec(v1, v3, multipliers_2);
		if (len & 32) {
			v0 = fold_vec(v0, *vp++, multipliers_2);
			v1 = fold_vec(v1, *vp++, multipliers_2);
		}

		v0 = fold_vec(v0, v1, multipliers_1);
		if (len & 16)
			v0 = fold_vec(v0, *vp++, multipliers_1);

		p = (const u8 *)vp;
	}
	len &= 15;

	/*
	 * If fold_partial_vec() is available, handle any remaining partial
	 * block now before reducing to 32 bits.
	 */
#if FOLD_PARTIAL_VECS
	if (len)
		v0 = fold_partial_vec(v0, p, len, multipliers_1);
#endif

	/*
	 * Fold 128 => 96 bits.  This also implicitly appends 32 zero bits,
	 * which is equivalent to multiplying by x^32.  This is needed because
	 * the CRC is defined as M(x)*x^32 mod G(x), not just M(x) mod G(x).
	 */
	v0 = _mm_xor_si128(_mm_srli_si128(v0, 8),
			   _mm_clmulepi64_si128(v0, multipliers_1, 0x10));

	/* Fold 96 => 64 bits. */
	v0 = _mm_xor_si128(_mm_srli_si128(v0, 4),
			   _mm_clmulepi64_si128(_mm_and_si128(v0, mask32),
						final_multiplier, 0x00));

	/*
	 * Reduce 64 => 32 bits using Barrett reduction.
	 *
	 * Let M(x) = A(x)*x^32 + B(x) be the remaining message.  The goal is to
	 * compute R(x) = M(x) mod G(x).  Since degree(B(x)) < degree(G(x)):
	 *
	 *	R(x) = (A(x)*x^32 + B(x)) mod G(x)
	 *	     = (A(x)*x^32) mod G(x) + B(x)
	 *
	 * Then, by the Division Algorithm there exists a unique q(x) such that:
	 *
	 *	A(x)*x^32 mod G(x) = A(x)*x^32 - q(x)*G(x)
	 *
	 * Since the left-hand side is of maximum degree 31, the right-hand side
	 * must be too.  This implies that we can apply 'mod x^32' to the
	 * right-hand side without changing its value:
	 *
	 *	(A(x)*x^32 - q(x)*G(x)) mod x^32 = q(x)*G(x) mod x^32
	 *
	 * Note that '+' is equivalent to '-' in polynomials over GF(2).
	 *
	 * We also know that:
	 *
	 *	              / A(x)*x^32 \
	 *	q(x) = floor (  ---------  )
	 *	              \    G(x)   /
	 *
	 * To compute this efficiently, we can multiply the top and bottom by
	 * x^32 and move the division by G(x) to the top:
	 *
	 *	              / A(x) * floor(x^64 / G(x)) \
	 *	q(x) = floor (  -------------------------  )
	 *	              \           x^32            /
	 *
	 * Note that floor(x^64 / G(x)) is a constant.
	 *
	 * So finally we have:
	 *
	 *	                          / A(x) * floor(x^64 / G(x)) \
	 *	R(x) = B(x) + G(x)*floor (  -------------------------  )
	 *	                          \           x^32            /
	 */
	v1 = _mm_clmulepi64_si128(_mm_and_si128(v0, mask32),
				  barrett_reduction_constants, 0x00);
	v1 = _mm_clmulepi64_si128(_mm_and_si128(v1, mask32),
				  barrett_reduction_constants, 0x10);
	v0 = _mm_xor_si128(v0, v1);
#if FOLD_PARTIAL_VECS
	crc = _mm_extract_epi32(v0, 1);
#else
	crc = _mm_cvtsi128_si32(_mm_shuffle_epi32(v0, 0x01));
	/* Process up to 15 bytes left over at the end. */
	crc = crc32_slice1(crc, p, len);
#endif
	return crc;
}

#undef SUFFIX
#undef ATTRIBUTES
#undef FOLD_PARTIAL_VECS
