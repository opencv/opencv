/*
 * adler32_vec_template.h - template for vectorized Adler-32 implementations
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
 * This file contains a template for vectorized Adler-32 implementations.
 *
 * The inner loop between reductions modulo 65521 of an unvectorized Adler-32
 * implementation looks something like this:
 *
 *	do {
 *		s1 += *p;
 *		s2 += s1;
 *	} while (++p != chunk_end);
 *
 * For vectorized calculation of s1, we only need to sum the input bytes.  They
 * can be accumulated into multiple counters which are eventually summed
 * together.
 *
 * For vectorized calculation of s2, the basic idea is that for each iteration
 * that processes N bytes, we can perform the following vectorizable
 * calculation:
 *
 *	s2 += N*byte_1 + (N-1)*byte_2 + (N-2)*byte_3 + ... + 1*byte_N
 *
 * Or, equivalently, we can sum the byte_1...byte_N for each iteration into N
 * separate counters, then do the multiplications by N...1 just once at the end
 * rather than once per iteration.
 *
 * Also, we must account for how previous bytes will affect s2 by doing the
 * following at beginning of each iteration:
 *
 *	s2 += s1 * N
 *
 * Furthermore, like s1, "s2" can actually be multiple counters which are
 * eventually summed together.
 */

static u32 ATTRIBUTES MAYBE_UNUSED
FUNCNAME(u32 adler, const u8 *p, size_t len)
{
	const size_t max_chunk_len =
		MIN(MAX_CHUNK_LEN, IMPL_MAX_CHUNK_LEN) -
		(MIN(MAX_CHUNK_LEN, IMPL_MAX_CHUNK_LEN) % IMPL_SEGMENT_LEN);
	u32 s1 = adler & 0xFFFF;
	u32 s2 = adler >> 16;
	const u8 * const end = p + len;
	const u8 *vend;

	/* Process a byte at a time until the needed alignment is reached. */
	if (p != end && (uintptr_t)p % IMPL_ALIGNMENT) {
		do {
			s1 += *p++;
			s2 += s1;
		} while (p != end && (uintptr_t)p % IMPL_ALIGNMENT);
		s1 %= DIVISOR;
		s2 %= DIVISOR;
	}

	/*
	 * Process "chunks" of bytes using vector instructions.  Chunk lengths
	 * are limited to MAX_CHUNK_LEN, which guarantees that s1 and s2 never
	 * overflow before being reduced modulo DIVISOR.  For vector processing,
	 * chunk lengths are also made evenly divisible by IMPL_SEGMENT_LEN and
	 * may be further limited to IMPL_MAX_CHUNK_LEN.
	 */
	STATIC_ASSERT(IMPL_SEGMENT_LEN % IMPL_ALIGNMENT == 0);
	vend = end - ((size_t)(end - p) % IMPL_SEGMENT_LEN);
	while (p != vend) {
		size_t chunk_len = MIN((size_t)(vend - p), max_chunk_len);

		s2 += s1 * chunk_len;

		FUNCNAME_CHUNK((const void *)p, (const void *)(p + chunk_len),
			       &s1, &s2);

		p += chunk_len;
		s1 %= DIVISOR;
		s2 %= DIVISOR;
	}

	/* Process any remaining bytes. */
	if (p != end) {
		do {
			s1 += *p++;
			s2 += s1;
		} while (p != end);
		s1 %= DIVISOR;
		s2 %= DIVISOR;
	}

	return (s2 << 16) | s1;
}

#undef FUNCNAME
#undef FUNCNAME_CHUNK
#undef ATTRIBUTES
#undef IMPL_ALIGNMENT
#undef IMPL_SEGMENT_LEN
#undef IMPL_MAX_CHUNK_LEN
