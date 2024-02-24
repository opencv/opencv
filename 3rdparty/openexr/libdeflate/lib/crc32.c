/*
 * crc32.c - CRC-32 checksum algorithm for the gzip format
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
 * High-level description of CRC
 * =============================
 *
 * Consider a bit sequence 'bits[1...len]'.  Interpret 'bits' as the "message"
 * polynomial M(x) with coefficients in GF(2) (the field of integers modulo 2),
 * where the coefficient of 'x^i' is 'bits[len - i]'.  Then, compute:
 *
 *			R(x) = M(x)*x^n mod G(x)
 *
 * where G(x) is a selected "generator" polynomial of degree 'n'.  The remainder
 * R(x) is a polynomial of max degree 'n - 1'.  The CRC of 'bits' is R(x)
 * interpreted as a bitstring of length 'n'.
 *
 * CRC used in gzip
 * ================
 *
 * In the gzip format (RFC 1952):
 *
 *	- The bitstring to checksum is formed from the bytes of the uncompressed
 *	  data by concatenating the bits from the bytes in order, proceeding
 *	  from the low-order bit to the high-order bit within each byte.
 *
 *	- The generator polynomial G(x) is: x^32 + x^26 + x^23 + x^22 + x^16 +
 *	  x^12 + x^11 + x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + x + 1.
 *	  Consequently, the CRC length is 32 bits ("CRC-32").
 *
 *	- The highest order 32 coefficients of M(x)*x^n are inverted.
 *
 *	- All 32 coefficients of R(x) are inverted.
 *
 * The two inversions cause added leading and trailing zero bits to affect the
 * resulting CRC, whereas with a regular CRC such bits would have no effect on
 * the CRC.
 *
 * Computation and optimizations
 * =============================
 *
 * We can compute R(x) through "long division", maintaining only 32 bits of
 * state at any given time.  Multiplication by 'x' can be implemented as
 * right-shifting by 1 (assuming the polynomial<=>bitstring mapping where the
 * highest order bit represents the coefficient of x^0), and both addition and
 * subtraction can be implemented as bitwise exclusive OR (since we are working
 * in GF(2)).  Here is an unoptimized implementation:
 *
 *	static u32 crc32_gzip(const u8 *p, size_t len)
 *	{
 *		u32 crc = 0;
 *		const u32 divisor = 0xEDB88320;
 *
 *		for (size_t i = 0; i < len * 8 + 32; i++) {
 *			int bit;
 *			u32 multiple;
 *
 *			if (i < len * 8)
 *				bit = (p[i / 8] >> (i % 8)) & 1;
 *			else
 *				bit = 0; // one of the 32 appended 0 bits
 *
 *			if (i < 32) // the first 32 bits are inverted
 *				bit ^= 1;
 *
 *			if (crc & 1)
 *				multiple = divisor;
 *			else
 *				multiple = 0;
 *
 *			crc >>= 1;
 *			crc |= (u32)bit << 31;
 *			crc ^= multiple;
 *		}
 *
 *		return ~crc;
 *	}
 *
 * In this implementation, the 32-bit integer 'crc' maintains the remainder of
 * the currently processed portion of the message (with 32 zero bits appended)
 * when divided by the generator polynomial.  'crc' is the representation of
 * R(x), and 'divisor' is the representation of G(x) excluding the x^32
 * coefficient.  For each bit to process, we multiply R(x) by 'x^1', then add
 * 'x^0' if the new bit is a 1.  If this causes R(x) to gain a nonzero x^32
 * term, then we subtract G(x) from R(x).
 *
 * We can speed this up by taking advantage of the fact that XOR is commutative
 * and associative, so the order in which we combine the inputs into 'crc' is
 * unimportant.  And since each message bit we add doesn't affect the choice of
 * 'multiple' until 32 bits later, we need not actually add each message bit
 * until that point:
 *
 *	static u32 crc32_gzip(const u8 *p, size_t len)
 *	{
 *		u32 crc = ~0;
 *		const u32 divisor = 0xEDB88320;
 *
 *		for (size_t i = 0; i < len * 8; i++) {
 *			int bit;
 *			u32 multiple;
 *
 *			bit = (p[i / 8] >> (i % 8)) & 1;
 *			crc ^= bit;
 *			if (crc & 1)
 *				multiple = divisor;
 *			else
 *				multiple = 0;
 *			crc >>= 1;
 *			crc ^= multiple;
 *		}
 *
 *		return ~crc;
 *	}
 *
 * With the above implementation we get the effect of 32 appended 0 bits for
 * free; they never affect the choice of a divisor, nor would they change the
 * value of 'crc' if they were to be actually XOR'ed in.  And by starting with a
 * remainder of all 1 bits, we get the effect of complementing the first 32
 * message bits.
 *
 * The next optimization is to process the input in multi-bit units.  Suppose
 * that we insert the next 'n' message bits into the remainder.  Then we get an
 * intermediate remainder of length '32 + n' bits, and the CRC of the extra 'n'
 * bits is the amount by which the low 32 bits of the remainder will change as a
 * result of cancelling out those 'n' bits.  Taking n=8 (one byte) and
 * precomputing a table containing the CRC of each possible byte, we get
 * crc32_slice1() defined below.
 *
 * As a further optimization, we could increase the multi-bit unit size to 16.
 * However, that is inefficient because the table size explodes from 256 entries
 * (1024 bytes) to 65536 entries (262144 bytes), which wastes memory and won't
 * fit in L1 cache on typical processors.
 *
 * However, we can actually process 4 bytes at a time using 4 different tables
 * with 256 entries each.  Logically, we form a 64-bit intermediate remainder
 * and cancel out the high 32 bits in 8-bit chunks.  Bits 32-39 are cancelled
 * out by the CRC of those bits, whereas bits 40-47 are be cancelled out by the
 * CRC of those bits with 8 zero bits appended, and so on.
 *
 * In crc32_slice8(), this method is extended to 8 bytes at a time.  The
 * intermediate remainder (which we never actually store explicitly) is 96 bits.
 *
 * On CPUs that support fast carryless multiplication, CRCs can be computed even
 * more quickly via "folding".  See e.g. the x86 PCLMUL implementation.
 */

#include "lib_common.h"
#include "crc32_multipliers.h"
#include "crc32_tables.h"

/* This is the default implementation.  It uses the slice-by-8 method. */
static u32 MAYBE_UNUSED
crc32_slice8(u32 crc, const u8 *p, size_t len)
{
	const u8 * const end = p + len;
	const u8 *end64;

	for (; ((uintptr_t)p & 7) && p != end; p++)
		crc = (crc >> 8) ^ crc32_slice8_table[(u8)crc ^ *p];

	end64 = p + ((end - p) & ~7);
	for (; p != end64; p += 8) {
		u32 v1 = le32_bswap(*(const u32 *)(p + 0));
		u32 v2 = le32_bswap(*(const u32 *)(p + 4));

		crc = crc32_slice8_table[0x700 + (u8)((crc ^ v1) >> 0)] ^
		      crc32_slice8_table[0x600 + (u8)((crc ^ v1) >> 8)] ^
		      crc32_slice8_table[0x500 + (u8)((crc ^ v1) >> 16)] ^
		      crc32_slice8_table[0x400 + (u8)((crc ^ v1) >> 24)] ^
		      crc32_slice8_table[0x300 + (u8)(v2 >> 0)] ^
		      crc32_slice8_table[0x200 + (u8)(v2 >> 8)] ^
		      crc32_slice8_table[0x100 + (u8)(v2 >> 16)] ^
		      crc32_slice8_table[0x000 + (u8)(v2 >> 24)];
	}

	for (; p != end; p++)
		crc = (crc >> 8) ^ crc32_slice8_table[(u8)crc ^ *p];

	return crc;
}

/*
 * This is a more lightweight generic implementation, which can be used as a
 * subroutine by architecture-specific implementations to process small amounts
 * of unaligned data at the beginning and/or end of the buffer.
 */
static forceinline u32 MAYBE_UNUSED
crc32_slice1(u32 crc, const u8 *p, size_t len)
{
	size_t i;

	for (i = 0; i < len; i++)
		crc = (crc >> 8) ^ crc32_slice1_table[(u8)crc ^ p[i]];
	return crc;
}

/* Include architecture-specific implementation(s) if available. */
#undef DEFAULT_IMPL
#undef arch_select_crc32_func
typedef u32 (*crc32_func_t)(u32 crc, const u8 *p, size_t len);
#if defined(ARCH_ARM32) || defined(ARCH_ARM64)
#  include "arm/crc32_impl.h"
#elif defined(ARCH_X86_32) || defined(ARCH_X86_64)
#  include "x86/crc32_impl.h"
#endif

#ifndef DEFAULT_IMPL
#  define DEFAULT_IMPL crc32_slice8
#endif

#ifdef arch_select_crc32_func
static u32 dispatch_crc32(u32 crc, const u8 *p, size_t len);

static volatile crc32_func_t crc32_impl = dispatch_crc32;

/* Choose the best implementation at runtime. */
static u32 dispatch_crc32(u32 crc, const u8 *p, size_t len)
{
	crc32_func_t f = arch_select_crc32_func();

	if (f == NULL)
		f = DEFAULT_IMPL;

	crc32_impl = f;
	return f(crc, p, len);
}
#else
/* The best implementation is statically known, so call it directly. */
#define crc32_impl DEFAULT_IMPL
#endif

LIBDEFLATEAPI u32
libdeflate_crc32(u32 crc, const void *p, size_t len)
{
	if (p == NULL) /* Return initial value. */
		return 0;
	return ~crc32_impl(~crc, p, len);
}
