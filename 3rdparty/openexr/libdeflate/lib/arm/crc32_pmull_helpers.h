/*
 * arm/crc32_pmull_helpers.h - helper functions for CRC-32 folding with PMULL
 *
 * Copyright 2022 Eric Biggers
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
 * This file is a "template" for instantiating helper functions for CRC folding
 * with pmull instructions.  It accepts the following parameters:
 *
 * SUFFIX:
 *	Name suffix to append to all instantiated functions.
 * ATTRIBUTES:
 *	Target function attributes to use.
 * ENABLE_EOR3:
 *	Use the eor3 instruction (from the sha3 extension).
 */

#include <arm_neon.h>

/* Create a vector with 'a' in the first 4 bytes, and the rest zeroed out. */
#undef u32_to_bytevec
static forceinline ATTRIBUTES uint8x16_t
ADD_SUFFIX(u32_to_bytevec)(u32 a)
{
	return vreinterpretq_u8_u32(vsetq_lane_u32(a, vdupq_n_u32(0), 0));
}
#define u32_to_bytevec	ADD_SUFFIX(u32_to_bytevec)

/* Load two 64-bit values into a vector. */
#undef load_multipliers
static forceinline ATTRIBUTES poly64x2_t
ADD_SUFFIX(load_multipliers)(const u64 p[2])
{
	return vreinterpretq_p64_u64(vld1q_u64(p));
}
#define load_multipliers	ADD_SUFFIX(load_multipliers)

/* Do carryless multiplication of the low halves of two vectors. */
#undef clmul_low
static forceinline ATTRIBUTES uint8x16_t
ADD_SUFFIX(clmul_low)(uint8x16_t a, poly64x2_t b)
{
	return vreinterpretq_u8_p128(
		     compat_vmull_p64(vgetq_lane_p64(vreinterpretq_p64_u8(a), 0),
				      vgetq_lane_p64(b, 0)));
}
#define clmul_low	ADD_SUFFIX(clmul_low)

/* Do carryless multiplication of the high halves of two vectors. */
#undef clmul_high
static forceinline ATTRIBUTES uint8x16_t
ADD_SUFFIX(clmul_high)(uint8x16_t a, poly64x2_t b)
{
#if defined(__clang__) && defined(ARCH_ARM64)
	/*
	 * Use inline asm to ensure that pmull2 is really used.  This works
	 * around clang bug https://github.com/llvm/llvm-project/issues/52868.
	 */
	uint8x16_t res;

	__asm__("pmull2 %0.1q, %1.2d, %2.2d" : "=w" (res) : "w" (a), "w" (b));
	return res;
#else
	return vreinterpretq_u8_p128(vmull_high_p64(vreinterpretq_p64_u8(a), b));
#endif
}
#define clmul_high	ADD_SUFFIX(clmul_high)

#undef eor3
static forceinline ATTRIBUTES uint8x16_t
ADD_SUFFIX(eor3)(uint8x16_t a, uint8x16_t b, uint8x16_t c)
{
#if ENABLE_EOR3
#if HAVE_SHA3_INTRIN
	return veor3q_u8(a, b, c);
#else
	uint8x16_t res;

	__asm__("eor3 %0.16b, %1.16b, %2.16b, %3.16b"
		: "=w" (res) : "w" (a), "w" (b), "w" (c));
	return res;
#endif
#else /* ENABLE_EOR3 */
	return veorq_u8(veorq_u8(a, b), c);
#endif /* !ENABLE_EOR3 */
}
#define eor3	ADD_SUFFIX(eor3)

#undef fold_vec
static forceinline ATTRIBUTES uint8x16_t
ADD_SUFFIX(fold_vec)(uint8x16_t src, uint8x16_t dst, poly64x2_t multipliers)
{
	uint8x16_t a = clmul_low(src, multipliers);
	uint8x16_t b = clmul_high(src, multipliers);

	return eor3(a, b, dst);
}
#define fold_vec	ADD_SUFFIX(fold_vec)

#undef vtbl
static forceinline ATTRIBUTES uint8x16_t
ADD_SUFFIX(vtbl)(uint8x16_t table, uint8x16_t indices)
{
#ifdef ARCH_ARM64
	return vqtbl1q_u8(table, indices);
#else
	uint8x8x2_t tab2;

	tab2.val[0] = vget_low_u8(table);
	tab2.val[1] = vget_high_u8(table);

	return vcombine_u8(vtbl2_u8(tab2, vget_low_u8(indices)),
			   vtbl2_u8(tab2, vget_high_u8(indices)));
#endif
}
#define vtbl	ADD_SUFFIX(vtbl)

/*
 * Given v containing a 16-byte polynomial, and a pointer 'p' that points to the
 * next '1 <= len <= 15' data bytes, rearrange the concatenation of v and the
 * data into vectors x0 and x1 that contain 'len' bytes and 16 bytes,
 * respectively.  Then fold x0 into x1 and return the result.  Assumes that
 * 'p + len - 16' is in-bounds.
 */
#undef fold_partial_vec
static forceinline ATTRIBUTES MAYBE_UNUSED uint8x16_t
ADD_SUFFIX(fold_partial_vec)(uint8x16_t v, const u8 *p, size_t len,
			     poly64x2_t multipliers_1)
{
	/*
	 * vtbl(v, shift_tab[len..len+15]) left shifts v by 16-len bytes.
	 * vtbl(v, shift_tab[len+16..len+31]) right shifts v by len bytes.
	 */
	static const u8 shift_tab[48] = {
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	};
	const uint8x16_t lshift = vld1q_u8(&shift_tab[len]);
	const uint8x16_t rshift = vld1q_u8(&shift_tab[len + 16]);
	uint8x16_t x0, x1, bsl_mask;

	/* x0 = v left-shifted by '16 - len' bytes */
	x0 = vtbl(v, lshift);

	/* Create a vector of '16 - len' 0x00 bytes, then 'len' 0xff bytes. */
	bsl_mask = vreinterpretq_u8_s8(
			vshrq_n_s8(vreinterpretq_s8_u8(rshift), 7));

	/*
	 * x1 = the last '16 - len' bytes from v (i.e. v right-shifted by 'len'
	 * bytes) followed by the remaining data.
	 */
	x1 = vbslq_u8(bsl_mask /* 0 bits select from arg3, 1 bits from arg2 */,
		      vld1q_u8(p + len - 16), vtbl(v, rshift));

	return fold_vec(x0, x1, multipliers_1);
}
#define fold_partial_vec	ADD_SUFFIX(fold_partial_vec)
