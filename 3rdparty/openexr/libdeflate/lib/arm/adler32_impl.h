/*
 * arm/adler32_impl.h - ARM implementations of Adler-32 checksum algorithm
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

#ifndef LIB_ARM_ADLER32_IMPL_H
#define LIB_ARM_ADLER32_IMPL_H

#include "cpu_features.h"

/* Regular NEON implementation */
#if HAVE_NEON_INTRIN && CPU_IS_LITTLE_ENDIAN()
#  define adler32_neon		adler32_neon
#  define FUNCNAME		adler32_neon
#  define FUNCNAME_CHUNK	adler32_neon_chunk
#  define IMPL_ALIGNMENT	16
#  define IMPL_SEGMENT_LEN	64
/* Prevent unsigned overflow of the 16-bit precision byte counters */
#  define IMPL_MAX_CHUNK_LEN	(64 * (0xFFFF / 0xFF))
#  if HAVE_NEON_NATIVE
#    define ATTRIBUTES
#  else
#    ifdef ARCH_ARM32
#      define ATTRIBUTES	_target_attribute("fpu=neon")
#    else
#      define ATTRIBUTES	_target_attribute("+simd")
#    endif
#  endif
#  include <arm_neon.h>
static forceinline ATTRIBUTES void
adler32_neon_chunk(const uint8x16_t *p, const uint8x16_t * const end,
		   u32 *s1, u32 *s2)
{
	static const u16 _aligned_attribute(16) mults[64] = {
		64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
		48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
		32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
		16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,
	};
	const uint16x8_t mults_a = vld1q_u16(&mults[0]);
	const uint16x8_t mults_b = vld1q_u16(&mults[8]);
	const uint16x8_t mults_c = vld1q_u16(&mults[16]);
	const uint16x8_t mults_d = vld1q_u16(&mults[24]);
	const uint16x8_t mults_e = vld1q_u16(&mults[32]);
	const uint16x8_t mults_f = vld1q_u16(&mults[40]);
	const uint16x8_t mults_g = vld1q_u16(&mults[48]);
	const uint16x8_t mults_h = vld1q_u16(&mults[56]);

	uint32x4_t v_s1 = vdupq_n_u32(0);
	uint32x4_t v_s2 = vdupq_n_u32(0);
	/*
	 * v_byte_sums_* contain the sum of the bytes at index i across all
	 * 64-byte segments, for each index 0..63.
	 */
	uint16x8_t v_byte_sums_a = vdupq_n_u16(0);
	uint16x8_t v_byte_sums_b = vdupq_n_u16(0);
	uint16x8_t v_byte_sums_c = vdupq_n_u16(0);
	uint16x8_t v_byte_sums_d = vdupq_n_u16(0);
	uint16x8_t v_byte_sums_e = vdupq_n_u16(0);
	uint16x8_t v_byte_sums_f = vdupq_n_u16(0);
	uint16x8_t v_byte_sums_g = vdupq_n_u16(0);
	uint16x8_t v_byte_sums_h = vdupq_n_u16(0);

	do {
		/* Load the next 64 bytes. */
		const uint8x16_t bytes1 = *p++;
		const uint8x16_t bytes2 = *p++;
		const uint8x16_t bytes3 = *p++;
		const uint8x16_t bytes4 = *p++;
		uint16x8_t tmp;

		/*
		 * Accumulate the previous s1 counters into the s2 counters.
		 * The needed multiplication by 64 is delayed to later.
		 */
		v_s2 = vaddq_u32(v_s2, v_s1);

		/*
		 * Add the 64 bytes to their corresponding v_byte_sums counters,
		 * while also accumulating the sums of each adjacent set of 4
		 * bytes into v_s1.
		 */
		tmp = vpaddlq_u8(bytes1);
		v_byte_sums_a = vaddw_u8(v_byte_sums_a, vget_low_u8(bytes1));
		v_byte_sums_b = vaddw_u8(v_byte_sums_b, vget_high_u8(bytes1));
		tmp = vpadalq_u8(tmp, bytes2);
		v_byte_sums_c = vaddw_u8(v_byte_sums_c, vget_low_u8(bytes2));
		v_byte_sums_d = vaddw_u8(v_byte_sums_d, vget_high_u8(bytes2));
		tmp = vpadalq_u8(tmp, bytes3);
		v_byte_sums_e = vaddw_u8(v_byte_sums_e, vget_low_u8(bytes3));
		v_byte_sums_f = vaddw_u8(v_byte_sums_f, vget_high_u8(bytes3));
		tmp = vpadalq_u8(tmp, bytes4);
		v_byte_sums_g = vaddw_u8(v_byte_sums_g, vget_low_u8(bytes4));
		v_byte_sums_h = vaddw_u8(v_byte_sums_h, vget_high_u8(bytes4));
		v_s1 = vpadalq_u16(v_s1, tmp);

	} while (p != end);

	/* s2 = 64*s2 + (64*bytesum0 + 63*bytesum1 + ... + 1*bytesum63) */
#ifdef ARCH_ARM32
#  define umlal2(a, b, c)  vmlal_u16((a), vget_high_u16(b), vget_high_u16(c))
#else
#  define umlal2	   vmlal_high_u16
#endif
	v_s2 = vqshlq_n_u32(v_s2, 6);
	v_s2 = vmlal_u16(v_s2, vget_low_u16(v_byte_sums_a), vget_low_u16(mults_a));
	v_s2 = umlal2(v_s2, v_byte_sums_a, mults_a);
	v_s2 = vmlal_u16(v_s2, vget_low_u16(v_byte_sums_b), vget_low_u16(mults_b));
	v_s2 = umlal2(v_s2, v_byte_sums_b, mults_b);
	v_s2 = vmlal_u16(v_s2, vget_low_u16(v_byte_sums_c), vget_low_u16(mults_c));
	v_s2 = umlal2(v_s2, v_byte_sums_c, mults_c);
	v_s2 = vmlal_u16(v_s2, vget_low_u16(v_byte_sums_d), vget_low_u16(mults_d));
	v_s2 = umlal2(v_s2, v_byte_sums_d, mults_d);
	v_s2 = vmlal_u16(v_s2, vget_low_u16(v_byte_sums_e), vget_low_u16(mults_e));
	v_s2 = umlal2(v_s2, v_byte_sums_e, mults_e);
	v_s2 = vmlal_u16(v_s2, vget_low_u16(v_byte_sums_f), vget_low_u16(mults_f));
	v_s2 = umlal2(v_s2, v_byte_sums_f, mults_f);
	v_s2 = vmlal_u16(v_s2, vget_low_u16(v_byte_sums_g), vget_low_u16(mults_g));
	v_s2 = umlal2(v_s2, v_byte_sums_g, mults_g);
	v_s2 = vmlal_u16(v_s2, vget_low_u16(v_byte_sums_h), vget_low_u16(mults_h));
	v_s2 = umlal2(v_s2, v_byte_sums_h, mults_h);
#undef umlal2

	/* Horizontal sum to finish up */
#ifdef ARCH_ARM32
	*s1 += vgetq_lane_u32(v_s1, 0) + vgetq_lane_u32(v_s1, 1) +
	       vgetq_lane_u32(v_s1, 2) + vgetq_lane_u32(v_s1, 3);
	*s2 += vgetq_lane_u32(v_s2, 0) + vgetq_lane_u32(v_s2, 1) +
	       vgetq_lane_u32(v_s2, 2) + vgetq_lane_u32(v_s2, 3);
#else
	*s1 += vaddvq_u32(v_s1);
	*s2 += vaddvq_u32(v_s2);
#endif
}
#  include "../adler32_vec_template.h"
#endif /* Regular NEON implementation */

/* NEON+dotprod implementation */
#if HAVE_DOTPROD_INTRIN && CPU_IS_LITTLE_ENDIAN()
#  define adler32_neon_dotprod	adler32_neon_dotprod
#  define FUNCNAME		adler32_neon_dotprod
#  define FUNCNAME_CHUNK	adler32_neon_dotprod_chunk
#  define IMPL_ALIGNMENT	16
#  define IMPL_SEGMENT_LEN	64
#  define IMPL_MAX_CHUNK_LEN	MAX_CHUNK_LEN
#  if HAVE_DOTPROD_NATIVE
#    define ATTRIBUTES
#  else
#    ifdef __clang__
#      define ATTRIBUTES  _target_attribute("dotprod")
     /*
      * With gcc, arch=armv8.2-a is needed for dotprod intrinsics, unless the
      * default target is armv8.3-a or later in which case it must be omitted.
      * armv8.3-a or later can be detected by checking for __ARM_FEATURE_JCVT.
      */
#    elif defined(__ARM_FEATURE_JCVT)
#      define ATTRIBUTES  _target_attribute("+dotprod")
#    else
#      define ATTRIBUTES  _target_attribute("arch=armv8.2-a+dotprod")
#    endif
#  endif
#  include <arm_neon.h>
static forceinline ATTRIBUTES void
adler32_neon_dotprod_chunk(const uint8x16_t *p, const uint8x16_t * const end,
			   u32 *s1, u32 *s2)
{
	static const u8 _aligned_attribute(16) mults[64] = {
		64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
		48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
		32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
		16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,
	};
	const uint8x16_t mults_a = vld1q_u8(&mults[0]);
	const uint8x16_t mults_b = vld1q_u8(&mults[16]);
	const uint8x16_t mults_c = vld1q_u8(&mults[32]);
	const uint8x16_t mults_d = vld1q_u8(&mults[48]);
	const uint8x16_t ones = vdupq_n_u8(1);
	uint32x4_t v_s1_a = vdupq_n_u32(0);
	uint32x4_t v_s1_b = vdupq_n_u32(0);
	uint32x4_t v_s1_c = vdupq_n_u32(0);
	uint32x4_t v_s1_d = vdupq_n_u32(0);
	uint32x4_t v_s2_a = vdupq_n_u32(0);
	uint32x4_t v_s2_b = vdupq_n_u32(0);
	uint32x4_t v_s2_c = vdupq_n_u32(0);
	uint32x4_t v_s2_d = vdupq_n_u32(0);
	uint32x4_t v_s1_sums_a = vdupq_n_u32(0);
	uint32x4_t v_s1_sums_b = vdupq_n_u32(0);
	uint32x4_t v_s1_sums_c = vdupq_n_u32(0);
	uint32x4_t v_s1_sums_d = vdupq_n_u32(0);
	uint32x4_t v_s1;
	uint32x4_t v_s2;
	uint32x4_t v_s1_sums;

	do {
		uint8x16_t bytes_a = *p++;
		uint8x16_t bytes_b = *p++;
		uint8x16_t bytes_c = *p++;
		uint8x16_t bytes_d = *p++;

		v_s1_sums_a = vaddq_u32(v_s1_sums_a, v_s1_a);
		v_s1_a = vdotq_u32(v_s1_a, bytes_a, ones);
		v_s2_a = vdotq_u32(v_s2_a, bytes_a, mults_a);

		v_s1_sums_b = vaddq_u32(v_s1_sums_b, v_s1_b);
		v_s1_b = vdotq_u32(v_s1_b, bytes_b, ones);
		v_s2_b = vdotq_u32(v_s2_b, bytes_b, mults_b);

		v_s1_sums_c = vaddq_u32(v_s1_sums_c, v_s1_c);
		v_s1_c = vdotq_u32(v_s1_c, bytes_c, ones);
		v_s2_c = vdotq_u32(v_s2_c, bytes_c, mults_c);

		v_s1_sums_d = vaddq_u32(v_s1_sums_d, v_s1_d);
		v_s1_d = vdotq_u32(v_s1_d, bytes_d, ones);
		v_s2_d = vdotq_u32(v_s2_d, bytes_d, mults_d);
	} while (p != end);

	v_s1 = vaddq_u32(vaddq_u32(v_s1_a, v_s1_b), vaddq_u32(v_s1_c, v_s1_d));
	v_s2 = vaddq_u32(vaddq_u32(v_s2_a, v_s2_b), vaddq_u32(v_s2_c, v_s2_d));
	v_s1_sums = vaddq_u32(vaddq_u32(v_s1_sums_a, v_s1_sums_b),
			      vaddq_u32(v_s1_sums_c, v_s1_sums_d));
	v_s2 = vaddq_u32(v_s2, vqshlq_n_u32(v_s1_sums, 6));

	*s1 += vaddvq_u32(v_s1);
	*s2 += vaddvq_u32(v_s2);
}
#  include "../adler32_vec_template.h"
#endif /* NEON+dotprod implementation */

#if defined(adler32_neon_dotprod) && HAVE_DOTPROD_NATIVE
#define DEFAULT_IMPL	adler32_neon_dotprod
#else
static inline adler32_func_t
arch_select_adler32_func(void)
{
	const u32 features MAYBE_UNUSED = get_arm_cpu_features();

#ifdef adler32_neon_dotprod
	if (HAVE_NEON(features) && HAVE_DOTPROD(features))
		return adler32_neon_dotprod;
#endif
#ifdef adler32_neon
	if (HAVE_NEON(features))
		return adler32_neon;
#endif
	return NULL;
}
#define arch_select_adler32_func	arch_select_adler32_func
#endif

#endif /* LIB_ARM_ADLER32_IMPL_H */
