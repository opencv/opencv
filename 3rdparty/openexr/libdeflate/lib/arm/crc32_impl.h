/*
 * arm/crc32_impl.h - ARM implementations of the gzip CRC-32 algorithm
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

#ifndef LIB_ARM_CRC32_IMPL_H
#define LIB_ARM_CRC32_IMPL_H

#include "cpu_features.h"

/*
 * crc32_arm_crc() - implementation using crc32 instructions (only)
 *
 * In general this implementation is straightforward.  However, naive use of the
 * crc32 instructions is serial: one of the two inputs to each crc32 instruction
 * is the output of the previous one.  To take advantage of CPUs that can
 * execute multiple crc32 instructions in parallel, when possible we interleave
 * the checksumming of several adjacent chunks, then combine their CRCs.
 *
 * However, without pmull, combining CRCs is fairly slow.  So in this pmull-less
 * version, we only use a large chunk length, and thus we only do chunked
 * processing if there is a lot of data to checksum.  This also means that a
 * variable chunk length wouldn't help much, so we just support a fixed length.
 */
#if HAVE_CRC32_INTRIN
#  if HAVE_CRC32_NATIVE
#    define ATTRIBUTES
#  else
#    ifdef ARCH_ARM32
#      ifdef __clang__
#        define ATTRIBUTES	_target_attribute("armv8-a,crc")
#      elif defined(__ARM_PCS_VFP)
	 /*
	  * +simd is needed to avoid a "selected architecture lacks an FPU"
	  * error with Debian arm-linux-gnueabihf-gcc when -mfpu is not
	  * explicitly specified on the command line.
	  */
#        define ATTRIBUTES	_target_attribute("arch=armv8-a+crc+simd")
#      else
#        define ATTRIBUTES	_target_attribute("arch=armv8-a+crc")
#      endif
#    else
#      ifdef __clang__
#        define ATTRIBUTES	_target_attribute("crc")
#      else
#        define ATTRIBUTES	_target_attribute("+crc")
#      endif
#    endif
#  endif

#ifndef _MSC_VER
#  include <arm_acle.h>
#endif

/*
 * Combine the CRCs for 4 adjacent chunks of length L = CRC32_FIXED_CHUNK_LEN
 * bytes each by computing:
 *
 *	[ crc0*x^(3*8*L) + crc1*x^(2*8*L) + crc2*x^(1*8*L) + crc3 ] mod G(x)
 *
 * This has been optimized in several ways:
 *
 *    - The needed multipliers (x to some power, reduced mod G(x)) were
 *	precomputed.
 *
 *    - The 3 multiplications are interleaved.
 *
 *    - The reduction mod G(x) is delayed to the end and done using __crc32d.
 *	Note that the use of __crc32d introduces an extra factor of x^32.  To
 *	cancel that out along with the extra factor of x^1 that gets introduced
 *	because of how the 63-bit products are aligned in their 64-bit integers,
 *	the multipliers are actually x^(j*8*L - 33) instead of x^(j*8*L).
 */
static forceinline ATTRIBUTES u32
combine_crcs_slow(u32 crc0, u32 crc1, u32 crc2, u32 crc3)
{
	u64 res0 = 0, res1 = 0, res2 = 0;
	int i;

	/* Multiply crc{0,1,2} by CRC32_FIXED_CHUNK_MULT_{3,2,1}. */
	for (i = 0; i < 32; i++) {
		if (CRC32_FIXED_CHUNK_MULT_3 & (1U << i))
			res0 ^= (u64)crc0 << i;
		if (CRC32_FIXED_CHUNK_MULT_2 & (1U << i))
			res1 ^= (u64)crc1 << i;
		if (CRC32_FIXED_CHUNK_MULT_1 & (1U << i))
			res2 ^= (u64)crc2 << i;
	}
	/* Add the different parts and reduce mod G(x). */
	return __crc32d(0, res0 ^ res1 ^ res2) ^ crc3;
}

#define crc32_arm_crc	crc32_arm_crc
static u32 ATTRIBUTES MAYBE_UNUSED
crc32_arm_crc(u32 crc, const u8 *p, size_t len)
{
	if (len >= 64) {
		const size_t align = -(uintptr_t)p & 7;

		/* Align p to the next 8-byte boundary. */
		if (align) {
			if (align & 1)
				crc = __crc32b(crc, *p++);
			if (align & 2) {
				crc = __crc32h(crc, le16_bswap(*(u16 *)p));
				p += 2;
			}
			if (align & 4) {
				crc = __crc32w(crc, le32_bswap(*(u32 *)p));
				p += 4;
			}
			len -= align;
		}
		/*
		 * Interleave the processing of multiple adjacent data chunks to
		 * take advantage of instruction-level parallelism.
		 *
		 * Some CPUs don't prefetch the data if it's being fetched in
		 * multiple interleaved streams, so do explicit prefetching.
		 */
		while (len >= CRC32_NUM_CHUNKS * CRC32_FIXED_CHUNK_LEN) {
			const u64 *wp0 = (const u64 *)p;
			const u64 * const wp0_end =
				(const u64 *)(p + CRC32_FIXED_CHUNK_LEN);
			u32 crc1 = 0, crc2 = 0, crc3 = 0;

			STATIC_ASSERT(CRC32_NUM_CHUNKS == 4);
			STATIC_ASSERT(CRC32_FIXED_CHUNK_LEN % (4 * 8) == 0);
			do {
				prefetchr(&wp0[64 + 0*CRC32_FIXED_CHUNK_LEN/8]);
				prefetchr(&wp0[64 + 1*CRC32_FIXED_CHUNK_LEN/8]);
				prefetchr(&wp0[64 + 2*CRC32_FIXED_CHUNK_LEN/8]);
				prefetchr(&wp0[64 + 3*CRC32_FIXED_CHUNK_LEN/8]);
				crc  = __crc32d(crc,  le64_bswap(wp0[0*CRC32_FIXED_CHUNK_LEN/8]));
				crc1 = __crc32d(crc1, le64_bswap(wp0[1*CRC32_FIXED_CHUNK_LEN/8]));
				crc2 = __crc32d(crc2, le64_bswap(wp0[2*CRC32_FIXED_CHUNK_LEN/8]));
				crc3 = __crc32d(crc3, le64_bswap(wp0[3*CRC32_FIXED_CHUNK_LEN/8]));
				wp0++;
				crc  = __crc32d(crc,  le64_bswap(wp0[0*CRC32_FIXED_CHUNK_LEN/8]));
				crc1 = __crc32d(crc1, le64_bswap(wp0[1*CRC32_FIXED_CHUNK_LEN/8]));
				crc2 = __crc32d(crc2, le64_bswap(wp0[2*CRC32_FIXED_CHUNK_LEN/8]));
				crc3 = __crc32d(crc3, le64_bswap(wp0[3*CRC32_FIXED_CHUNK_LEN/8]));
				wp0++;
				crc  = __crc32d(crc,  le64_bswap(wp0[0*CRC32_FIXED_CHUNK_LEN/8]));
				crc1 = __crc32d(crc1, le64_bswap(wp0[1*CRC32_FIXED_CHUNK_LEN/8]));
				crc2 = __crc32d(crc2, le64_bswap(wp0[2*CRC32_FIXED_CHUNK_LEN/8]));
				crc3 = __crc32d(crc3, le64_bswap(wp0[3*CRC32_FIXED_CHUNK_LEN/8]));
				wp0++;
				crc  = __crc32d(crc,  le64_bswap(wp0[0*CRC32_FIXED_CHUNK_LEN/8]));
				crc1 = __crc32d(crc1, le64_bswap(wp0[1*CRC32_FIXED_CHUNK_LEN/8]));
				crc2 = __crc32d(crc2, le64_bswap(wp0[2*CRC32_FIXED_CHUNK_LEN/8]));
				crc3 = __crc32d(crc3, le64_bswap(wp0[3*CRC32_FIXED_CHUNK_LEN/8]));
				wp0++;
			} while (wp0 != wp0_end);
			crc = combine_crcs_slow(crc, crc1, crc2, crc3);
			p += CRC32_NUM_CHUNKS * CRC32_FIXED_CHUNK_LEN;
			len -= CRC32_NUM_CHUNKS * CRC32_FIXED_CHUNK_LEN;
		}
		/*
		 * Due to the large fixed chunk length used above, there might
		 * still be a lot of data left.  So use a 64-byte loop here,
		 * instead of a loop that is less unrolled.
		 */
		while (len >= 64) {
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 0)));
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 8)));
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 16)));
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 24)));
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 32)));
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 40)));
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 48)));
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 56)));
			p += 64;
			len -= 64;
		}
	}
	if (len & 32) {
		crc = __crc32d(crc, get_unaligned_le64(p + 0));
		crc = __crc32d(crc, get_unaligned_le64(p + 8));
		crc = __crc32d(crc, get_unaligned_le64(p + 16));
		crc = __crc32d(crc, get_unaligned_le64(p + 24));
		p += 32;
	}
	if (len & 16) {
		crc = __crc32d(crc, get_unaligned_le64(p + 0));
		crc = __crc32d(crc, get_unaligned_le64(p + 8));
		p += 16;
	}
	if (len & 8) {
		crc = __crc32d(crc, get_unaligned_le64(p));
		p += 8;
	}
	if (len & 4) {
		crc = __crc32w(crc, get_unaligned_le32(p));
		p += 4;
	}
	if (len & 2) {
		crc = __crc32h(crc, get_unaligned_le16(p));
		p += 2;
	}
	if (len & 1)
		crc = __crc32b(crc, *p);
	return crc;
}
#undef ATTRIBUTES
#endif /* crc32_arm_crc() */

/*
 * crc32_arm_crc_pmullcombine() - implementation using crc32 instructions, plus
 *	pmull instructions for CRC combining
 *
 * This is similar to crc32_arm_crc(), but it enables the use of pmull
 * (carryless multiplication) instructions for the steps where the CRCs of
 * adjacent data chunks are combined.  As this greatly speeds up CRC
 * combination, this implementation also differs from crc32_arm_crc() in that it
 * uses a variable chunk length which can get fairly small.  The precomputed
 * multipliers needed for the selected chunk length are loaded from a table.
 *
 * Note that pmull is used here only for combining the CRCs of separately
 * checksummed chunks, not for folding the data itself.  See crc32_arm_pmull*()
 * for implementations that use pmull for folding the data itself.
 */
#if HAVE_CRC32_INTRIN && HAVE_PMULL_INTRIN
#  if HAVE_CRC32_NATIVE && HAVE_PMULL_NATIVE && !USE_PMULL_TARGET_EVEN_IF_NATIVE
#    define ATTRIBUTES
#  else
#    ifdef ARCH_ARM32
#      define ATTRIBUTES	_target_attribute("arch=armv8-a+crc,fpu=crypto-neon-fp-armv8")
#    else
#      ifdef __clang__
#        define ATTRIBUTES	_target_attribute("crc,aes")
#      else
#        define ATTRIBUTES	_target_attribute("+crc,+crypto")
#      endif
#    endif
#  endif

#ifndef _MSC_VER
#  include <arm_acle.h>
#endif
#include <arm_neon.h>

/* Do carryless multiplication of two 32-bit values. */
static forceinline ATTRIBUTES u64
clmul_u32(u32 a, u32 b)
{
	uint64x2_t res = vreinterpretq_u64_p128(
				compat_vmull_p64((poly64_t)a, (poly64_t)b));

	return vgetq_lane_u64(res, 0);
}

/*
 * Like combine_crcs_slow(), but uses vmull_p64 to do the multiplications more
 * quickly, and supports a variable chunk length.  The chunk length is
 * 'i * CRC32_MIN_VARIABLE_CHUNK_LEN'
 * where 1 <= i < ARRAY_LEN(crc32_mults_for_chunklen).
 */
static forceinline ATTRIBUTES u32
combine_crcs_fast(u32 crc0, u32 crc1, u32 crc2, u32 crc3, size_t i)
{
	u64 res0 = clmul_u32(crc0, crc32_mults_for_chunklen[i][0]);
	u64 res1 = clmul_u32(crc1, crc32_mults_for_chunklen[i][1]);
	u64 res2 = clmul_u32(crc2, crc32_mults_for_chunklen[i][2]);

	return __crc32d(0, res0 ^ res1 ^ res2) ^ crc3;
}

#define crc32_arm_crc_pmullcombine	crc32_arm_crc_pmullcombine
static u32 ATTRIBUTES MAYBE_UNUSED
crc32_arm_crc_pmullcombine(u32 crc, const u8 *p, size_t len)
{
	const size_t align = -(uintptr_t)p & 7;

	if (len >= align + CRC32_NUM_CHUNKS * CRC32_MIN_VARIABLE_CHUNK_LEN) {
		/* Align p to the next 8-byte boundary. */
		if (align) {
			if (align & 1)
				crc = __crc32b(crc, *p++);
			if (align & 2) {
				crc = __crc32h(crc, le16_bswap(*(u16 *)p));
				p += 2;
			}
			if (align & 4) {
				crc = __crc32w(crc, le32_bswap(*(u32 *)p));
				p += 4;
			}
			len -= align;
		}
		/*
		 * Handle CRC32_MAX_VARIABLE_CHUNK_LEN specially, so that better
		 * code is generated for it.
		 */
		while (len >= CRC32_NUM_CHUNKS * CRC32_MAX_VARIABLE_CHUNK_LEN) {
			const u64 *wp0 = (const u64 *)p;
			const u64 * const wp0_end =
				(const u64 *)(p + CRC32_MAX_VARIABLE_CHUNK_LEN);
			u32 crc1 = 0, crc2 = 0, crc3 = 0;

			STATIC_ASSERT(CRC32_NUM_CHUNKS == 4);
			STATIC_ASSERT(CRC32_MAX_VARIABLE_CHUNK_LEN % (4 * 8) == 0);
			do {
				prefetchr(&wp0[64 + 0*CRC32_MAX_VARIABLE_CHUNK_LEN/8]);
				prefetchr(&wp0[64 + 1*CRC32_MAX_VARIABLE_CHUNK_LEN/8]);
				prefetchr(&wp0[64 + 2*CRC32_MAX_VARIABLE_CHUNK_LEN/8]);
				prefetchr(&wp0[64 + 3*CRC32_MAX_VARIABLE_CHUNK_LEN/8]);
				crc  = __crc32d(crc,  le64_bswap(wp0[0*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc1 = __crc32d(crc1, le64_bswap(wp0[1*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc2 = __crc32d(crc2, le64_bswap(wp0[2*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc3 = __crc32d(crc3, le64_bswap(wp0[3*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				wp0++;
				crc  = __crc32d(crc,  le64_bswap(wp0[0*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc1 = __crc32d(crc1, le64_bswap(wp0[1*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc2 = __crc32d(crc2, le64_bswap(wp0[2*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc3 = __crc32d(crc3, le64_bswap(wp0[3*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				wp0++;
				crc  = __crc32d(crc,  le64_bswap(wp0[0*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc1 = __crc32d(crc1, le64_bswap(wp0[1*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc2 = __crc32d(crc2, le64_bswap(wp0[2*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc3 = __crc32d(crc3, le64_bswap(wp0[3*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				wp0++;
				crc  = __crc32d(crc,  le64_bswap(wp0[0*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc1 = __crc32d(crc1, le64_bswap(wp0[1*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc2 = __crc32d(crc2, le64_bswap(wp0[2*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				crc3 = __crc32d(crc3, le64_bswap(wp0[3*CRC32_MAX_VARIABLE_CHUNK_LEN/8]));
				wp0++;
			} while (wp0 != wp0_end);
			crc = combine_crcs_fast(crc, crc1, crc2, crc3,
						ARRAY_LEN(crc32_mults_for_chunklen) - 1);
			p += CRC32_NUM_CHUNKS * CRC32_MAX_VARIABLE_CHUNK_LEN;
			len -= CRC32_NUM_CHUNKS * CRC32_MAX_VARIABLE_CHUNK_LEN;
		}
		/* Handle up to one variable-length chunk. */
		if (len >= CRC32_NUM_CHUNKS * CRC32_MIN_VARIABLE_CHUNK_LEN) {
			const size_t i = len / (CRC32_NUM_CHUNKS *
						CRC32_MIN_VARIABLE_CHUNK_LEN);
			const size_t chunk_len =
				i * CRC32_MIN_VARIABLE_CHUNK_LEN;
			const u64 *wp0 = (const u64 *)(p + 0*chunk_len);
			const u64 *wp1 = (const u64 *)(p + 1*chunk_len);
			const u64 *wp2 = (const u64 *)(p + 2*chunk_len);
			const u64 *wp3 = (const u64 *)(p + 3*chunk_len);
			const u64 * const wp0_end = wp1;
			u32 crc1 = 0, crc2 = 0, crc3 = 0;

			STATIC_ASSERT(CRC32_NUM_CHUNKS == 4);
			STATIC_ASSERT(CRC32_MIN_VARIABLE_CHUNK_LEN % (4 * 8) == 0);
			do {
				prefetchr(wp0 + 64);
				prefetchr(wp1 + 64);
				prefetchr(wp2 + 64);
				prefetchr(wp3 + 64);
				crc  = __crc32d(crc,  le64_bswap(*wp0++));
				crc1 = __crc32d(crc1, le64_bswap(*wp1++));
				crc2 = __crc32d(crc2, le64_bswap(*wp2++));
				crc3 = __crc32d(crc3, le64_bswap(*wp3++));
				crc  = __crc32d(crc,  le64_bswap(*wp0++));
				crc1 = __crc32d(crc1, le64_bswap(*wp1++));
				crc2 = __crc32d(crc2, le64_bswap(*wp2++));
				crc3 = __crc32d(crc3, le64_bswap(*wp3++));
				crc  = __crc32d(crc,  le64_bswap(*wp0++));
				crc1 = __crc32d(crc1, le64_bswap(*wp1++));
				crc2 = __crc32d(crc2, le64_bswap(*wp2++));
				crc3 = __crc32d(crc3, le64_bswap(*wp3++));
				crc  = __crc32d(crc,  le64_bswap(*wp0++));
				crc1 = __crc32d(crc1, le64_bswap(*wp1++));
				crc2 = __crc32d(crc2, le64_bswap(*wp2++));
				crc3 = __crc32d(crc3, le64_bswap(*wp3++));
			} while (wp0 != wp0_end);
			crc = combine_crcs_fast(crc, crc1, crc2, crc3, i);
			p += CRC32_NUM_CHUNKS * chunk_len;
			len -= CRC32_NUM_CHUNKS * chunk_len;
		}

		while (len >= 32) {
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 0)));
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 8)));
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 16)));
			crc = __crc32d(crc, le64_bswap(*(u64 *)(p + 24)));
			p += 32;
			len -= 32;
		}
	} else {
		while (len >= 32) {
			crc = __crc32d(crc, get_unaligned_le64(p + 0));
			crc = __crc32d(crc, get_unaligned_le64(p + 8));
			crc = __crc32d(crc, get_unaligned_le64(p + 16));
			crc = __crc32d(crc, get_unaligned_le64(p + 24));
			p += 32;
			len -= 32;
		}
	}
	if (len & 16) {
		crc = __crc32d(crc, get_unaligned_le64(p + 0));
		crc = __crc32d(crc, get_unaligned_le64(p + 8));
		p += 16;
	}
	if (len & 8) {
		crc = __crc32d(crc, get_unaligned_le64(p));
		p += 8;
	}
	if (len & 4) {
		crc = __crc32w(crc, get_unaligned_le32(p));
		p += 4;
	}
	if (len & 2) {
		crc = __crc32h(crc, get_unaligned_le16(p));
		p += 2;
	}
	if (len & 1)
		crc = __crc32b(crc, *p);
	return crc;
}
#undef ATTRIBUTES
#endif /* crc32_arm_crc_pmullcombine() */

/*
 * crc32_arm_pmullx4() - implementation using "folding" with pmull instructions
 *
 * This implementation is intended for CPUs that support pmull instructions but
 * not crc32 instructions.
 */
#if HAVE_PMULL_INTRIN
#  define crc32_arm_pmullx4	crc32_arm_pmullx4
#  define SUFFIX			 _pmullx4
#  if HAVE_PMULL_NATIVE && !USE_PMULL_TARGET_EVEN_IF_NATIVE
#    define ATTRIBUTES
#  else
#    ifdef ARCH_ARM32
#      define ATTRIBUTES    _target_attribute("fpu=crypto-neon-fp-armv8")
#    else
#      ifdef __clang__
	 /*
	  * This used to use "crypto", but that stopped working with clang 16.
	  * Now only "aes" works.  "aes" works with older versions too, so use
	  * that.  No "+" prefix; clang 15 and earlier doesn't accept that.
	  */
#        define ATTRIBUTES  _target_attribute("aes")
#      else
	 /*
	  * With gcc, only "+crypto" works.  Both the "+" prefix and the
	  * "crypto" (not "aes") are essential...
	  */
#        define ATTRIBUTES  _target_attribute("+crypto")
#      endif
#    endif
#  endif
#  define ENABLE_EOR3		0
#  include "crc32_pmull_helpers.h"

static u32 ATTRIBUTES MAYBE_UNUSED
crc32_arm_pmullx4(u32 crc, const u8 *p, size_t len)
{
	static const u64 _aligned_attribute(16) mults[3][2] = {
		CRC32_1VECS_MULTS,
		CRC32_4VECS_MULTS,
		CRC32_2VECS_MULTS,
	};
	static const u64 _aligned_attribute(16) final_mults[3][2] = {
		{ CRC32_FINAL_MULT, 0 },
		{ CRC32_BARRETT_CONSTANT_1, 0 },
		{ CRC32_BARRETT_CONSTANT_2, 0 },
	};
	const uint8x16_t zeroes = vdupq_n_u8(0);
	const uint8x16_t mask32 = vreinterpretq_u8_u64(vdupq_n_u64(0xFFFFFFFF));
	const poly64x2_t multipliers_1 = load_multipliers(mults[0]);
	uint8x16_t v0, v1, v2, v3;

	if (len < 64 + 15) {
		if (len < 16)
			return crc32_slice1(crc, p, len);
		v0 = veorq_u8(vld1q_u8(p), u32_to_bytevec(crc));
		p += 16;
		len -= 16;
		while (len >= 16) {
			v0 = fold_vec(v0, vld1q_u8(p), multipliers_1);
			p += 16;
			len -= 16;
		}
	} else {
		const poly64x2_t multipliers_4 = load_multipliers(mults[1]);
		const poly64x2_t multipliers_2 = load_multipliers(mults[2]);
		const size_t align = -(uintptr_t)p & 15;
		const uint8x16_t *vp;

		v0 = veorq_u8(vld1q_u8(p), u32_to_bytevec(crc));
		p += 16;
		/* Align p to the next 16-byte boundary. */
		if (align) {
			v0 = fold_partial_vec(v0, p, align, multipliers_1);
			p += align;
			len -= align;
		}
		vp = (const uint8x16_t *)p;
		v1 = *vp++;
		v2 = *vp++;
		v3 = *vp++;
		while (len >= 64 + 64) {
			v0 = fold_vec(v0, *vp++, multipliers_4);
			v1 = fold_vec(v1, *vp++, multipliers_4);
			v2 = fold_vec(v2, *vp++, multipliers_4);
			v3 = fold_vec(v3, *vp++, multipliers_4);
			len -= 64;
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
		len &= 15;
	}

	/* Handle any remaining partial block now before reducing to 32 bits. */
	if (len)
		v0 = fold_partial_vec(v0, p, len, multipliers_1);

	/*
	 * Fold 128 => 96 bits.  This also implicitly appends 32 zero bits,
	 * which is equivalent to multiplying by x^32.  This is needed because
	 * the CRC is defined as M(x)*x^32 mod G(x), not just M(x) mod G(x).
	 */

	v0 = veorq_u8(vextq_u8(v0, zeroes, 8),
		      clmul_high(vextq_u8(zeroes, v0, 8), multipliers_1));

	/* Fold 96 => 64 bits. */
	v0 = veorq_u8(vextq_u8(v0, zeroes, 4),
		      clmul_low(vandq_u8(v0, mask32),
				load_multipliers(final_mults[0])));

	/* Reduce 64 => 32 bits using Barrett reduction. */
	v1 = clmul_low(vandq_u8(v0, mask32), load_multipliers(final_mults[1]));
	v1 = clmul_low(vandq_u8(v1, mask32), load_multipliers(final_mults[2]));
	return vgetq_lane_u32(vreinterpretq_u32_u8(veorq_u8(v0, v1)), 1);
}
#undef SUFFIX
#undef ATTRIBUTES
#undef ENABLE_EOR3
#endif /* crc32_arm_pmullx4() */

/*
 * crc32_arm_pmullx12_crc() - large-stride implementation using "folding" with
 *	pmull instructions, where crc32 instructions are also available
 *
 * See crc32_pmull_wide.h for explanation.
 */
#if defined(ARCH_ARM64) && HAVE_PMULL_INTRIN && HAVE_CRC32_INTRIN
#  define crc32_arm_pmullx12_crc	crc32_arm_pmullx12_crc
#  define SUFFIX				 _pmullx12_crc
#  if HAVE_PMULL_NATIVE && HAVE_CRC32_NATIVE && !USE_PMULL_TARGET_EVEN_IF_NATIVE
#    define ATTRIBUTES
#  else
#    ifdef __clang__
#      define ATTRIBUTES  _target_attribute("aes,crc")
#    else
#      define ATTRIBUTES  _target_attribute("+crypto,+crc")
#    endif
#  endif
#  define ENABLE_EOR3	0
#  include "crc32_pmull_wide.h"
#endif

/*
 * crc32_arm_pmullx12_crc_eor3()
 *
 * This like crc32_arm_pmullx12_crc(), but it adds the eor3 instruction (from
 * the sha3 extension) for even better performance.
 *
 * Note: we require HAVE_SHA3_TARGET (or HAVE_SHA3_NATIVE) rather than
 * HAVE_SHA3_INTRIN, as we have an inline asm fallback for eor3.
 */
#if defined(ARCH_ARM64) && HAVE_PMULL_INTRIN && HAVE_CRC32_INTRIN && \
	(HAVE_SHA3_TARGET || HAVE_SHA3_NATIVE)
#  define crc32_arm_pmullx12_crc_eor3	crc32_arm_pmullx12_crc_eor3
#  define SUFFIX				 _pmullx12_crc_eor3
#  if HAVE_PMULL_NATIVE && HAVE_CRC32_NATIVE && HAVE_SHA3_NATIVE && \
	!USE_PMULL_TARGET_EVEN_IF_NATIVE
#    define ATTRIBUTES
#  else
#    ifdef __clang__
#      define ATTRIBUTES  _target_attribute("aes,crc,sha3")
     /*
      * With gcc, arch=armv8.2-a is needed for the sha3 intrinsics, unless the
      * default target is armv8.3-a or later in which case it must be omitted.
      * armv8.3-a or later can be detected by checking for __ARM_FEATURE_JCVT.
      */
#    elif defined(__ARM_FEATURE_JCVT)
#      define ATTRIBUTES  _target_attribute("+crypto,+crc,+sha3")
#    else
#      define ATTRIBUTES  _target_attribute("arch=armv8.2-a+crypto+crc+sha3")
#    endif
#  endif
#  define ENABLE_EOR3	1
#  include "crc32_pmull_wide.h"
#endif

/*
 * On the Apple M1 processor, crc32 instructions max out at about 25.5 GB/s in
 * the best case of using a 3-way or greater interleaved chunked implementation,
 * whereas a pmull-based implementation achieves 68 GB/s provided that the
 * stride length is large enough (about 10+ vectors with eor3, or 12+ without).
 *
 * For now we assume that crc32 instructions are preferable in other cases.
 */
#define PREFER_PMULL_TO_CRC	0
#ifdef __APPLE__
#  include <TargetConditionals.h>
#  if TARGET_OS_OSX
#    undef PREFER_PMULL_TO_CRC
#    define PREFER_PMULL_TO_CRC	1
#  endif
#endif

/*
 * If the best implementation is statically available, use it unconditionally.
 * Otherwise choose the best implementation at runtime.
 */
#if PREFER_PMULL_TO_CRC && defined(crc32_arm_pmullx12_crc_eor3) && \
	HAVE_PMULL_NATIVE && HAVE_CRC32_NATIVE && HAVE_SHA3_NATIVE
#  define DEFAULT_IMPL	crc32_arm_pmullx12_crc_eor3
#elif !PREFER_PMULL_TO_CRC && defined(crc32_arm_crc_pmullcombine) && \
	HAVE_CRC32_NATIVE && HAVE_PMULL_NATIVE
#  define DEFAULT_IMPL	crc32_arm_crc_pmullcombine
#else
static inline crc32_func_t
arch_select_crc32_func(void)
{
	const u32 features MAYBE_UNUSED = get_arm_cpu_features();

#if PREFER_PMULL_TO_CRC && defined(crc32_arm_pmullx12_crc_eor3)
	if (HAVE_PMULL(features) && HAVE_CRC32(features) && HAVE_SHA3(features))
		return crc32_arm_pmullx12_crc_eor3;
#endif
#if PREFER_PMULL_TO_CRC && defined(crc32_arm_pmullx12_crc)
	if (HAVE_PMULL(features) && HAVE_CRC32(features))
		return crc32_arm_pmullx12_crc;
#endif
#ifdef crc32_arm_crc_pmullcombine
	if (HAVE_CRC32(features) && HAVE_PMULL(features))
		return crc32_arm_crc_pmullcombine;
#endif
#ifdef crc32_arm_crc
	if (HAVE_CRC32(features))
		return crc32_arm_crc;
#endif
#ifdef crc32_arm_pmullx4
	if (HAVE_PMULL(features))
		return crc32_arm_pmullx4;
#endif
	return NULL;
}
#define arch_select_crc32_func	arch_select_crc32_func
#endif

#endif /* LIB_ARM_CRC32_IMPL_H */
