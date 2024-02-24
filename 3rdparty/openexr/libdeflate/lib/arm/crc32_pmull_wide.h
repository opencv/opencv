/*
 * arm/crc32_pmull_wide.h - gzip CRC-32 with PMULL (extra-wide version)
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
 * This file is a "template" for instantiating PMULL-based crc32_arm functions.
 * The "parameters" are:
 *
 * SUFFIX:
 *	Name suffix to append to all instantiated functions.
 * ATTRIBUTES:
 *	Target function attributes to use.
 * ENABLE_EOR3:
 *	Use the eor3 instruction (from the sha3 extension).
 *
 * This is the extra-wide version; it uses an unusually large stride length of
 * 12, and it assumes that crc32 instructions are available too.  It's intended
 * for powerful CPUs that support both pmull and crc32 instructions, but where
 * throughput of pmull and xor (given enough instructions issued in parallel) is
 * significantly higher than that of crc32, thus making the crc32 instructions
 * (counterintuitively) not actually the fastest way to compute the CRC-32.  The
 * Apple M1 processor is an example of such a CPU.
 */

#ifndef _MSC_VER
#  include <arm_acle.h>
#endif
#include <arm_neon.h>

#include "crc32_pmull_helpers.h"

static u32 ATTRIBUTES MAYBE_UNUSED
ADD_SUFFIX(crc32_arm)(u32 crc, const u8 *p, size_t len)
{
	uint8x16_t v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11;

	if (len < 3 * 192) {
		static const u64 _aligned_attribute(16) mults[3][2] = {
			CRC32_4VECS_MULTS, CRC32_2VECS_MULTS, CRC32_1VECS_MULTS,
		};
		poly64x2_t multipliers_4, multipliers_2, multipliers_1;

		if (len < 64)
			goto tail;
		multipliers_4 = load_multipliers(mults[0]);
		multipliers_2 = load_multipliers(mults[1]);
		multipliers_1 = load_multipliers(mults[2]);
		/*
		 * Short length; don't bother aligning the pointer, and fold
		 * 64 bytes (4 vectors) at a time, at most.
		 */
		v0 = veorq_u8(vld1q_u8(p + 0), u32_to_bytevec(crc));
		v1 = vld1q_u8(p + 16);
		v2 = vld1q_u8(p + 32);
		v3 = vld1q_u8(p + 48);
		p += 64;
		len -= 64;
		while (len >= 64) {
			v0 = fold_vec(v0, vld1q_u8(p + 0), multipliers_4);
			v1 = fold_vec(v1, vld1q_u8(p + 16), multipliers_4);
			v2 = fold_vec(v2, vld1q_u8(p + 32), multipliers_4);
			v3 = fold_vec(v3, vld1q_u8(p + 48), multipliers_4);
			p += 64;
			len -= 64;
		}
		v0 = fold_vec(v0, v2, multipliers_2);
		v1 = fold_vec(v1, v3, multipliers_2);
		if (len >= 32) {
			v0 = fold_vec(v0, vld1q_u8(p + 0), multipliers_2);
			v1 = fold_vec(v1, vld1q_u8(p + 16), multipliers_2);
			p += 32;
			len -= 32;
		}
		v0 = fold_vec(v0, v1, multipliers_1);
	} else {
		static const u64 _aligned_attribute(16) mults[4][2] = {
			CRC32_12VECS_MULTS, CRC32_6VECS_MULTS,
			CRC32_3VECS_MULTS, CRC32_1VECS_MULTS,
		};
		const poly64x2_t multipliers_12 = load_multipliers(mults[0]);
		const poly64x2_t multipliers_6 = load_multipliers(mults[1]);
		const poly64x2_t multipliers_3 = load_multipliers(mults[2]);
		const poly64x2_t multipliers_1 = load_multipliers(mults[3]);
		const size_t align = -(uintptr_t)p & 15;
		const uint8x16_t *vp;

		/* Align p to the next 16-byte boundary. */
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
			if (align & 8) {
				crc = __crc32d(crc, le64_bswap(*(u64 *)p));
				p += 8;
			}
			len -= align;
		}
		vp = (const uint8x16_t *)p;
		v0 = veorq_u8(*vp++, u32_to_bytevec(crc));
		v1 = *vp++;
		v2 = *vp++;
		v3 = *vp++;
		v4 = *vp++;
		v5 = *vp++;
		v6 = *vp++;
		v7 = *vp++;
		v8 = *vp++;
		v9 = *vp++;
		v10 = *vp++;
		v11 = *vp++;
		len -= 192;
		/* Fold 192 bytes (12 vectors) at a time. */
		do {
			v0 = fold_vec(v0, *vp++, multipliers_12);
			v1 = fold_vec(v1, *vp++, multipliers_12);
			v2 = fold_vec(v2, *vp++, multipliers_12);
			v3 = fold_vec(v3, *vp++, multipliers_12);
			v4 = fold_vec(v4, *vp++, multipliers_12);
			v5 = fold_vec(v5, *vp++, multipliers_12);
			v6 = fold_vec(v6, *vp++, multipliers_12);
			v7 = fold_vec(v7, *vp++, multipliers_12);
			v8 = fold_vec(v8, *vp++, multipliers_12);
			v9 = fold_vec(v9, *vp++, multipliers_12);
			v10 = fold_vec(v10, *vp++, multipliers_12);
			v11 = fold_vec(v11, *vp++, multipliers_12);
			len -= 192;
		} while (len >= 192);

		/*
		 * Fewer than 192 bytes left.  Fold v0-v11 down to just v0,
		 * while processing up to 144 more bytes.
		 */
		v0 = fold_vec(v0, v6, multipliers_6);
		v1 = fold_vec(v1, v7, multipliers_6);
		v2 = fold_vec(v2, v8, multipliers_6);
		v3 = fold_vec(v3, v9, multipliers_6);
		v4 = fold_vec(v4, v10, multipliers_6);
		v5 = fold_vec(v5, v11, multipliers_6);
		if (len >= 96) {
			v0 = fold_vec(v0, *vp++, multipliers_6);
			v1 = fold_vec(v1, *vp++, multipliers_6);
			v2 = fold_vec(v2, *vp++, multipliers_6);
			v3 = fold_vec(v3, *vp++, multipliers_6);
			v4 = fold_vec(v4, *vp++, multipliers_6);
			v5 = fold_vec(v5, *vp++, multipliers_6);
			len -= 96;
		}
		v0 = fold_vec(v0, v3, multipliers_3);
		v1 = fold_vec(v1, v4, multipliers_3);
		v2 = fold_vec(v2, v5, multipliers_3);
		if (len >= 48) {
			v0 = fold_vec(v0, *vp++, multipliers_3);
			v1 = fold_vec(v1, *vp++, multipliers_3);
			v2 = fold_vec(v2, *vp++, multipliers_3);
			len -= 48;
		}
		v0 = fold_vec(v0, v1, multipliers_1);
		v0 = fold_vec(v0, v2, multipliers_1);
		p = (const u8 *)vp;
	}
	/* Reduce 128 to 32 bits using crc32 instructions. */
	crc = __crc32d(0, vgetq_lane_u64(vreinterpretq_u64_u8(v0), 0));
	crc = __crc32d(crc, vgetq_lane_u64(vreinterpretq_u64_u8(v0), 1));
tail:
	/* Finish up the remainder using crc32 instructions. */
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

#undef SUFFIX
#undef ATTRIBUTES
#undef ENABLE_EOR3
