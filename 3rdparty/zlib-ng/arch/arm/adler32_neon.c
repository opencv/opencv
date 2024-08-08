/* Copyright (C) 1995-2011, 2016 Mark Adler
 * Copyright (C) 2017 ARM Holdings Inc.
 * Authors:
 *   Adenilson Cavalcanti <adenilson.cavalcanti@arm.com>
 *   Adam Stylinski <kungfujesus06@gmail.com>
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#ifdef ARM_NEON
#include "neon_intrins.h"
#include "../../zbuild.h"
#include "../../adler32_p.h"

static void NEON_accum32(uint32_t *s, const uint8_t *buf, size_t len) {
    static const uint16_t ALIGNED_(16) taps[64] = {
        64, 63, 62, 61, 60, 59, 58, 57,
        56, 55, 54, 53, 52, 51, 50, 49,
        48, 47, 46, 45, 44, 43, 42, 41,
        40, 39, 38, 37, 36, 35, 34, 33,
        32, 31, 30, 29, 28, 27, 26, 25,
        24, 23, 22, 21, 20, 19, 18, 17,
        16, 15, 14, 13, 12, 11, 10, 9,
        8, 7, 6, 5, 4, 3, 2, 1 };

    uint32x4_t adacc = vdupq_n_u32(0);
    uint32x4_t s2acc = vdupq_n_u32(0);
    uint32x4_t s2acc_0 = vdupq_n_u32(0);
    uint32x4_t s2acc_1 = vdupq_n_u32(0);
    uint32x4_t s2acc_2 = vdupq_n_u32(0);

    adacc = vsetq_lane_u32(s[0], adacc, 0);
    s2acc = vsetq_lane_u32(s[1], s2acc, 0);

    uint32x4_t s3acc = vdupq_n_u32(0);
    uint32x4_t adacc_prev = adacc;

    uint16x8_t s2_0, s2_1, s2_2, s2_3;
    s2_0 = s2_1 = s2_2 = s2_3 = vdupq_n_u16(0);

    uint16x8_t s2_4, s2_5, s2_6, s2_7;
    s2_4 = s2_5 = s2_6 = s2_7 = vdupq_n_u16(0);

    size_t num_iter = len >> 2;
    int rem = len & 3;

    for (size_t i = 0; i < num_iter; ++i) {
        uint8x16x4_t d0_d3 = vld1q_u8_x4(buf);

        /* Unfortunately it doesn't look like there's a direct sum 8 bit to 32
         * bit instruction, we'll have to make due summing to 16 bits first */
        uint16x8x2_t hsum, hsum_fold;
        hsum.val[0] = vpaddlq_u8(d0_d3.val[0]);
        hsum.val[1] = vpaddlq_u8(d0_d3.val[1]);

        hsum_fold.val[0] = vpadalq_u8(hsum.val[0], d0_d3.val[2]);
        hsum_fold.val[1] = vpadalq_u8(hsum.val[1], d0_d3.val[3]);

        adacc = vpadalq_u16(adacc, hsum_fold.val[0]);
        s3acc = vaddq_u32(s3acc, adacc_prev);
        adacc = vpadalq_u16(adacc, hsum_fold.val[1]);

        /* If we do straight widening additions to the 16 bit values, we don't incur
         * the usual penalties of a pairwise add. We can defer the multiplications
         * until the very end. These will not overflow because we are incurring at
         * most 408 loop iterations (NMAX / 64), and a given lane is only going to be
         * summed into once. This means for the maximum input size, the largest value
         * we will see is 255 * 102 = 26010, safely under uint16 max */
        s2_0 = vaddw_u8(s2_0, vget_low_u8(d0_d3.val[0]));
        s2_1 = vaddw_high_u8(s2_1, d0_d3.val[0]);
        s2_2 = vaddw_u8(s2_2, vget_low_u8(d0_d3.val[1]));
        s2_3 = vaddw_high_u8(s2_3, d0_d3.val[1]);
        s2_4 = vaddw_u8(s2_4, vget_low_u8(d0_d3.val[2]));
        s2_5 = vaddw_high_u8(s2_5, d0_d3.val[2]);
        s2_6 = vaddw_u8(s2_6, vget_low_u8(d0_d3.val[3]));
        s2_7 = vaddw_high_u8(s2_7, d0_d3.val[3]);

        adacc_prev = adacc;
        buf += 64;
    }

    s3acc = vshlq_n_u32(s3acc, 6);

    if (rem) {
        uint32x4_t s3acc_0 = vdupq_n_u32(0);
        while (rem--) {
            uint8x16_t d0 = vld1q_u8(buf);
            uint16x8_t adler;
            adler = vpaddlq_u8(d0);
            s2_6 = vaddw_u8(s2_6, vget_low_u8(d0));
            s2_7 = vaddw_high_u8(s2_7, d0);
            adacc = vpadalq_u16(adacc, adler);
            s3acc_0 = vaddq_u32(s3acc_0, adacc_prev);
            adacc_prev = adacc;
            buf += 16;
        }

        s3acc_0 = vshlq_n_u32(s3acc_0, 4);
        s3acc = vaddq_u32(s3acc_0, s3acc);
    }

    uint16x8x4_t t0_t3 = vld1q_u16_x4(taps);
    uint16x8x4_t t4_t7 = vld1q_u16_x4(taps + 32);

    s2acc = vmlal_high_u16(s2acc, t0_t3.val[0], s2_0);
    s2acc_0 = vmlal_u16(s2acc_0, vget_low_u16(t0_t3.val[0]), vget_low_u16(s2_0));
    s2acc_1 = vmlal_high_u16(s2acc_1, t0_t3.val[1], s2_1);
    s2acc_2 = vmlal_u16(s2acc_2, vget_low_u16(t0_t3.val[1]), vget_low_u16(s2_1));

    s2acc = vmlal_high_u16(s2acc, t0_t3.val[2], s2_2);
    s2acc_0 = vmlal_u16(s2acc_0, vget_low_u16(t0_t3.val[2]), vget_low_u16(s2_2));
    s2acc_1 = vmlal_high_u16(s2acc_1, t0_t3.val[3], s2_3);
    s2acc_2 = vmlal_u16(s2acc_2, vget_low_u16(t0_t3.val[3]), vget_low_u16(s2_3));

    s2acc = vmlal_high_u16(s2acc, t4_t7.val[0], s2_4);
    s2acc_0 = vmlal_u16(s2acc_0, vget_low_u16(t4_t7.val[0]), vget_low_u16(s2_4));
    s2acc_1 = vmlal_high_u16(s2acc_1, t4_t7.val[1], s2_5);
    s2acc_2 = vmlal_u16(s2acc_2, vget_low_u16(t4_t7.val[1]), vget_low_u16(s2_5));

    s2acc = vmlal_high_u16(s2acc, t4_t7.val[2], s2_6);
    s2acc_0 = vmlal_u16(s2acc_0, vget_low_u16(t4_t7.val[2]), vget_low_u16(s2_6));
    s2acc_1 = vmlal_high_u16(s2acc_1, t4_t7.val[3], s2_7);
    s2acc_2 = vmlal_u16(s2acc_2, vget_low_u16(t4_t7.val[3]), vget_low_u16(s2_7));

    s2acc = vaddq_u32(s2acc_0, s2acc);
    s2acc_2 = vaddq_u32(s2acc_1, s2acc_2);
    s2acc = vaddq_u32(s2acc, s2acc_2);

    uint32x2_t adacc2, s2acc2, as;
    s2acc = vaddq_u32(s2acc, s3acc);
    adacc2 = vpadd_u32(vget_low_u32(adacc), vget_high_u32(adacc));
    s2acc2 = vpadd_u32(vget_low_u32(s2acc), vget_high_u32(s2acc));
    as = vpadd_u32(adacc2, s2acc2);
    s[0] = vget_lane_u32(as, 0);
    s[1] = vget_lane_u32(as, 1);
}

static void NEON_handle_tail(uint32_t *pair, const uint8_t *buf, size_t len) {
    unsigned int i;
    for (i = 0; i < len; ++i) {
        pair[0] += buf[i];
        pair[1] += pair[0];
    }
}

Z_INTERNAL uint32_t adler32_neon(uint32_t adler, const uint8_t *buf, size_t len) {
    /* split Adler-32 into component sums */
    uint32_t sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if (len == 1)
        return adler32_len_1(adler, buf, sum2);

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (buf == NULL)
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (len < 16)
        return adler32_len_16(adler, buf, len, sum2);

    uint32_t pair[2];
    int n = NMAX;
    unsigned int done = 0;

    /* Split Adler-32 into component sums, it can be supplied by
     * the caller sites (e.g. in a PNG file).
     */
    pair[0] = adler;
    pair[1] = sum2;

    /* If memory is not SIMD aligned, do scalar sums to an aligned
     * offset, provided that doing so doesn't completely eliminate
     * SIMD operation. Aligned loads are still faster on ARM, even
     * though there's no explicit aligned load instruction */
    unsigned int align_offset = ((uintptr_t)buf & 15);
    unsigned int align_adj = (align_offset) ? 16 - align_offset : 0;

    if (align_offset && len >= (16 + align_adj)) {
        NEON_handle_tail(pair, buf, align_adj);
        n -= align_adj;
        done += align_adj;

    } else {
        /* If here, we failed the len criteria test, it wouldn't be
         * worthwhile to do scalar aligning sums */
        align_adj = 0;
    }

    while (done < len) {
        int remaining = (int)(len - done);
        n = MIN(remaining, (done == align_adj) ? n : NMAX);

        if (n < 16)
            break;

        NEON_accum32(pair, buf + done, n >> 4);
        pair[0] %= BASE;
        pair[1] %= BASE;

        int actual_nsums = (n >> 4) << 4;
        done += actual_nsums;
    }

    /* Handle the tail elements. */
    if (done < len) {
        NEON_handle_tail(pair, (buf + done), len - done);
        pair[0] %= BASE;
        pair[1] %= BASE;
    }

    /* D = B * 65536 + A, see: https://en.wikipedia.org/wiki/Adler-32. */
    return (pair[1] << 16) | pair[0];
}

#endif
